from __future__ import division

from models import *
from utils.utils import *
from utils.datasets import *

import os
import sys
import time
import datetime
import argparse
import pyclipper
import json
import shapely.geometry as sg
from PIL import Image, ImageDraw

import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable


layersize = [416, 208, 208, 104, 104, 104, 52, 52, 52, 52, 52, 52, 52, 52, 52, 26, 26, 26, 26, 26, 
             26, 26, 26, 26, 13, 13, 13, 13, 13, 13, 13, 13, 26, 26, 26, 52, 52, 52]

layer_index = np.zeros(38)
for j in range(30):
    layer_index[j] = j+1
layer_index[30] = 32
layer_index[31] = 34
layer_index[32] = 37
layer_index[33] = 39
layer_index[34] = 41
layer_index[35] = 44
layer_index[36] = 46
layer_index[37] = 48
layer_index = layer_index.astype('int32')
layer_type = [1, 3, 1, 3, 1, 1, 3, 1, 1, 1, 1, 1, 1, 1, 1, 3, 
              1, 1, 1, 1, 1, 1, 1, 1, 3, 1, 1, 1, 1, 0, 1, 1, 4, 1, 1, 4, 1, 1]
layers_idx = [0, 1, 4, 5, 8, 11, 12, 15, 18, 21, 24, 27, 30, 33, 36, 37, 40, 
             43, 46, 49, 52, 55, 58, 61, 62, 65, 68, 71, 74, 75, 77, 79, 87, 89, 91, 99, 101, 103]
def shrink(bboxes, offset):
    shrinked_bboxes = []
    for bbox in bboxes:
        pco = pyclipper.PyclipperOffset()
        pco.AddPath(bbox, pyclipper.JT_MITER, pyclipper.ET_CLOSEDPOLYGON)
        shrinked_bbox = pco.Execute(offset)
        if len(shrinked_bbox) > 0:
            shrinked_bbox = np.array(shrinked_bbox)[0]
            shrinked_bbox = np.asarray(shrinked_bbox)
            shrinked_bbox = shrinked_bbox.ravel().tolist()
            shrinked_bboxes.append(shrinked_bbox)
    return shrinked_bboxes


def RoI_extension(RoIs, rate):
    def dist(a, b):
        return np.sqrt(np.sum((a - b) ** 2))

    def perimeter(bbox):
        peri = 0.0
        for i in range(bbox.shape[0]):
            peri += dist(bbox[i], bbox[(i + 1) % bbox.shape[0]])
        return peri
    _RoIs = []
    for box in RoIs:
        box_info = np.asarray(box).reshape((-1, 2))
        _RoIs.append(box_info)
    rate = rate * rate
    new_RoIs = []
    for box in _RoIs:
        area = sg.Polygon(box).area
        peri = perimeter(box)
        offset = int(area * (1 - rate) / (peri + 0.0001) + 0.5)
        pco = pyclipper.PyclipperOffset()
        pco.AddPath(box, pyclipper.JT_MITER, pyclipper.ET_CLOSEDPOLYGON)
        new_box = pco.Execute(offset)
        if len(new_box) > 0:
            new_box = np.asarray(new_box)[0]
            new_box = np.asarray(new_box)
            new_box = new_box.ravel().tolist()
            new_RoIs.append(new_box)
    return new_RoIs

def RoI_for_layers(RoI):
    # input RoI is the region of interest for the raw frame
    # this function is to calculate the RoI for padding function in each layer
    RoIs = []
    RoIs_padded = []
    temp_RoI = []   # to reshape
    for i in range(38):
        if i != 29 and i != 32 and i != 35:
            gt_boxes = []
            for temp in RoI:
                temp_RoI = np.asarray(temp).reshape(-1, 2)
                gt_boxes.append(temp_RoI)
            new_gt_boxes = shrink(gt_boxes, -1)
            if layer_type[i] == 1:
                # offset = -1
                RoIs.append(new_gt_boxes)

            elif layer_type[i] == 3:
            #
                for _list in new_gt_boxes:
                    _list[:] = [int(x / 2) for x in _list]
                RoIs.append(new_gt_boxes)
        elif i == 29:
            new_gt_boxes = RoI
            RoIs.append(new_gt_boxes)
        elif i == 32:
            new_gt_boxes = RoIs[23]
            RoIs.append(new_gt_boxes)
        elif i == 35:
            new_gt_boxes = RoIs[14]
            RoIs.append(new_gt_boxes)
        # update RoI input for next layer
        # RoI will be modified by the padding function
        gt_boxes = []
        for temp in new_gt_boxes:
            temp_RoI = np.asarray(temp).reshape(-1, 2)
            gt_boxes.append(temp_RoI)
        new_gt_boxes = shrink(gt_boxes, 2)
        # check if new_gt_boxes is outside the edges
        for temp_box in new_gt_boxes:
            for j in range(len(temp_box)):
                if temp_box[j] < 0:
                    temp_box[j] = 0
                elif temp_box[j] >= layersize[i]:
                    temp_box[j] = layersize[i] - 1
        # need to check if these new boxes intersect with each other(union if they intersect)
        polygon = sg.Polygon([[0, 0], [0, 0], [0, 0]])
        for bbox in new_gt_boxes:
            x_cor = [bbox[2 * k] for k in range(int(len(bbox) / 2))]
            y_cor = [bbox[2 * k + 1] for k in range(int(len(bbox) / 2))]
            _list = [[x_cor[k], y_cor[k]] for k in range(len(x_cor))]
            if polygon.area == 0:
                polygon = sg.Polygon(_list)
                if not polygon.is_valid:
                    polygon = polygon.buffer(0)
            else:
                try:
                    temp_poly = sg.Polygon(_list)
                    if not temp_poly.is_valid:
                        temp_poly = temp_poly.buffer(0)
                    polygon = polygon.union(temp_poly)
                except:
                    print(_list)
                    print(polygon)
                    print(sg.Polygon(_list))
        new_gt_boxes = []
        if polygon.geom_type == 'MultiPolygon':
            for _poly in polygon:
                x, y = _poly.exterior.coords.xy
                g = [item for sublist in zip(x, y) for item in sublist]
                g = g[:len(g) - 2]
                new_gt_boxes.append(g)
        else:
            x, y = polygon.exterior.coords.xy
            g = [item for sublist in zip(x, y) for item in sublist]
            g = g[:len(g) - 2]
            new_gt_boxes.append(g)
        RoIs_padded.append(new_gt_boxes)
        RoI = new_gt_boxes
    return RoIs, RoIs_padded 

def _store(Layers, RoI, frame_id):
    RoIs, RoIs_padded = RoI_for_layers(RoI)
    path = 'store_s1_e{}_{}_{}/res_for_frame{}'.format(extend, distance, start_point, frame_id)
    try:
        os.mkdir(path)
    except OSError as error:
        print(error)
    RoIfile = open('store_s1_e{}_{}_{}/res_for_frame{}/RoI.txt'.format(extend, distance, start_point, frame_id), 'w')
    for i in range(38):
        width = layersize[i]
        height = layersize[i]
        RoI = RoIs[i]
        RoIfile.write("***************\n")
        for temp in RoI:
            RoIfile.write(json.dumps(temp))
            RoIfile.write("\n")
        RoI_padded = RoIs_padded[i]
        temp_list = []
        if len(RoI[0]) == 8:
            [x1, y1, x2, y2, x3, y3, x4, y4] = RoI[0]
            temp_list = [x1, y1, x2, y2, x3, y3, x4, y4]
            temp_list.sort()
        if temp_list == [0, 0, 0, 0, width, width, height, height]:
            #
            idx = None
        else:
            img = Image.new('L', (width, height), 0)
            for poly in RoI:
                ImageDraw.Draw(img).polygon(poly, outline=1, fill=1)
            mask = np.asarray(img)
            
            img = Image.new('L', (width, height), 0)
            for poly in RoI_padded:
                ImageDraw.Draw(img).polygon(poly, outline=1, fill=1)
            mask_padded = np.asarray(img)
            mask = mask_padded - mask
            idx = np.transpose(np.nonzero(mask))
        resfile = open("store_s1_e{}_{}_{}/res_for_frame{}/layer{}.txt".format(extend, distance, start_point, frame_id, i + 1), "w")
        if idx is not None:
            for _idx in idx:
                [x, y] = _idx
                temp = Layers[i][0, :, x, y]
                temp = np.asarray(temp).reshape(1, -1)
                text = str(x) + ',' + str(y) + '\n'
                resfile.write(text)
                np.savetxt(resfile, temp, fmt = '%.3f')
        else:
            resfile.write('')
        resfile.close()
    RoIfile.close()


def method_1(roi, bboxes):
    T1 = 0.1
    s1 = roi.area
    rate = 0
    rate_2 = 0
    bboxes = np.asarray(bboxes).reshape(-1, 4)
    for box in bboxes:
        temp = sg.box(box[0], box[1], box[2], box[3])
        s2 = temp.area
        s3 = temp.union(roi).area
        if (s1 + s2 - s3) / s1 > rate:
            rate = (s1 + s2 - s3) / s1
            # intersection with bbox / mv_roi
    if rate < T1:
        for box in bboxes:
            temp = sg.box(box[0], box[1], box[2], box[3])
            s2 = temp.area
            if s1 / s2 > rate_2:
                rate_2 = s1 / s2
    return rate, rate_2

def compare_with_edge(roi):
    def method_3(roi, edge):
        s1 = roi.area
        s2 = edge.area
        s3 = roi.union(edge).area
        return (s1+s2-s3)/s1
    w1 = 1920 * 0.1
    w2 = 1920 * 0.9
    h1 = 1080 * 0.1
    h2 = 1080 * 0.9
    r1 = sg.box(0, 0, w1, 1080)
    r2 = sg.box(0, 0, 1920, h1)
    r3 = sg.box(w2, 0, 1920, 1080)
    r4 = sg.box(0, h2, 1920, 1080)
    flag = 0
    a = method_3(roi, r1)
    b = method_3(roi, r2)
    c = method_3(roi, r3)
    d = method_3(roi, r4)
    if a > 0.8:
        flag = 1
    elif b > 0.8:
        flag = 1
    elif c > 0.8:
        flag = 1
    elif d > 0.8:
        flag = 1
    #print('in compare....',max(a, b, c, d))
    return flag


def get_RoI(prev_boxes, frame_id):
    mvs = np.loadtxt('/i3c/hpcl/zjy5087/git/YOLOv3_TensorFlow/mb/mv{}.txt'.format(frame_id))
    r = sg.box(0, 0, 0, 0)
    if len(prev_boxes) > 0:
        for box in prev_boxes:
            r1 = sg.box(box[0], box[1], box[2], box[3])
            if r.area == 0:
                r = r1 
            else:
                r = r.union(r1)
    mv_roi = sg.box(0, 0, 0, 0)
    mvs = np.asarray(mvs).reshape(-1, 4)
    if len(mvs) > 0:
        for mv in mvs:
            xmin = int(mv[0])
            ymin = int(mv[1])
            xmax = int(mv[2])
            ymax = int(mv[3])
            if mv_roi.area == 0:
                mv_roi = sg.box(xmin, ymin, xmax, ymax)
            else:
                mv_roi = mv_roi.union(sg.box(xmin, ymin, xmax, ymax))
    T1 = 0.1
    T2 = 0.7
    T3 = 0.8
    if mv_roi.geom_type == 'MultiPolygon':
        num = len(mv_roi.geoms)
        for j in range(num):
            roi = mv_roi.geoms[j]
            if roi.area > 1000:
                rate_1, rate_2 = method_1(roi, prev_boxes)
                if(rate_1 > T1 and rate_1 < T2) or rate_2 > T3:
                    r = r.union(roi)
                elif rate_1 <= T1:
                    if compare_with_edge(roi):
                        r = r.union(roi)
    elif mv_roi.area > 1000:
        rate_1, rate_2 = method_1(mv_roi, prev_boxes)
        if (rate_1 > T1 and rate_1 < T2) or (rate_2 > T3):
            r = r.union(mv_roi)
        elif rate_1 <= T1:
            if compare_with_edge(mv_roi):
                r = r.union(mv_roi) 
    res = []
    if r.geom_type == 'MultiPolygon':
        polysize = len(r.geoms)
        for i in range(polysize):
            ploy = r.geoms[i]
            x, y = ploy.exterior.coords.xy
            _x = [int(x[i] * (416 / 1920)) for i in range(len(x))]
            _y = [int(y[i] * (416 / 1920) + 91) for i in range(len(y))]
            g = [item for sublist in zip(_x, _y) for item in sublist]
            g = g[:len(g)-2]
            res.append(g)
    else:
        if r.area != 0:
            x, y = r.exterior.coords.xy
            _x = [int(x[i] * (416 / 1920)) for i in range(len(x))]
            _y = [int(y[i] * (416 / 1920) + 91) for i in range(len(y))]
            g = [item for sublist in zip(_x, _y) for item in sublist]
            g = g[:len(g)-2]
            res.append(g)
    return res

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_folder", type=str, default="data/samples", help="path to dataset")
    parser.add_argument("--model_def", type=str, default="config/yolov3.cfg", help="path to model definition file")
    parser.add_argument("--weights_path", type=str, default="weights/yolov3.weights", help="path to weights file")
    parser.add_argument("--class_path", type=str, default="data/coco.names", help="path to class label file")
    parser.add_argument("--conf_thres", type=float, default=0.2, help="object confidence threshold")
    parser.add_argument("--nms_thres", type=float, default=0.4, help="iou thresshold for non-maximum suppression")
    parser.add_argument("--batch_size", type=int, default=1, help="size of the batches")
    parser.add_argument("--n_cpu", type=int, default=0, help="number of cpu threads to use during batch generation")
    parser.add_argument("--img_size", type=int, default=416, help="size of each image dimension")
    parser.add_argument("--checkpoint_model", type=str, help="path to checkpoint model")
    parser.add_argument("--reuse_d", type=int, help="reuse distance")
    parser.add_argument("--start_p", type=int, help="start point")
    opt = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    os.makedirs("output", exist_ok=True)

    # Set up model
    model = Darknet(opt.model_def, img_size=opt.img_size).to(device)
    extend = 1

    if opt.weights_path.endswith(".weights"):
        # Load darknet weights
        model.load_darknet_weights(opt.weights_path)
    else:
        # Load checkpoint weights
        model.load_state_dict(torch.load(opt.weights_path))

    model.eval()  # Set in evaluation mode

    dataloader = DataLoader(
        ImageFolder(opt.image_folder, img_size=opt.img_size),
        batch_size=opt.batch_size,
        shuffle=False,
        num_workers=opt.n_cpu,
    )

    classes = load_classes(opt.class_path)  # Extracts class labels from file

    Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor
    distance = opt.reuse_d
    start_point = opt.start_p    #[0, distance - 1]
    try:
        os.mkdir("store_s1_e{}_{}_{}".format(extend, distance, start_point))
    except:
        print("mkdir error\n")
    print("\nStoring the his_info:")
    timefile = open('time.txt', 'w')
    for batch_i, (img_paths, input_imgs) in enumerate(dataloader):
        input_imgs = Variable(input_imgs.type(Tensor))
        if batch_i >= start_point:
            if (batch_i - start_point) % distance == 0:
                # perform inference, get layers
                prev_time = time.time()
                Layers = []
                with torch.no_grad():
                    detections, layers = model(input_imgs)
                current_time = time.time()
                inference_time = datetime.timedelta(seconds=current_time - prev_time)
                print("\t+ Batch %d, Inference Time: %s" % (batch_i, inference_time))
                timefile.write(str(inference_time) + "\n")
                for i in range(len(layers_idx)):
                    idx = layers_idx[i]
                    Layers.append(layers[idx])
            else:
                bboxes = np.loadtxt('/i3c/hpcl/zjy5087/git/PyTorch-YOLOv3/res_for_s1_bl/res{}.txt'.format(batch_i + 1))
                bboxes = np.asarray(bboxes)
                bboxes = bboxes.reshape(-1, 6)
                bboxes = bboxes[:, 2:6]
                prev_time = time.time()
                Region_of_interests = get_RoI(bboxes, batch_i + 4200)
                #Region_of_interests = RoI_extension(Region_of_interests, 0.25)
                _store(Layers, Region_of_interests, batch_i)
                current_time = time.time()
                inference_time = datetime.timedelta(seconds=current_time - prev_time)
                print("\t+ Batch %d, Inference Time: %s" % (batch_i, inference_time))
                timefile.write(str(inference_time) + "\n")
    timefile.close()





