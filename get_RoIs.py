from __future__ import division

from my_conv_models import *
from my_conv_models_2_copy import *
from utils.utils import *
from utils.datasets import *
from intergated_version import RoI_for_layers as RL 
import os
import sys
import time
import datetime
import argparse
import pyclipper
import json
import shapely.geometry as sg
from PIL import Image, ImageDraw
import cv2

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
    def padding_RoI(new_gt_boxes):
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
                temp_poly = sg.Polygon(_list)
                if not temp_poly.is_valid:
                    temp_poly = temp_poly.buffer(0)
                polygon = polygon.union(temp_poly)
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
        return new_gt_boxes
    RoIs = []
    RoIs_padded = []
    temp_RoI = []   # to reshape
    flag = 0
    for i in range(38):
        if flag == 0:
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
            new_gt_boxes = padding_RoI(new_gt_boxes)
            if len(new_gt_boxes) == 1:
                g = new_gt_boxes[0]
                if len(g) == 8:
                    [x1, y1, x2, y2, x3, y3, x4, y4] = g
                    temp_list = [x1, y1, x2, y2, x3, y3, x4, y4]
                    temp_list.sort()
                    w = layersize[i] - 1
                    if temp_list == [0, 0, 0, 0, w, w, w, w]:
                        flag = 1
            RoIs_padded.append(new_gt_boxes)
            RoI = new_gt_boxes
        else:
            if i == 32:
                #
                new_gt_boxes = RoIs[23]
                RoIs.append(new_gt_boxes)
                if new_gt_boxes != None:
                    flag = 0
                    new_gt_boxes = padding_RoI(new_gt_boxes)
                    if len(new_gt_boxes) == 0:
                        g = new_gt_boxes[0]
                        if len(g) == 8:
                            [x1, y1, x2, y2, x3, y3, x4, y4] = g
                            temp_list = [x1, y1, x2, y2, x3, y3, x4, y4]
                            temp_list.sort()
                            w = layersize[i] - 1
                            if temp_list == [0, 0, 0, 0, w, w, w, w]:
                                flag = 1
                    RoIs_padded.append(new_gt_boxes)
                    RoI = new_gt_boxes
                else:
                    RoIs_padded.append(None)
            elif i == 35:
                new_gt_boxes = RoIs[14]
                RoIs.append(new_gt_boxes)
                if new_gt_boxes != None:
                    flag = 0
                    new_gt_boxes = padding_RoI(new_gt_boxes)
                    if len(new_gt_boxes) == 1:
                        g = new_gt_boxes[0]
                        if len(g) == 8:
                            [x1, y1, x2, y2, x3, y3, x4, y4] = g
                            temp_list = [x1, y1, x2, y2, x3, y3, x4, y4]
                            temp_list.sort()
                            w = layersize[i] - 1
                            if temp_list == [0, 0, 0, 0, w, w, w, w]:
                                flag = 1
                    RoIs_padded.append(new_gt_boxes)
                    RoI = new_gt_boxes
                else:
                    RoIs_padded.append(None)

            else:
                RoIs.append(None)
                RoIs_padded.append(None)
    return RoIs, RoIs_padded 



def _draw_RoI_update(RoI, RoI2, id):
    RoIs, RoIs_padded = RoI_for_layers(RoI)
    RoIs_2, RoIs_padded_2 = RL(RoI2)
    os.makedirs('frame{}'.format(id), exist_ok=True)
    new_img = np.zeros([416, 416, 3], np.uint8)
    for temp in RoI:
        temp = np.asarray(temp).reshape(-1, 2)
        cv2.polylines(new_img, np.int32([temp]), True, (80, 90, 100), 1)
        idx = np.where(temp[:,0] == temp[:,0].min())
        text = str(temp[idx][0][0]) + ',' + str(temp[idx][0][1])
        cv2.putText(new_img, text, (temp[idx][0][0], temp[idx][0][1]), cv2.FONT_HERSHEY_SIMPLEX, 0.2, (255, 255, 255), 0)
    im = Image.fromarray(new_img)
    im.save("frame{}/initial_RoI.jpeg".format(id))
    for i in range(38):
        width = layersize[i]
        height = layersize[i]
        RoI = RoIs[i]
        RoI_padded = RoIs_padded[i]
        RoI_2 = RoIs_2[i]
        RoI_padded_2 = RoIs_padded_2[i]
        idx = []
        new_img = np.zeros([width, height, 3], np.uint8)
        if RoI_padded is not None:
            for temp in RoI_padded:
                temp = np.asarray(temp).reshape(-1, 2)
                _b = sg.Polygon(temp).bounds
                cv2.polylines(new_img, np.int32([temp]), True, (0, 255, 0), 1)
                idx = np.where(temp[:,0] == temp[:,0].min())
                text = str(int(temp[idx][0][0])) + ',' + str(int(temp[idx][0][1]))
                cv2.putText(new_img, text, (int(temp[idx][0][0]), int(temp[idx][0][1])), cv2.FONT_HERSHEY_SIMPLEX, 0.2, (255, 255, 255), 0)
                #cv2.rectangle(new_img, (int(_b[0]), int(_b[1])), (int(_b[2]), int(_b[3])), (0, 0, 255), 1) 

        if RoI is not None:
            for temp in RoI:
                temp = np.asarray(temp).reshape(-1, 2)
                cv2.polylines(new_img, np.int32([temp]), True, (255, 0, 0), 1)
        
        if RoI_padded_2 is not None:
            for temp in RoI_padded_2:
                cv2.rectangle(new_img, (int(temp[0]), int(temp[1])), (int(temp[2]), int(temp[3])), (255, 255, 255), 1)
        if RoI_2 is not None:
            for temp in RoI_2:
                cv2.rectangle(new_img, (int(temp[0]), int(temp[1])), (int(temp[2]), int(temp[3])), (0, 255, 255), 1)
        
        im = Image.fromarray(new_img)
        im.save("frame{}/layer{}.jpeg".format(id, i))


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
        rate = 0
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
    mvs = np.loadtxt('/i3c/hpcl/zjy5087/YOLO/mb/mb_v3/mv{}.txt'.format(frame_id))
    if len(prev_boxes) > 0:
        r = sg.box(prev_boxes[0][0], prev_boxes[0][1], prev_boxes[0][2], prev_boxes[0][3])
        for box in prev_boxes[1:]:
            r1 = sg.box(box[0], box[1], box[2], box[3])
            r = r.union(r1)
    mvs = np.asarray(mvs).reshape(-1, 4).astype('int32')
    if len(mvs) > 0:
        mv_roi = sg.box(mvs[0][0], mvs[0][1], mvs[0][2], mvs[0][3])
        for mv in mvs[1:]:
            [xmin, ymin, xmax, ymax] = mv
            mv_roi = mv_roi.union(sg.box(xmin, ymin, xmax, ymax))
    
        T1 = 0.4
        T2 = 0.7
        T3 = 0.8
        flag = 0
        if mv_roi.geom_type == 'MultiPolygon':
            num = len(mv_roi.geoms)
            for j in range(num):
                roi = mv_roi.geoms[j]
                if roi.area > 1000:
                    rate_1, rate_2 = method_1(roi, prev_boxes)
                    if rate_1 > 0 or rate_2 > T3:
                        r = r.union(roi)
                        #flag = 2
                    elif rate_1 == 0:
                        if compare_with_edge(roi):
                            r = r.union(roi)
        elif mv_roi.area > 1000:
            rate_1, rate_2 = method_1(mv_roi, prev_boxes)
            if rate_1 > 0 or (rate_2 > T3):
                r = r.union(mv_roi)
            elif rate_1 == 0:
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

def format_result(detections):
    dec = []
    if detections[0] is not None:
        for temp in detections:
            temp = temp.tolist()
            dec.append(temp)
        dec = torch.tensor(dec)
        dec = torch.reshape(dec, (-1, 7))
        dec = rescale_boxes(dec, opt.img_size, img_shape)
        dec = dec.numpy()
    res = []
    if len(dec) > 0:
        res = np.zeros(len(dec) * 6).reshape(-1, 6)
        res[:, 0] = dec[:, 6]
        res[:, 1] = dec[:, 5]
        res[:, 2:6] = dec[:, 0:4]
    return res
def get_image_id(path):
    items = path.split('/')
    image_name = items[-1]
    image_name = image_name[5 : -5]
    return int(image_name)

def RoI_box(Region_of_interests):
    return_box = []
    for RoI in Region_of_interests:
        RoI = np.asarray(RoI).reshape(-1, 2)
        _list = sg.Polygon(RoI).bounds
        n_list = [int(x) for x in _list]
        return_box.append(n_list)
    return return_box

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_folder", type=str, default="data/samples", help="path to dataset")
    parser.add_argument("--sample_index", type=int, default=1, help="the index of sample folder")
    parser.add_argument("--extension", type=int, default=1, help="extension for RoIs")
    parser.add_argument("--model_def", type=str, default="config/yolov3.cfg", help="path to model definition file")
    parser.add_argument("--weights_path", type=str, default="weights/yolov3.weights", help="path to weights file")
    parser.add_argument("--class_path", type=str, default="data/coco.names", help="path to class label file")
    parser.add_argument("--conf_thres", type=float, default=0.4, help="object confidence threshold")
    parser.add_argument("--nms_thres", type=float, default=0.4, help="iou thresshold for non-maximum suppression")
    parser.add_argument("--batch_size", type=int, default=1, help="size of the batches")
    parser.add_argument("--n_cpu", type=int, default=0, help="number of cpu threads to use during batch generation")
    parser.add_argument("--img_size", type=int, default=416, help="size of each image dimension")
    parser.add_argument("--checkpoint_model", type=str, help="path to checkpoint model")
    parser.add_argument("--reuse_d", type=int, default=2, help="reuse distance")
    parser.add_argument("--start_p", type=int, default=0, help="start point")
    opt = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Set up model
    model = Darknet(opt.model_def, img_size=opt.img_size).to(device)
    extend = opt.extension
    # 
    idx_for_mv = 4200

    if opt.weights_path.endswith(".weights"):
        # Load darknet weights
        weights_dict, bias_dict = model.load_darknet_weights(opt.weights_path)
    else:
        # Load checkpoint weights
        model.load_state_dict(torch.load(opt.weights_path))

    model.eval()  # Set in evaluation mode


    image_sets = ImageFolder(opt.image_folder, img_size=opt.img_size)

    classes = load_classes(opt.class_path)  # Extracts class labels from file

    Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor
    
    distance = opt.reuse_d
    start_point = opt.start_p    #[0, distance - 1]
    idx_sam = opt.sample_index
    img_shape = (1080, 1920)

    for batch_i in range(len(image_sets)):
        (img_paths, input_imgs) = image_sets[batch_i]
        input_imgs = input_imgs.unsqueeze(0)
        input_imgs = Variable(input_imgs.type(Tensor))

        if batch_i >= start_point:
        # s: idx for samples file, e: expension (1: no expension), dist, start_point 
            if (batch_i - start_point) % distance == 0:
            # perform full inference, get layers
                with torch.no_grad():
                    prev_time = time.time()
                    detections, layers = model(input_imgs, weights_dict, bias_dict)
                    current_time = time.time()
                    detections = non_max_suppression(detections, opt.conf_thres, opt.nms_thres)

                inference_time = datetime.timedelta(seconds=current_time - prev_time)
                print("\t+ Batch %d, Inference Time: %s" % (batch_i, inference_time))
                # process detection results:
                bboxes = []
                res = format_result(detections)
                print('full inf, res:\n', res)
                if len(res) > 0:
                    bboxes = np.zeros(4 * len(res)).reshape(-1, 4)
                    bboxes[:, :] = res[:, 2 : 6]

            else:
                # do inf for batch_i, hisinfo is in total_dicts[idx_for_partial]
                Region_of_Interest = get_RoI(bboxes, batch_i + 1)
                Region_of_Interest2 = RoI_box(Region_of_Interest)
                _draw_RoI_update(Region_of_Interest, Region_of_Interest2, batch_i + 1 + idx_for_mv)

