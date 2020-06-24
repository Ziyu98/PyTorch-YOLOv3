from __future__ import division

from my_conv_models import *
from my_conv_models_2_copy import *
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

def _store(Layers, RoI):
    RoIs, RoIs_padded = RoI_for_layers(RoI)
    hisinfo_dicts = []
    hisinfo_dict = {}
    for i in range(38):
        width = layersize[i]
        height = layersize[i]
        RoI = RoIs[i]
        RoI_padded = RoIs_padded[i]
        temp_list = []
        idx = []
        if RoI is not None:
            if len(RoI[0]) == 8:
                [x1, y1, x2, y2, x3, y3, x4, y4] = RoI[0]
                temp_list = [x1, y1, x2, y2, x3, y3, x4, y4]
                temp_list.sort()
            if temp_list == [0, 0, 0, 0, width, width, height, height]:
            #
                idx = None
        else:
            idx = None
        if idx is not None:
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
        
            for _idx in idx:
                [x, y] = _idx
                key = tuple([x, y])
                temp = Layers[i][0, :, x, y]
                temp = np.asarray(temp).reshape(1, -1)
                hisinfo_dict[key] = torch.tensor(temp)
        else:
            key = tuple([0, 0])
            hisinfo_dict[key] = None
        hisinfo_dicts.append(hisinfo_dict)
        hisinfo_dict = {}
    return hisinfo_dicts, RoIs


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
                if(rate_1 < T1 and rate_1 > 0) or rate_2 > T3:
                    r = r.union(roi)
                    flag = 2
                elif rate_1 == 0:
                    if compare_with_edge(roi):
                        r = r.union(roi)
                        flag = 2
                elif rate_1 > T1 and rate_1 < T2:
                    r = r.union(roi)             # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
                    flag = 1
    elif mv_roi.area > 1000:
        rate_1, rate_2 = method_1(mv_roi, prev_boxes)
        if (rate_1 < T1 and rate_1 > 0) or (rate_2 > T3):
            r = r.union(mv_roi)
            flag = 2
        elif rate_1 == 0:
            if compare_with_edge(mv_roi):
                r = r.union(mv_roi) 
                flag = 2
        elif rate_1 > T1 and rate_1 < T2:
            r = r.union(mv_roi)
            flag = 1
    if flag != 0:
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
        return flag, res
    else:
        return 0, None

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

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_folder", type=str, default="data/samples_4", help="path to dataset")
    parser.add_argument("--sample_index", type=int, default=4, help="the index of sample folder")
    parser.add_argument("--extension", type=int, default=1, help="extension for RoIs")
    parser.add_argument("--model_def", type=str, default="config/yolov3.cfg", help="path to model definition file")
    parser.add_argument("--model_2_def", type=str, default="config/yolov3_2.cfg", help="path to model_2 definition file")
    parser.add_argument("--weights_path", type=str, default="weights/yolov3.weights", help="path to weights file")
    parser.add_argument("--class_path", type=str, default="data/coco.names", help="path to class label file")
    parser.add_argument("--conf_thres", type=float, default=0.4, help="object confidence threshold")
    parser.add_argument("--nms_thres", type=float, default=0.4, help="iou thresshold for non-maximum suppression")
    parser.add_argument("--batch_size", type=int, default=1, help="size of the batches")
    parser.add_argument("--n_cpu", type=int, default=0, help="number of cpu threads to use during batch generation")
    parser.add_argument("--img_size", type=int, default=416, help="size of each image dimension")
    parser.add_argument("--checkpoint_model", type=str, help="path to checkpoint model")
    #parser.add_argument("--reuse_d", type=int, default=2, help="reuse distance")
    #parser.add_argument("--start_p", type=int, default=0, help="start point")
    opt = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_num_threads(1)

    # Set up model
    model = Darknet(opt.model_def, img_size=opt.img_size).to(device)
    model_2 = Darknet_2(opt.model_2_def, img_size=opt.img_size).to(device)
    extend = opt.extension
    # 
    idx_for_mv = 9210

    if opt.weights_path.endswith(".weights"):
        # Load darknet weights
        weights_dict, bias_dict = model.load_darknet_weights(opt.weights_path)
    else:
        # Load checkpoint weights
        model.load_state_dict(torch.load(opt.weights_path))

    model.eval()  # Set in evaluation mode
    if opt.weights_path.endswith(".weights"):
        # Load darknet weights
        weights_dict_2, bias_dict_2 = model_2.load_darknet_weights(opt.weights_path)
    else:
        # Load checkpoint weights
        model_2.load_state_dict(torch.load(opt.weights_path))

    model_2.eval()  # Set in evaluation mode

    dataloader = DataLoader(
        ImageFolder(opt.image_folder, img_size=opt.img_size),
        batch_size=opt.batch_size,
        shuffle=False,
        num_workers=opt.n_cpu,
    )

    classes = load_classes(opt.class_path)  # Extracts class labels from file

    Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor
    
    #distance = opt.reuse_d
    #start_point = opt.start_p    #[0, distance - 1]
    idx_sam = opt.sample_index
    img_shape = (1080, 1920)
    os.makedirs('dynres_for_s{}_e{}'.format(idx_sam, extend), exist_ok=True)
    timefile = open('dynres_for_s{}_e{}/_time'.format(idx_sam, extend), 'w')
    if os.path.exists('dynres_for_s{}_e{}/RoI'.format(idx_sam, extend)):
        os.remove('dynres_for_s{}_e{}/RoI'.format(idx_sam, extend))
    RoIfile = open('dynres_for_s{}_e{}/RoI'.format(idx_sam, extend), 'a')
    full_flag = True
    for batch_i, (img_paths, input_imgs) in enumerate(dataloader):
        input_imgs = Variable(input_imgs.type(Tensor))
        #if batch_i >= start_point:
        resfile = open('dynres_for_s{}_e{}/res{}.txt'.format(idx_sam, extend, batch_i + 1), 'w')
        # s: idx for samples file, e: expension (1: no expension), dist, start_point 
        if full_flag:
            # perform full inference, get layers
            Layers = []
            total_dicts = []
            total_RoIs = []
            with torch.no_grad():
                try:
                    prev_time = time.time()
                    detections, layers = model(input_imgs, weights_dict, bias_dict)
                    current_time = time.time()
                    detections = non_max_suppression(detections, opt.conf_thres, opt.nms_thres)
                except:
                    continue
            inference_time = datetime.timedelta(seconds=current_time - prev_time)
            text = "full inference, time = " + str(inference_time) + "\n"
            timefile.write(text)
            print("\t+ Batch %d, Inference Time: %s" % (batch_i, inference_time))
            # process detection results:
            bboxes = []
            res = format_result(detections)
            print('full inf, res:\n', res)
            np.savetxt(resfile, res, fmt='%.3f')
            resfile.close()
            if len(res) > 0:
                bboxes = np.zeros(4 * len(res)).reshape(-1, 4)
                bboxes[:, :] = res[:, 2 : 6]
            prev_time = time.time()
            for i in range(len(layers_idx)):
                idx = layers_idx[i]
                Layers.append(layers[idx])
            #for j in range(1, distance):
            j = 0
            while(True):
                j += 1
                try:
                    flag, Region_of_interests = get_RoI(bboxes, batch_i + j + idx_for_mv)
                except:
                    continue
                # if expand?
                if flag == 1:    #partial
                    try:
                        if extend != 1:
                            Region_of_interests = RoI_extension(Region_of_interests, (1 - 1. / extend))
                        temp_his, temp_RoI = _store(Layers, Region_of_interests)
                        total_dicts.append(temp_his)
                        total_RoIs.append(temp_RoI)               # dictionary for the following n frames
                        full_flag = False
                    except:
                        next_frames_cnt = j - 1
                        break
                elif flag == 0:
                    # do nothing
                    total_dicts.append(None)
                    total_RoIs.append(None)
                    full_flag = False
                else:
                    # need full inf for this frame
                    next_frames_cnt = j - 1
                    break

            current_time = time.time()
            idx_for_partial = 0
            exe_time = datetime.timedelta(seconds=current_time - prev_time)
            text = "store hist info. for*" + str(next_frames_cnt) + "*frames, time = " + str(exe_time) + "\n"
            timefile.write(text)

        else:
            # do inf for batch_i, hisinfo is in total_dicts[idx_for_partial]
            if idx_for_partial < next_frames_cnt:
                # idx_for_partial(and skips)

                RoIs = total_RoIs[idx_for_partial]
                RoIfile.write("RoI for frame " + str(batch_i + 1) + "\n")
                if RoIs is not None:
                    for layer_i in range(38):
                        RoI = RoIs[layer_i]
                        RoIfile.write("***************\n")
                        if RoI is not None:
                            for temp in RoI:
                                RoIfile.write(json.dumps(temp))
                                RoIfile.write("\n")
                        else:
                            RoIfile.write("None\n")
                    prev_time = time.time()
                    for j in range(len(layer_index)):
                        temp = total_dicts[idx_for_partial][j]
                        locals()['_layer' + str(layer_index[j])] = temp
                        locals()['_RoI_' + str(layer_index[j])] = RoIs[j]
                    current_time = time.time()
                    exe_time = datetime.timedelta(seconds=current_time - prev_time)
                    text = "load hist info. for frame" + str(batch_i + 1) + ", time = " + str(exe_time) + "\n"
                    timefile.write(text)

                    with torch.no_grad():
                        try:
                            prev_time = time.time()
                            detections = model_2(input_imgs, weights_dict_2, bias_dict_2, _layer1, _layer2, _layer3, _layer4, _layer5, _layer6, _layer7, _layer8, _layer9, _layer10, _layer11, _layer12, _layer13, _layer14, _layer15, _layer16, _layer17, _layer18, _layer19, _layer20, _layer21, _layer22, _layer23, _layer24, _layer25, _layer26, _layer27, _layer28, _layer29, _layer30, _layer32, _layer34, _layer37, _layer39, _layer41, _layer44, _layer46, _layer48, _RoI_1, _RoI_2, _RoI_3, _RoI_4, _RoI_5, _RoI_6, _RoI_7, _RoI_8, _RoI_9, _RoI_10, _RoI_11, _RoI_12, _RoI_13, _RoI_14, _RoI_15, _RoI_16, _RoI_17, _RoI_18, _RoI_19, _RoI_20, _RoI_21, _RoI_22, _RoI_23, _RoI_24, _RoI_25, _RoI_26, _RoI_27, _RoI_28, _RoI_29, _RoI_30, _RoI_32, _RoI_34, _RoI_37, _RoI_39, _RoI_41, _RoI_44, _RoI_46, _RoI_48)
                            current_time = time.time()
                            detections = non_max_suppression(detections, opt.conf_thres, opt.nms_thres) 
                            inference_time = datetime.timedelta(seconds=current_time - prev_time)
                        except:
                            continue
                        print("\t+ Batch %d, Inference Time: %s" % (batch_i, inference_time))
                        text = "partial inference, time = " + str(inference_time) + "\n"
                        timefile.write(text)
                        partial_res = format_result(detections)
                        print('partial inf, res:\n', partial_res)
                        np.savetxt(resfile, partial_res, fmt='%.3f')
                        resfile.close()
                        idx_for_partial += 1 # next frame
                        if idx_for_partial == next_frames_cnt:
                            full_flag = True
                else:
                    print('reuse, res:\n', res)
                    np.savetxt(resfile, res, fmt='%.3f')
                    resfile.close()
                    idx_for_partial += 1
                    if idx_for_partial == next_frames_cnt:
                        full_flag = True
    RoIfile.close()
    timefile.close()







