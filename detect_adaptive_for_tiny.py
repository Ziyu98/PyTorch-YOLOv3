from __future__ import division

from models.models_torch_conv_0 import *
from models.models_tiny_torch_conv_impl_1 import *
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
import cv2
from PIL import Image, ImageDraw
from math import ceil, floor, sqrt

import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable


layersize = [416, 208, 208, 104, 104, 52, 52, 26, 26, 13, 13, 13, 13, 13, 13, 13]

layer_type = [1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 3, 1, 4, 1, 4]

def RoI_extension(RoIs, rate):
    cnt = len(RoIs)
    for box in RoIs:
        w = box[2] - box[0]
        h = box[3] - box[1]
        dh = (sqrt(rate) - 1) * h / 2
        dw = (sqrt(rate) - 1) * w / 2
        box[:] = [max(int(box[0] - dw), 0), max(int(box[1] - dh), 0), min(int(box[2] + dw), 415), min(int(box[3] + dh), 415)]
    if cnt > 1:
        polygon = sg.box(RoIs[0][0], RoIs[0][1], RoIs[0][2], RoIs[0][3])
        for box in RoIs[1:]:
            polygon = polygon.union(sg.box(box[0], box[1], box[2], box[3]))
        new_gt_boxes = []
        if polygon.geom_type == 'MultiPolygon':
            if len(polygon) != cnt:
                for _poly in polygon:
                    g = _poly.bounds
                    _g = [ing(x) for x in g]
                    new_gt_boxes.append(_g)
            else:
                return RoIs
        else:
            g = polygon.bounds
            _g = [int(x) for x in g]
            new_gt_boxes.append(_g)
        return new_gt_boxes

    return RoIs

def RoI_for_layers(RoI):
    # input RoI is the region of interest for the raw frame
    # this function is to calculate the RoI for padding function in each layer
    def padding_RoI(ori_RoIs, temp_size):
        if ori_RoIs is None:
            return None, None
        new_RoIs = []
        for temp in ori_RoIs:
            temp = [max(temp[0] - 2, 0), 
                    max(temp[1] - 2, 0),
                    min(temp[2] + 2, temp_size),
                    min(temp[3] + 2, temp_size)]
            new_RoIs.append(temp)
        # check if new_gt_boxes is outside the edges
        # need to check if these new boxes intersect with each other(union if they intersect)
        if len(new_RoIs) > 1:
            polygon = sg.box(new_RoIs[0][0], new_RoIs[0][1], new_RoIs[0][2], new_RoIs[0][3])
            for i in range(1, len(new_RoIs)):
                polygon = polygon.union(sg.box(new_RoIs[i][0], new_RoIs[i][1], new_RoIs[i][2], new_RoIs[i][3]))
            new_gt_boxes = []
            if polygon.geom_type == 'MultiPolygon':
                if len(polygon) != len(new_RoIs):
                    for _poly in polygon:
                        g = _poly.bounds
                        _g = [int(x) for x in g]
                        new_gt_boxes.append(_g)
                else:
                    return new_RoIs
            else:
                g = polygon.bounds
                _g = [int(x) for x in g]
                new_gt_boxes.append(_g)
            return new_gt_boxes
        else:
            return new_RoIs


    RoIs = list()
    RoIs_padded = list()
    flag = 0
    for i in range(16):
        temp_size = layersize[i] - 1
        new_boxes = []
        if flag == 0:
            if layer_type[i] == 1:
                for temp in RoI:
                    new_box = [int(temp[0] + 1), int(temp[1] + 1), int(temp[2] - 1), int(temp[3] - 1)]
                    new_boxes.append(new_box)
            elif layer_type[i] == 2:
                for temp in RoI:
                    new_box = [ceil(temp[0] / 2), ceil(temp[1] / 2), floor(temp[2] / 2), floor(temp[3] / 2)]
                    new_boxes.append(new_box)
            elif layer_type[i] == 3:
                for temp in RoI:
                    new_box = [int(temp[0] + 1), int(temp[1] + 1), int(temp[2]), int(temp[3])]
                    new_boxes.append(new_box)
            elif layer_type[i] == 4:
                new_boxes = RoI

            # update RoI input for next layer
            # RoI will be modified by the padding function
            RoIs.append(new_boxes)
            padded_boxes = padding_RoI(new_boxes, temp_size)
            RoIs_padded.append(padded_boxes)
            if padded_boxes is not None and padded_boxes[0] == [0, 0, temp_size, temp_size]:
                flag = 1
                padded_boxes = None
            #print('in line 144:, raw and united:', raw_padded_boxes, padded_boxes)
            RoI = padded_boxes
        else:
            RoIs.append(None)
            RoIs_padded.append(None)
    return RoIs, RoIs_padded

def _store(Layers, RoI):
    RoIs, RoIs_padded = RoI_for_layers(RoI)
    hisinfo_all_layers = []
    hisinfo_one_layer = []
    for i in range(16):
        RoI = RoIs[i]
        RoI_padded = RoIs_padded[i]
        
        if RoI is not None and RoI_padded is not None:
            for R, R_p in zip(RoI, RoI_padded):
                his_for_one_box = []
                [xmin, ymin, xmax, ymax] = R
                [xmin_2, ymin_2, xmax_2, ymax_2] = R_p
                his1 = Layers[i][0, :, ymin_2:ymin, xmin:xmax+1]
                his_for_one_box.append(his1)
                his2 = Layers[i][0, :, ymax+1:ymax_2+1, xmin:xmax+1]
                his_for_one_box.append(his2)
                his3 = Layers[i][0, :, ymin_2:ymax_2+1, xmin_2:xmin]
                his_for_one_box.append(his3)
                his4 = Layers[i][0, :, ymin_2:ymax_2+1, xmax+1:xmax_2+1]
                his_for_one_box.append(his4)
                hisinfo_one_layer.append(his_for_one_box)
        else:
            hisinfo_one_layer = None
        hisinfo_all_layers.append(hisinfo_one_layer)
        hisinfo_one_layer = []
    return hisinfo_all_layers, RoIs, RoIs_padded


def method_1(roi, bboxes):
    T1 = 0.005
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
    
    w1 = img_shape[1] * 0.1
    w2 = img_shape[1] * 0.9
    h1 = img_shape[0] * 0.1
    h2 = img_shape[0] * 0.9
    x1, y1, x2, y2 = roi.bounds
    if x2 < w1 or y2 < h1 or x1 > w2 or y1 >h2:
        flag = 1
    else:
        flag = 0
    #print('in compare....',max(a, b, c, d))
    return flag


def get_RoI(prev_boxes, frame_id):
    if len(prev_boxes) == 0:
        return 2, None
    temp_idx = opt.image_folder.find('V')
    mvpath = 'MV_' + opt.image_folder[temp_idx:]
    mvpath = os.path.join('/i3c/hpcl/zjy5087/YOLO/mb', mvpath, 'mv{}.txt'.format(frame_id))
    mvs = np.loadtxt(mvpath)
    area_thre = 1920 * 1080
    #mvs = np.loadtxt('/i3c/hpcl/zjy5087/YOLO/mb/mb_v3/mv{}.txt'.format(frame_id))
    if len(mvs) == 0:
        return 2, None
    r = sg.box(0, 0, 0, 0)
    if len(prev_boxes) > 0:
        for box in prev_boxes:
            r1 = sg.box(box[0], box[1], box[2], box[3])
            if r1.area < area_thre:
                area_thre = r1.area
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
    T1 = 0.3
    T2 = 0.7
    T3 = 0.8
    flag = 0
    area_thre = area_thre * 0.25 if area_thre * 0.25 > 1024 else 1024
    area_thre_min = 1024
    if mv_roi.geom_type == 'MultiPolygon':
        num = len(mv_roi.geoms)
        for j in range(num):
            roi = mv_roi.geoms[j]
            if roi.area > area_thre_min:
                if roi.area > area_thre:
                    rate_1, rate_2 = method_1(roi, prev_boxes)
                    if(rate_1 < T1 and rate_1 > 0) or rate_2 > T3:
                        return 2, None
                    elif rate_1 > T1 and rate_1 < T2:
                        r = r.union(roi)
                        flag = 1
                else:
                    if compare_with_edge(roi):
                        return 2, None

    elif mv_roi.area > area_thre_min:
        if mv_roi.area > area_thre:
            rate_1, rate_2 = method_1(mv_roi, prev_boxes)
            if(rate_1 < T1 and rate_1 > 0) or rate_2 > T3:
                return 2, None
            elif rate_1 > T1 and rate_1 < T2:
                r = r.union(mv_roi)
                flag = 1
        else:
            if compare_with_edge(mv_roi):
                return 2, None
    if flag != 0:
        res = []
        max_boundary = max(img_shape[0], img_shape[1])
        if r.geom_type == 'MultiPolygon':
            polysize = len(r.geoms)
            for i in range(polysize):
                ploy = r.geoms[i]
                x, y = ploy.exterior.coords.xy
                _x = [min(int(x[i] * (416 / max_boundary)), 415) for i in range(len(x))]
                _y = [int(y[i] * (416 / max_boundary) + 91) for i in range(len(y))]
                g = [item for sublist in zip(_x, _y) for item in sublist]
                g = g[:len(g)-2]
                res.append(g)
        else:
            if r.area != 0:
                x, y = r.exterior.coords.xy
                _x = [min(int(x[i] * (416 / max_boundary)), 415) for i in range(len(x))]
                _y = [int(y[i] * (416 / max_boundary) + 91) for i in range(len(y))]
                g = [item for sublist in zip(_x, _y) for item in sublist]
                g = g[:len(g)-2]
                res.append(g)
        return flag, res
    else:
        return 0, None

def RoI_box(Region_of_interests):
    return_box = []
    for RoI in Region_of_interests:
        RoI = np.asarray(RoI).reshape(-1, 2)
        _list = sg.Polygon(RoI).bounds
        n_list = [int(x) for x in _list]
        return_box.append(n_list)
    return return_box


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
        res = dec[:, [6, 5, 0, 1, 2, 3]]
        res[res < 0] = 0
        for box in res:
            if box[0] == 0:
                box[0] = 1
            elif box[0] == 7:
                box[0] = 2
    return res
def get_image_id(path):
    items = path.split('/')
    image_name = items[-1]
    image_name = image_name[5 : -5]
    return int(image_name)

def store_layers(layers):
    Layers = []
    for i in range(13):
        Layers.append(layers[i])
    return Layers

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_folder", type=str, default="data/samples", help="path to dataset")
    parser.add_argument("--sample_index", type=int, default=1, help="the index of sample folder")
    parser.add_argument("--extension", type=int, default=1, help="extension for RoIs")
    parser.add_argument("--model_def", type=str, default="config/yolov3-tiny.cfg", help="path to model definition file")
    parser.add_argument("--model_2_def", type=str, default="config/yolov3-tiny-new-model.cfg", help="path to model_2 definition file")
    parser.add_argument("--weights_path", type=str, default="weights/yolov3-tiny.weights", help="path to weights file")
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
    idx_for_mv = 1

    if opt.weights_path.endswith(".weights"):
        # Load darknet weights
        print('model 1 loading......')
        model.load_darknet_weights(opt.weights_path)
    else:
        # Load checkpoint weights
        model.load_state_dict(torch.load(opt.weights_path))

    model.eval()  # Set in evaluation mode
    if opt.weights_path.endswith(".weights"):
        # Load darknet weights
        print('model 2 loading......')
        model_2.load_darknet_weights(opt.weights_path)
    else:
        # Load checkpoint weights
        model_2.load_state_dict(torch.load(opt.weights_path))

    model_2.eval()  # Set in evaluation mode

    image_sets = ImageFolder(opt.image_folder, img_size=opt.img_size)

    classes = load_classes(opt.class_path)  # Extracts class labels from file

    Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor

    #distance = opt.reuse_d
    #start_point = opt.start_p    #[0, distance - 1]
    idx_sam = opt.sample_index
    first_file = os.listdir(opt.image_folder)[0]
    first_file = os.path.join(opt.image_folder, first_file)
    img_shape = cv2.imread(first_file).shape[:2]
    temp_idx = opt.image_folder.find('V')
    path = 'dynres_for_' +  opt.image_folder[temp_idx:] + '_e{}'.format(extend)
    os.makedirs(path, exist_ok=True)
    timepath = os.path.join(path, '_time')
    timefile = open(timepath, 'w')
    mappath = os.path.join(path, 'map')
    mapfile = open(mappath, 'w')
    RoIpath = os.path.join(path, 'RoI')
    if os.path.exists(RoIpath):
        os.remove(RoIpath)
    RoIfile = open(RoIpath, 'a')
    full_flag = True
    for batch_i in range(len(image_sets)):
        if batch_i >= 5244:
            (img_paths, input_imgs) = image_sets[batch_i]
            input_imgs = input_imgs.unsqueeze(0)
            input_imgs = Variable(input_imgs.type(Tensor))
            respath = os.path.join(path, 'res{}.txt'.format(batch_i + 1))
            resfile = open(respath, 'w')
            # s: idx for samples file, e: expension (1: no expension), dist, start_point 
            if full_flag:
                # perform full inference, get layers
                Layers = []
                total_dicts = []
                total_RoIs = []
                total_RoIs_p = []
                raw_RoIs = []
                with torch.no_grad():
                    prev_time = time.time()
                    detections, layers = model(input_imgs)
                    current_time = time.time()
                    detections = non_max_suppression(detections, opt.conf_thres, opt.nms_thres)
                    text = 'frame = ' + str(batch_i + 1) + ', inf. = full' + '\n'
                    mapfile.write(text)
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

                Layers = store_layers(layers)
                
                #for j in range(1, distance):
                j = 0
                while(True):
                    j += 1
                    prev_time = time.time()
                    if batch_i + j + idx_for_mv >= len(image_sets):
                        next_frames_cnt = j - 1
                        break
                    flag, Region_of_interests = get_RoI(bboxes, batch_i + j + idx_for_mv)
                    current_time = time.time()
                    exe_time = datetime.timedelta(seconds=current_time - prev_time)
                    text = "get roi time = " + str(exe_time) + "\n"
                    timefile.write(text)
                    print(text)

                    # if expand?
                    if flag == 1:    #partial
                        prev_time = time.time()
                        Region_of_interests = RoI_box(Region_of_interests)
                        if extend != 1:
                            Region_of_interests = RoI_extension(Region_of_interests, (extend))
                        temp_his, temp_RoI, temp_RoI_p = _store(Layers, Region_of_interests)
                        total_dicts.append(temp_his)
                        total_RoIs.append(temp_RoI)               # dictionary for the following n frames
                        total_RoIs_p.append(temp_RoI_p)
                        raw_RoIs.append(Region_of_interests)
                        full_flag = False
                        current_time = time.time()
                        exe_time = datetime.timedelta(seconds=current_time - prev_time)
                        text = "store hist info. for one frame, time = " + str(exe_time) + "\n"
                        timefile.write(text)
                        print(text)
                    elif flag == 0:
                        # do nothing
                        total_dicts.append(None)
                        total_RoIs.append(None)
                        total_RoIs_p.append(None)
                        raw_RoIs.append(None)
                        full_flag = False
                    else:
                        # need full inf for this frame
                        next_frames_cnt = j - 1
                        break

                idx_for_partial = 0

            else:
                # do inf for batch_i, hisinfo is in total_dicts[idx_for_partial]
                if idx_for_partial < next_frames_cnt:
                    # idx_for_partial(and skips)

                    RoIs = total_RoIs[idx_for_partial]
                    RoIs_p = total_RoIs_p[idx_for_partial]
                    raw_RoI = raw_RoIs[idx_for_partial]
                    if RoIs is not None:
                        RoIfile.write("RoI for frame " + str(batch_i + 1) + "\n")
                        for layer_i in range(16):
                            RoI = RoIs[layer_i]
                            RoIfile.write("***************\n")
                            if RoI is not None:
                                for temp in RoI:
                                    RoIfile.write(json.dumps(temp))
                                    RoIfile.write("\n")
                            else:
                                RoIfile.write("None\n")
                        prev_time = time.time()
                        _RoI_0 = []
                        _RoI_0.append([])
                        _RoI_0.append(raw_RoI)
                        for j in range(13):
                            temp = total_dicts[idx_for_partial][j]
                            locals()['_layer' + str(j + 1)] = temp
                            locals()['_RoI_' + str(j + 1)] = list()
                            locals()['_RoI_' + str(j + 1)].append(RoIs[j])
                            locals()['_RoI_' + str(j + 1)].append(RoIs_p[j])
                        current_time = time.time()
                        exe_time = datetime.timedelta(seconds=current_time - prev_time)
                        text = "load hist info. for frame" + str(batch_i + 1) + ", time = " + str(exe_time) + "\n"
                        timefile.write(text)

                        with torch.no_grad():
                            if True:
                                prev_time = time.time()
                                detections = model_2(input_imgs, _layer1, _layer2, _layer3, _layer4, _layer5, _layer6, _layer7, _layer8, _layer9, _layer10, _layer11, _layer12, _layer13, _RoI_0, _RoI_1, _RoI_2, _RoI_3, _RoI_4, _RoI_5, _RoI_6, _RoI_7, _RoI_8, _RoI_9, _RoI_10, _RoI_11, _RoI_12, _RoI_13)
                                current_time = time.time()
                                detections = non_max_suppression(detections, opt.conf_thres, opt.nms_thres) 
                                inference_time = datetime.timedelta(seconds=current_time - prev_time)
                                text = 'frame = ' + str(batch_i + 1) + ', inf. = partial' + '\n'
                                mapfile.write(text)
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
                        text = 'frame = ' + str(batch_i + 1) + ', inf. = skip' + '\n'
                        mapfile.write(text)
                        idx_for_partial += 1
                        if idx_for_partial == next_frames_cnt:
                            full_flag = True
    RoIfile.close()
    timefile.close()
    mapfile.close()
