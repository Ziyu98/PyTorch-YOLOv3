import numpy as np
import pyclipper
import shapely.geometry as sg

def parse_config(path):
    file = open(path, 'r')
    lines = file.read().split('\n')
    lines = [x for x in lines if x and not x.startswith('#')]
    lines = [x.rstrip().lstrip() for x in lines]
    channels = []
    layersize = []
    roi_idx = []
    for line in lines:
        if not line.startswith('['):
            key, value = line.split("=")
            value = int(value.strip())
            if key.rstrip() == "filters":
                channels.append(value)
            elif key.rstrip() == "size":
                layersize.append(value)
            elif key.rstrip() == "RoI_idx":
                roi_idx.append(value)
    return channels, layersize, roi_idx
        
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

def get_RoIs(path):
    with open(path) as f:
        RoIs = []
        total_RoIs = []
        _list = []
        for line in f:
            if line.startswith("R"):
                if len(_list) != 0:
                    RoIs.append(_list)
                    _list = []
                if len(RoIs) != 0:
                    total_RoIs.append(RoIs)
                    RoIs = []
            elif not line.startswith("*"):
                if line.startswith("["):
                    _line = line[1 : -2]
                    _line = _line.split(",")
                    temp = [int(float(_line[i])) for i in range(len(_line))]
                    _list.append(temp)
                elif line[0:4] == "None":
                    _list.append(None)
            else:
                if len(_list) != 0:
                    RoIs.append(_list)
                    _list = []
    return total_RoIs

def cal_area(RoI):
    _area = 0
    poly = sg.Polygon([[0, 0], [0, 0], [0, 0]])
    for temp in RoI:
        _list = [[temp[2*i], temp[2*i+1]] for i in range(int(len(temp)/2))]
        if poly.area == 0:
            poly = sg.Polygon(_list)
            if not poly.is_valid:
                poly = poly.buffer(0)
        else:
            temp_poly = sg.Polygon(_list)
            if not temp_poly.is_valid:
                temp_poly = temp_poly.buffer(0)
            poly.union(temp_poly)
    return poly.area




path_cfg = "../PyTorch-YOLOv3_partial_inf/config_for_comp.cfg"
channels, layersize, roi_idx = parse_config(path_cfg)
cnt = len(channels)
path_RoI = "/i3c/hpcl/zjy5087/PyTorch-YOLOv3/res_for_s4_e2/res_for_s4_e2_dist9_0/RoI"
resfile = open("num_of_comp.txt", "w")
b1 = 0
b3 = 0
p1 = 0
p3 = 0
count = 0
total_RoIs = get_RoIs(path_RoI)
for RoIs in total_RoIs:
    baseline_3 = 0
    baseline_1 = 0
    partial_3 = 0
    partial_1 = 0
    for i in range(cnt):
        if channels[i] > 0:
            # 3*3
            j = roi_idx[i]
            RoI = RoIs[j]
            if RoI != [None]:
                _area = cal_area(RoI)
            else:
                _area = layersize[i] * layersize[i]
            partial_3 += _area * abs(channels[i])
            baseline_3 += layersize[i] * layersize[i] * abs(channels[i])

        else:
            # 1*1
            j = roi_idx[i]
            gt_boxes = RoIs[j]
            if gt_boxes != [None]:
                new_gt_boxes = []
                for temp in gt_boxes:
                    temp_RoI = np.asarray(temp).reshape(-1, 2)
                    new_gt_boxes.append(temp_RoI)
                gt_boxes = shrink(new_gt_boxes, 2)
                RoI = gt_boxes
                _area = cal_area(RoI)
            else:
                _area = layersize[i] * layersize[i]
            partial_1 += _area * abs(channels[i])
            baseline_1 += layersize[i] * layersize[i] * abs(channels[i])
    text = str(baseline_1) + "|" + str(baseline_3) + "|" + str(partial_1) + "|" + str(partial_3) + "\n"
    resfile.write(text)
    b1 += baseline_1
    p1 += partial_1
    b3 += baseline_3
    p3 += partial_3
    count += 1
resfile.close()
print(b1/count, b3/count, p1/count, p3/count)
