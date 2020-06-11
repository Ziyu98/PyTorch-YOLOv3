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
    for line in lines:
        if not line.startswith('['):
            key, value = line.split("=")
            value = int(value.strip())
            if key.rstrip() == "filters":
                channels.append(value)
            elif key.rstrip() == "size":
                layersize.append(value)
    return channels, layersize
        
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

def get_RoIs(frame_id):
    distance = 4
    start_point = 3
    with open("/i3c/hpcl/zjy5087/PyTorch-YOLOv3/store_{}_{}/res_for_frame{}/RoI.txt".format(distance, start_point, frame_id)) as f:
        _list = []
        _lists = []
        for line in f:
            if not line.startswith("*"):
                if line.startswith("["):
                    _line = line[1 : -2]
                    _line = _line.split(",")
                    temp = [int(float(_line[i])) for i in range(len(_line))]
                    _list.append(temp)
            else:
                if len(_list) != 0:
                    _lists.append(_list)
                    _list = []
        _lists.append(_list)
    return _lists

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




path = "config_for_comp.cfg"
channels, layersize = parse_config(path)
cnt = len(channels)
resfile = open("num_of_comp.txt", "w")
b1 = 0
b3 = 0
p1 = 0
p3 = 0
count = 0
for batch_i in range(3, 499):
    if (batch_i - 2) % 4 != 0:
        baseline_3 = 0
        baseline_1 = 0
        partial_3 = 0
        partial_1 = 0
        j = 0
        RoIs = get_RoIs(batch_i + 1)
        for i in range(cnt):
            if channels[i] > 0:
                # 3*3
                RoI = RoIs[j]
                j += 1
                _area = cal_area(RoI)
                partial_3 += _area
                baseline_3 += layersize[i] * layersize[i]

            else:
                # 1*1
                gt_boxes = RoI
                new_gt_boxes = []
                for temp in gt_boxes:
                    temp_RoI = np.asarray(temp).reshape(-1, 2)
                    new_gt_boxes.append(temp_RoI)
                gt_boxes = shrink(new_gt_boxes, 2)
                RoI = gt_boxes
                _area = cal_area(RoI)
                partial_1 += _area
                baseline_1 += layersize[i] * layersize[i]
        text = str(baseline_1) + "|" + str(baseline_3) + "|" + str(partial_1) + "|" + str(partial_3) + "\n"
        resfile.write(text)
        b1 += baseline_1
        b3 += baseline_3
        p1 += partial_1
        p3 += partial_3
        count += 1
        print(baseline_1, baseline_3, partial_1, partial_3)
    else:
        print("error")
resfile.close()
print(b1 / count, b3 / count, p1 / count, p3 / count)