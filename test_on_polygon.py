
import cv2
import pyclipper
import numpy as np


def shrink(bboxes, offset):
    shrinked_bboxes = []
    for bbox in bboxes:
        pco = pyclipper.PyclipperOffset()
        pco.AddPath(bbox, pyclipper.JT_MITER, pyclipper.ET_CLOSEDPOLYGON)
        shrinked_bbox = pco.Execute(offset)
        """if len(shrinked_bboxes) == 0:
            shrinked_bboxes.append(bbox)
            continue
        shrinked_bbox = np.array(shrinked_bbox)[0]
        if shrinked_bbox.shape[0] <= 2:
            shrinked_bboxes.append(bbox)
            continue"""
        shrinked_bboxes.append(shrinked_bbox)
    return shrinked_bboxes

line = [20, 10, 30, 10, 30, 30, 10, 30, 10, 20, 0, 20, 0, 0, 20, 0]
gt_box = []
box_info = np.array(line).reshape((-1, 2))
gt_box.append(box_info)
#ori_mask_img = np.zeros((30, 30))
#ori_mask_img = cv2.fillPoly(ori_mask_img, [box_info], (255))
#shrink_mask_img = np.zeros((30, 30))
new_gt_box = shrink(gt_box, 2)
print(new_gt_box)
#for new_box in new_gt_box:
#    shrink_mask_img = cv2.fillPoly(shrink_mask_img, [new_box], (255))
#cv2.imshow("ori_mask", ori_mask_img)
#cv2.imshow("shrink_mask", shrink_mask_img)
#cv2.waitKey(0)






