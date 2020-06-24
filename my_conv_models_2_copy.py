from __future__ import division

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
from PIL import Image, ImageDraw
from shapely import geometry
import pyclipper
import time

from utils.parse_config import *
from utils.utils import build_targets, to_cpu, non_max_suppression
from new_im2col import im2col

import matplotlib.pyplot as plt
import matplotlib.patches as patches
def my_conv(X, W, b, RoIs, layersize, stride=1, padding=1):
    # layersize and RoIs should be the layersize and RoIs of next layer
    #p_t = time.time()
    n_filters, d_filters, h_filters, w_filters = W.shape
    n_x, d_x, h_x, w_x = X.shape
    h_out = (h_x - h_filters + 2 * padding) // stride + 1
    w_out = (w_x - w_filters + 2 * padding) // stride + 1
    #print('X shape:', X.shape, 'W shape:', W.shape)
    #print(h_out, w_out)
    #if not h_out.is_integer() or not w_out.is_integer():
        #raise Exception('Invalid output dimension!')
    
    X_col = im2col(X, h_filters, w_filters, stride=stride, pad=padding)
    W_col = W.reshape(n_filters, -1).T
    #c_t = time.time()
    #print('t_1: %s' % (c_t - p_t))
    #p_t = time.time()
    if RoIs is not None:
        res = np.zeros((W_col.shape[1], X_col.shape[0])).astype('float32')
        img = Image.new('L', (layersize, layersize), 0)
        for poly in RoIs:
            ImageDraw.Draw(img).polygon(poly, outline=1, fill=1)
        img = np.array(img)
        idx = np.transpose(np.nonzero(img))
        _list = []
        for [x, y] in idx:
            temp = x * layersize + y
            _list.append(temp)
        X_col = X_col[_list, :]
        #c_t = time.time()
        #print('process X_col time: %s' % (c_t - p_t))
        #p_t = time.time()
        out = np.dot(X_col, W_col).T
        #c_t = time.time()
        #print('partial t_3: %s' % (c_t - p_t))
        res[:, _list] = out
    else:
        res = np.dot(X_col, W_col).T
        #c_t = time.time()
        #print('full t_3: %s' % (c_t - p_t))
    b = b.reshape(-1, 1)
    res += b
    res = res.reshape(n_filters, h_out, w_out, n_x)
    res = res.transpose(3, 0, 1, 2)
    return res

def create_modules_2(module_defs):
    """
    Constructs module list of layer blocks from module configuration in module_defs
    """
    hyperparams = module_defs.pop(0)
    output_filters = [int(hyperparams["channels"])]
    module_list = nn.ModuleList()
    weight_nums = {}
    for module_i, module_def in enumerate(module_defs):
        modules = nn.Sequential()

        if module_def["type"] == "convolutional":
            bn = int(module_def["batch_normalize"])
            filters = int(module_def["filters"])
            kernel_size = int(module_def["size"])
            pad = (kernel_size - 1) // 2
            n_size = int(module_def["next_size"])
            """modules.add_module(
                f"conv_{module_i}",
                nn.Conv2d(
                    in_channels=output_filters[-1],
                    out_channels=filters,
                    kernel_size=kernel_size,
                    stride=int(module_def["stride"]),
                    padding=pad,
                    bias=not bn,
                ),
            )"""
            weight_nums[module_i] = [filters, output_filters[-1], kernel_size, kernel_size]
            my_conv_layer = MyConvLayer(n_size, int(module_def["stride"]), pad)
            modules.add_module(f"conv_{module_i}", my_conv_layer)
            if bn:
                modules.add_module(f"batch_norm_{module_i}", nn.BatchNorm2d(filters, momentum=0.9, eps=1e-5))
            if module_def["activation"] == "leaky":
                modules.add_module(f"leaky_{module_i}", nn.LeakyReLU(0.1))

        elif module_def["type"] == "maxpool":
            kernel_size = int(module_def["size"])
            stride = int(module_def["stride"])
            if kernel_size == 2 and stride == 1:
                modules.add_module(f"_debug_padding_{module_i}", nn.ZeroPad2d((0, 1, 0, 1)))
            maxpool = nn.MaxPool2d(kernel_size=kernel_size, stride=stride, padding=int((kernel_size - 1) // 2))
            modules.add_module(f"maxpool_{module_i}", maxpool)

        elif module_def["type"] == "upsample":
            upsample = Upsample(scale_factor=int(module_def["stride"]), mode="nearest")
            modules.add_module(f"upsample_{module_i}", upsample)

        elif module_def["type"] == "route":
            layers = [int(x) for x in module_def["layers"].split(",")]
            filters = sum([output_filters[1:][i] for i in layers])
            modules.add_module(f"route_{module_i}", EmptyLayer())

        elif module_def["type"] == "shortcut":
            filters = output_filters[1:][int(module_def["from"])]
            modules.add_module(f"shortcut_{module_i}", EmptyLayer())

        elif module_def["type"] == "yolo":
            anchor_idxs = [int(x) for x in module_def["mask"].split(",")]
            # Extract anchors
            anchors = [int(x) for x in module_def["anchors"].split(",")]
            anchors = [(anchors[i], anchors[i + 1]) for i in range(0, len(anchors), 2)]
            anchors = [anchors[i] for i in anchor_idxs]
            num_classes = int(module_def["classes"])
            img_size = int(hyperparams["height"])
            # Define detection layer
            yolo_layer = YOLOLayer_2(anchors, num_classes, img_size)
            modules.add_module(f"yolo_{module_i}", yolo_layer)
        
        elif module_def["type"] == "padding":
            layersize = int(module_def["layersize"])
            layertype = int(module_def["layertype"])
            padding_layer = PADLayer(layersize, layertype)
            modules.add_module(f"padding_{module_i}", padding_layer)
        # Register module list and number of output filters
        module_list.append(modules)
        output_filters.append(filters)

    return hyperparams, module_list, weight_nums 


class Upsample(nn.Module):
    """ nn.Upsample is deprecated """

    def __init__(self, scale_factor, mode="nearest"):
        super(Upsample, self).__init__()
        self.scale_factor = scale_factor
        self.mode = mode

    def forward(self, x):
        x = F.interpolate(x, scale_factor=self.scale_factor, mode=self.mode)
        return x


class EmptyLayer(nn.Module):
    """Placeholder for 'route' and 'shortcut' layers"""

    def __init__(self):
        super(EmptyLayer, self).__init__()

class MyConvLayer(nn.Module):
    def __init__(self, next_size, stride, padding):
        super(MyConvLayer, self).__init__()
        self.stride = stride
        self.padding = padding
        self.next_size = next_size
    
    def forward(self, input_x, input_w, input_b, next_RoI):
        s = self.stride
        p = self.padding
        n_s = self.next_size
        # input_x and input_w to numpy
        input_x, input_w, input_b = input_x.numpy(), input_w.numpy(), input_b.numpy()
        out = my_conv(input_x, input_w, input_b, next_RoI, n_s, s, p)
        # out to tensor
        out = torch.from_numpy(out) 
        return out

class PADLayer(nn.Module):
    def __init__(self, layersize, layertype):
        super(PADLayer, self).__init__()
        self.layersize = layersize
        self.layertype = layertype
    
    def forward(self, input_x, RoIs, his_info):
        if RoIs is not None:
            width = self.layersize
            height = self.layersize
            shape_3 = input_x.shape[1]
            RoI = []
            img = Image.new('L', (width, height), 0)
            for poly in RoIs:
                ImageDraw.Draw(img).polygon(poly, outline=1, fill=1)
            img = np.array(img)
            _img = np.repeat(img[np.newaxis, :, :], shape_3, axis=0)
            _img = _img[np.newaxis, :, :, :]
            mask = torch.tensor(_img)
            input_x = torch.mul(input_x, mask)
            for key in his_info:
                [x, y] = key
                value = his_info[key]
                input_x[0, :, x, y] = value
        return input_x

class YOLOLayer_2(nn.Module):
    """Detection layer"""

    def __init__(self, anchors, num_classes, img_dim=416):
        super(YOLOLayer_2, self).__init__()
        self.anchors = anchors
        self.num_anchors = len(anchors)
        self.num_classes = num_classes
        self.ignore_thres = 0.5
        self.mse_loss = nn.MSELoss()
        self.bce_loss = nn.BCELoss()
        self.obj_scale = 1
        self.noobj_scale = 100
        self.metrics = {}
        self.img_dim = img_dim
        self.grid_size = 0  # grid size

    def compute_grid_offsets(self, grid_size, cuda=True):
        self.grid_size = grid_size
        g = self.grid_size
        FloatTensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
        self.stride = self.img_dim / self.grid_size
        # Calculate offsets for each grid
        self.grid_x = torch.arange(g).repeat(g, 1).view([1, 1, g, g]).type(FloatTensor)
        self.grid_y = torch.arange(g).repeat(g, 1).t().view([1, 1, g, g]).type(FloatTensor)
        self.scaled_anchors = FloatTensor([(a_w / self.stride, a_h / self.stride) for a_w, a_h in self.anchors])
        self.anchor_w = self.scaled_anchors[:, 0:1].view((1, self.num_anchors, 1, 1))
        self.anchor_h = self.scaled_anchors[:, 1:2].view((1, self.num_anchors, 1, 1))

    def forward(self, x, targets=None, img_dim=None):

        # Tensors for cuda support
        FloatTensor = torch.cuda.FloatTensor if x.is_cuda else torch.FloatTensor
        LongTensor = torch.cuda.LongTensor if x.is_cuda else torch.LongTensor
        ByteTensor = torch.cuda.ByteTensor if x.is_cuda else torch.ByteTensor

        self.img_dim = img_dim
        num_samples = x.size(0)
        grid_size = x.size(2)

        prediction = (
            x.view(num_samples, self.num_anchors, self.num_classes + 5, grid_size, grid_size)
            .permute(0, 1, 3, 4, 2)
            .contiguous()
        )

        # Get outputs
        x = torch.sigmoid(prediction[..., 0])  # Center x
        y = torch.sigmoid(prediction[..., 1])  # Center y
        w = prediction[..., 2]  # Width
        h = prediction[..., 3]  # Height
        pred_conf = torch.sigmoid(prediction[..., 4])  # Conf
        pred_cls = torch.sigmoid(prediction[..., 5:])  # Cls pred.

        # If grid size does not match current we compute new offsets
        if grid_size != self.grid_size:
            self.compute_grid_offsets(grid_size, cuda=x.is_cuda)

        # Add offset and scale with anchors
        pred_boxes = FloatTensor(prediction[..., :4].shape)
        pred_boxes[..., 0] = x.data + self.grid_x
        pred_boxes[..., 1] = y.data + self.grid_y
        pred_boxes[..., 2] = torch.exp(w.data) * self.anchor_w
        pred_boxes[..., 3] = torch.exp(h.data) * self.anchor_h

        output = torch.cat(
            (
                pred_boxes.view(num_samples, -1, 4) * self.stride,
                pred_conf.view(num_samples, -1, 1),
                pred_cls.view(num_samples, -1, self.num_classes),
            ),
            -1,
        )

        if targets is None:
            return output, 0
        else:
            iou_scores, class_mask, obj_mask, noobj_mask, tx, ty, tw, th, tcls, tconf = build_targets(
                pred_boxes=pred_boxes,
                pred_cls=pred_cls,
                target=targets,
                anchors=self.scaled_anchors,
                ignore_thres=self.ignore_thres,
            )

            # Loss : Mask outputs to ignore non-existing objects (except with conf. loss)
            loss_x = self.mse_loss(x[obj_mask], tx[obj_mask])
            loss_y = self.mse_loss(y[obj_mask], ty[obj_mask])
            loss_w = self.mse_loss(w[obj_mask], tw[obj_mask])
            loss_h = self.mse_loss(h[obj_mask], th[obj_mask])
            loss_conf_obj = self.bce_loss(pred_conf[obj_mask], tconf[obj_mask])
            loss_conf_noobj = self.bce_loss(pred_conf[noobj_mask], tconf[noobj_mask])
            loss_conf = self.obj_scale * loss_conf_obj + self.noobj_scale * loss_conf_noobj
            loss_cls = self.bce_loss(pred_cls[obj_mask], tcls[obj_mask])
            total_loss = loss_x + loss_y + loss_w + loss_h + loss_conf + loss_cls

            # Metrics
            cls_acc = 100 * class_mask[obj_mask].mean()
            conf_obj = pred_conf[obj_mask].mean()
            conf_noobj = pred_conf[noobj_mask].mean()
            conf50 = (pred_conf > 0.5).float()
            iou50 = (iou_scores > 0.5).float()
            iou75 = (iou_scores > 0.75).float()
            detected_mask = conf50 * class_mask * tconf
            precision = torch.sum(iou50 * detected_mask) / (conf50.sum() + 1e-16)
            recall50 = torch.sum(iou50 * detected_mask) / (obj_mask.sum() + 1e-16)
            recall75 = torch.sum(iou75 * detected_mask) / (obj_mask.sum() + 1e-16)

            self.metrics = {
                "loss": to_cpu(total_loss).item(),
                "x": to_cpu(loss_x).item(),
                "y": to_cpu(loss_y).item(),
                "w": to_cpu(loss_w).item(),
                "h": to_cpu(loss_h).item(),
                "conf": to_cpu(loss_conf).item(),
                "cls": to_cpu(loss_cls).item(),
                "cls_acc": to_cpu(cls_acc).item(),
                "recall50": to_cpu(recall50).item(),
                "recall75": to_cpu(recall75).item(),
                "precision": to_cpu(precision).item(),
                "conf_obj": to_cpu(conf_obj).item(),
                "conf_noobj": to_cpu(conf_noobj).item(),
                "grid_size": grid_size,
            }

            return output, total_loss


class Darknet_2(nn.Module):
    """YOLOv3 object detection model"""

    def __init__(self, config_path, img_size=416):
        super(Darknet_2, self).__init__()
        self.module_defs = parse_model_config(config_path)
        self.hyperparams, self.module_list, self.weight_nums = create_modules_2(self.module_defs)
        self.yolo_layers = [layer[0] for layer in self.module_list if hasattr(layer[0], "metrics")]
        self.img_size = img_size
        self.seen = 0
        self.header_info = np.array([0, 0, 0, self.seen, 0], dtype=np.int32)

    def forward(self, x, weights_dict, bias_dict, input_n1, input_n2, input_n3, input_n4, input_n5, input_n6, input_n7, input_n8, input_n9, input_n10, input_n11, input_n12, input_n13, input_n14, input_n15, input_n16, input_n17, input_n18, input_n19, input_n20, input_n21, input_n22, input_n23, input_n24, input_n25, input_n26, input_n27, input_n28, input_n29, input_n30, input_n32, input_n34, input_n37, input_n39, input_n41, input_n44, input_n46, input_n48, RoI_1, RoI_2, RoI_3, RoI_4, RoI_5, RoI_6, RoI_7, RoI_8, RoI_9, RoI_10, RoI_11, RoI_12, RoI_13, RoI_14, RoI_15, RoI_16, RoI_17, RoI_18, RoI_19, RoI_20, RoI_21, RoI_22, RoI_23, RoI_24, RoI_25, RoI_26, RoI_27, RoI_28, RoI_29, RoI_30, RoI_32, RoI_34, RoI_37, RoI_39, RoI_41, RoI_44, RoI_46, RoI_48, targets=None):
        img_dim = x.shape[2]
        loss = 0
        layer_outputs, yolo_outputs = [], []
        for i, (module_def, module) in enumerate(zip(self.module_defs, self.module_list)):
            if module_def["type"] in ["upsample", "maxpool"]:
                x = module(x)
            elif module_def["type"] == "convolutional":
                w = weights_dict[i]
                bn = int(module_def["batch_normalize"])
                if not bn:
                    b = bias_dict[i]
                else:
                    b = torch.zeros(w.shape[0])
                _idx = int(module_def["next_RoI"])
                if _idx == 0:
                    RoI = None
                else:
                    RoI = locals()['RoI_' + str(_idx)]
                #p_t = time.time()
                x = module[0](x, w, b, RoI)
                #c_t = time.time()
                #print('partial, layer %d, conv time: %s' % (i, c_t - p_t))
                if bn:
                    #p_t = time.time()
                    x = module[1](x)
                    #c_t = time.time()
                    #print('partial, layer %d, bn time: %s' % (i, c_t - p_t))
                if module_def["activation"] == "leaky":
                    #p_t = time.time()
                    x = module[2](x)
                    #c_t = time.time()
                    #print('partial, layer %d, leaky time: %s' % (i, c_t - p_t))
            elif module_def["type"] == "route":
                x = torch.cat([layer_outputs[int(layer_i)] for layer_i in module_def["layers"].split(",")], 1)
            elif module_def["type"] == "shortcut":
                layer_i = int(module_def["from"])
                x = layer_outputs[-1] + layer_outputs[layer_i]
            elif module_def["type"] == "yolo":
                #p_t = time.time()
                x, layer_loss = module[0](x, targets, img_dim)
                loss += layer_loss
                yolo_outputs.append(x)
                #c_t = time.time()
                #print('partial, layer %d, yolo time: %s' % (i, c_t - p_t))
            elif module_def["type"] == "padding":
                #p_t = time.time()
                _his_index = int(module_def["his_idx"])
                temp = locals()['input_n' + str(_his_index)]
                RoI = locals()['RoI_' + str(_his_index)]
                x = module[0](x, RoI, temp)
                #c_t = time.time()
                #print('partial, layer %d, padding time: %s' % (i, c_t - p_t))
            layer_outputs.append(x)
        yolo_outputs = to_cpu(torch.cat(yolo_outputs, 1))
        return yolo_outputs if targets is None else (loss, yolo_outputs)

    def load_darknet_weights(self, weights_path):
        """Parses and loads the weights stored in 'weights_path'"""
        my_weights_dict = {}
        my_bias_dict = {}
        # Open the weights file
        with open(weights_path, "rb") as f:
            header = np.fromfile(f, dtype=np.int32, count=5)  # First five are header values
            self.header_info = header  # Needed to write header when saving weights
            self.seen = header[3]  # number of images seen during training
            weights = np.fromfile(f, dtype=np.float32)  # The rest are weights

        # Establish cutoff for loading backbone weights
        cutoff = None
        if "darknet53.conv.74" in weights_path:
            cutoff = 75

        ptr = 0
        for i, (module_def, module) in enumerate(zip(self.module_defs, self.module_list)):
            if i == cutoff:
                break
            if module_def["type"] == "convolutional":
                conv_layer = module[0]
                if module_def["batch_normalize"]:
                    # Load BN bias, weights, running mean and running variance
                    bn_layer = module[1]
                    num_b = bn_layer.bias.numel()  # Number of biases
                    # Bias
                    bn_b = torch.from_numpy(weights[ptr : ptr + num_b]).view_as(bn_layer.bias)
                    bn_layer.bias.data.copy_(bn_b)
                    ptr += num_b
                    # Weight
                    bn_w = torch.from_numpy(weights[ptr : ptr + num_b]).view_as(bn_layer.weight)
                    bn_layer.weight.data.copy_(bn_w)
                    ptr += num_b
                    # Running Mean
                    bn_rm = torch.from_numpy(weights[ptr : ptr + num_b]).view_as(bn_layer.running_mean)
                    bn_layer.running_mean.data.copy_(bn_rm)
                    ptr += num_b
                    # Running Var
                    bn_rv = torch.from_numpy(weights[ptr : ptr + num_b]).view_as(bn_layer.running_var)
                    bn_layer.running_var.data.copy_(bn_rv)
                    ptr += num_b
                else:
                    # Load conv. bias
                    num_b = self.weight_nums[i][0]
                    conv_b = torch.from_numpy(weights[ptr : ptr + num_b])
                    my_bias_dict[i] = conv_b
                    ptr += num_b
                # Load conv. weights
                num_w = np.prod(self.weight_nums[i])
                conv_w = weights[ptr : ptr + num_w].reshape(self.weight_nums[i])
                conv_w = torch.from_numpy(conv_w)
                my_weights_dict[i] = conv_w
                ptr += num_w
        return my_weights_dict, my_bias_dict
        
    def save_darknet_weights(self, path, cutoff=-1):
        """
            @:param path    - path of the new weights file
            @:param cutoff  - save layers between 0 and cutoff (cutoff = -1 -> all are saved)
        """
        fp = open(path, "wb")
        self.header_info[3] = self.seen
        self.header_info.tofile(fp)

        # Iterate through layers
        for i, (module_def, module) in enumerate(zip(self.module_defs[:cutoff], self.module_list[:cutoff])):
            if module_def["type"] == "convolutional":
                conv_layer = module[0]
                # If batch norm, load bn first
                if module_def["batch_normalize"]:
                    bn_layer = module[1]
                    bn_layer.bias.data.cpu().numpy().tofile(fp)
                    bn_layer.weight.data.cpu().numpy().tofile(fp)
                    bn_layer.running_mean.data.cpu().numpy().tofile(fp)
                    bn_layer.running_var.data.cpu().numpy().tofile(fp)
                # Load conv bias
                else:
                    conv_layer.bias.data.cpu().numpy().tofile(fp)
                # Load conv weights
                conv_layer.weight.data.cpu().numpy().tofile(fp)

        fp.close()
