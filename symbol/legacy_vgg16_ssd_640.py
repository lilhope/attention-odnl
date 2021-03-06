# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

import mxnet as mx
from common import legacy_conv_act_layer
from common import multibox_layer

def get_symbol_train(num_classes=1, nms_thresh=0.5, force_suppress=False, nms_topk=400):
    """
    Single-shot multi-box detection with VGG 16 layers ConvNet
    This is a modified version, with fc6/fc7 layers replaced by conv layers
    And the network is slightly smaller than original VGG 16 network
    This is a training network with losses

    Parameters:
    ----------
    num_classes: int
        number of object classes not including background
    nms_thresh : float
        non-maximum suppression threshold
    force_suppress : boolean
        whether suppress different class objects
    nms_topk : int
        apply NMS to top K detections

    Returns:
    ----------
    mx.Symbol
    """
    data = mx.symbol.Variable(name="data")
    label = mx.symbol.Variable(name="label")

    # group 1
    conv1_1 = mx.symbol.Convolution(
        data=data, kernel=(3, 3), pad=(1, 1), num_filter=64, name="conv1_1")
    relu1_1 = mx.symbol.Activation(data=conv1_1, act_type="relu", name="relu1_1")
    conv1_2 = mx.symbol.Convolution(
        data=relu1_1, kernel=(3, 3), pad=(1, 1), num_filter=64, name="conv1_2")
    relu1_2 = mx.symbol.Activation(data=conv1_2, act_type="relu", name="relu1_2")
    pool1 = mx.symbol.Pooling(
        data=relu1_2, pool_type="max", kernel=(2, 2), stride=(2, 2), name="pool1")
    # group 2
    conv2_1 = mx.symbol.Convolution(
        data=pool1, kernel=(3, 3), pad=(1, 1), num_filter=128, name="conv2_1")
    relu2_1 = mx.symbol.Activation(data=conv2_1, act_type="relu", name="relu2_1")
    conv2_2 = mx.symbol.Convolution(
        data=relu2_1, kernel=(3, 3), pad=(1, 1), num_filter=128, name="conv2_2")
    relu2_2 = mx.symbol.Activation(data=conv2_2, act_type="relu", name="relu2_2")
    pool2 = mx.symbol.Pooling(
        data=relu2_2, pool_type="max", kernel=(2, 2), stride=(2, 2), name="pool2")
    # group 3
    conv3_1 = mx.symbol.Convolution(
        data=pool2, kernel=(3, 3), pad=(1, 1), num_filter=256, name="conv3_1")
    relu3_1 = mx.symbol.Activation(data=conv3_1, act_type="relu", name="relu3_1")
    conv3_2 = mx.symbol.Convolution(
        data=relu3_1, kernel=(3, 3), pad=(1, 1), num_filter=256, name="conv3_2")
    relu3_2 = mx.symbol.Activation(data=conv3_2, act_type="relu", name="relu3_2")
    conv3_3 = mx.symbol.Convolution(
        data=relu3_2, kernel=(3, 3), pad=(1, 1), num_filter=256, name="conv3_3")
    relu3_3 = mx.symbol.Activation(data=conv3_3, act_type="relu", name="relu3_3")
    pool3 = mx.symbol.Pooling(
        data=relu3_3, pool_type="max", kernel=(2, 2), stride=(2, 2), \
        pooling_convention="full", name="pool3")
    # group 4
    conv4_1 = mx.symbol.Convolution(
        data=pool3, kernel=(3, 3), pad=(1, 1), num_filter=512, name="conv4_1")
    relu4_1 = mx.symbol.Activation(data=conv4_1, act_type="relu", name="relu4_1")
    conv4_2 = mx.symbol.Convolution(
        data=relu4_1, kernel=(3, 3), pad=(1, 1), num_filter=512, name="conv4_2")
    relu4_2 = mx.symbol.Activation(data=conv4_2, act_type="relu", name="relu4_2")
    conv4_3 = mx.symbol.Convolution(
        data=relu4_2, kernel=(3, 3), pad=(1, 1), num_filter=512, name="conv4_3")
    relu4_3 = mx.symbol.Activation(data=conv4_3, act_type="relu", name="relu4_3")
    pool4 = mx.symbol.Pooling(
        data=relu4_3, pool_type="max", kernel=(2, 2), stride=(2, 2), name="pool4")
    # group 5
    conv5_1 = mx.symbol.Convolution(
        data=pool4, kernel=(3, 3), pad=(1, 1), num_filter=512, name="conv5_1")
    relu5_1 = mx.symbol.Activation(data=conv5_1, act_type="relu", name="relu5_1")
    conv5_2 = mx.symbol.Convolution(
        data=relu5_1, kernel=(3, 3), pad=(1, 1), num_filter=512, name="conv5_2")
    relu5_2 = mx.symbol.Activation(data=conv5_2, act_type="relu", name="relu5_2")
    conv5_3 = mx.symbol.Convolution(
        data=relu5_2, kernel=(3, 3), pad=(1, 1), num_filter=512, name="conv5_3")
    relu5_3 = mx.symbol.Activation(data=conv5_3, act_type="relu", name="relu5_3")
    pool5 = mx.symbol.Pooling(
        data=relu5_3, pool_type="max", kernel=(2, 2), stride=(2, 2),name='pool5')
    
    P5 = mx.symbol.Convolution(data=pool5,num_filter=256,kernel=(1,1),stride=(1,1),pad=(0,0),name='P5')
    
    P5_topdown = mx.symbol.Deconvolution(data=P5,num_filter=256,kernel=(4,4),stride=(2,2),pad=(1,1),name='P5_topdown')
    P4_lateral = mx.symbol.Convolution(data=pool4,num_filter=256,kernel=(1,1),stride=(1,1),pad=(0,0),name='P4_lateral')
    P4 = P4_lateral + P5_topdown
    
    P4_topdown = mx.symbol.Deconvolution(data=P4,num_filter=256,kernel=(4,4),stride=(2,2),pad=(1,1),name='P4_topdown')
    P3_lateral = mx.symbol.Convolution(data=pool3,num_filter=256,kernel=(1,1),stride=(1,1),pad=(0,0),name='P3_lateral')
    P3 = P3_lateral + P4_topdown
    
    P3_topdown = mx.symbol.Deconvolution(data=P3,num_filter=256,kernel=(4,4),stride=(2,2),pad=(1,1),name='P3_topdown')
    P2_lateral = mx.symbol.Convolution(data=pool2,num_filter=256,kernel=(1,1),stride=(1,1),pad=(0,0),name='P2_lateral')
    P2 = P3_topdown + P2_lateral
    
    # specific parameters for VGG16 network
    from_layers = [P2,P3,P4,P5]
    sizes = [[0.01, .03], [.05,.07], [.09, .11], [.13, .15]]
    ratios = [[1,], [1,], [1,], [1,]]
    normalizations = [20, -1, -1, -1]
    steps = [ x / 640.0 for x in [4, 8, 16, 32]]
    num_channels = [256]

    loc_preds, cls_preds, anchor_boxes = multibox_layer(from_layers, \
        num_classes, sizes=sizes, ratios=ratios, normalization=normalizations, \
        num_channels=num_channels, clip=False, interm_layer=0, steps=steps)

    tmp = mx.contrib.symbol.MultiBoxTarget(
        *[anchor_boxes, label, cls_preds], overlap_threshold=.5, \
        ignore_label=-1, negative_mining_ratio=3, minimum_negative_samples=0, \
        negative_mining_thresh=.5, variances=(0.1, 0.1, 0.2, 0.2),
        name="multibox_target")
    loc_target = tmp[0]
    loc_target_mask = tmp[1]
    cls_target = tmp[2]

    cls_prob = mx.symbol.SoftmaxOutput(data=cls_preds, label=cls_target, \
        ignore_label=-1, use_ignore=True, grad_scale=1., multi_output=True, \
        normalization='valid', name="cls_prob")
    loc_loss_ = mx.symbol.smooth_l1(name="loc_loss_", \
        data=loc_target_mask * (loc_preds - loc_target), scalar=1.0)
    loc_loss = mx.symbol.MakeLoss(loc_loss_, grad_scale=1., \
        normalization='valid', name="loc_loss")

    # monitoring training status
    cls_label = mx.symbol.MakeLoss(data=cls_target, grad_scale=0, name="cls_label")
    det = mx.contrib.symbol.MultiBoxDetection(*[cls_prob, loc_preds, anchor_boxes], \
        name="detection", nms_threshold=nms_thresh, force_suppress=force_suppress,
        variances=(0.1, 0.1, 0.2, 0.2), nms_topk=nms_topk)
    det = mx.symbol.MakeLoss(data=det, grad_scale=0, name="det_out")

    # group output
    out = mx.symbol.Group([cls_prob, loc_loss, cls_label, det])
    return out

def get_symbol(num_classes=20, nms_thresh=0.5, force_suppress=False, nms_topk=400):
    """
    Single-shot multi-box detection with VGG 16 layers ConvNet
    This is a modified version, with fc6/fc7 layers replaced by conv layers
    And the network is slightly smaller than original VGG 16 network
    This is the detection network

    Parameters:
    ----------
    num_classes: int
        number of object classes not including background
    nms_thresh : float
        threshold of overlap for non-maximum suppression
    force_suppress : boolean
        whether suppress different class objects
    nms_topk : int
        apply NMS to top K detections

    Returns:
    ----------
    mx.Symbol
    """
    net = get_symbol_train(num_classes)
    cls_preds = net.get_internals()["multibox_cls_pred_output"]
    loc_preds = net.get_internals()["multibox_loc_pred_output"]
    anchor_boxes = net.get_internals()["multibox_anchors_output"]

    cls_prob = mx.symbol.SoftmaxActivation(data=cls_preds, mode='channel', \
        name='cls_prob')
    out = mx.contrib.symbol.MultiBoxDetection(*[cls_prob, loc_preds, anchor_boxes], \
        name="detection", nms_threshold=nms_thresh, force_suppress=force_suppress,
        variances=(0.1, 0.1, 0.2, 0.2), nms_topk=nms_topk)
    return out
