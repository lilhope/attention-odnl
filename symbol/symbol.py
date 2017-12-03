#!/usr/bin/env python2
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
from config.config import cfg

#global variable
eps = 2e-5
use_global_stats = True
workspace = 512
stack = mx.rnn.SequentialRNNCell()
for i in range(2):
    stack.add(mx.rnn.GRUCell(num_hidden=512,prefix='gru_l%d'%i))

def get_rnn_feat(seq_len,expression):
    """sequence to vector"""
    embed= mx.symbol.Embedding(data=expression,input_dim=72704,output_dim=300,name='embed')
    outputs,states = stack.unroll(seq_len,inputs=embed,merge_outputs=False)
    output = outputs[-1]
    e_fc1 = mx.symbol.FullyConnected(data=output,num_hidden=256,name='e_fc1')
    e_relu1 = mx.symbol.Activation(data=e_fc1,act_type='relu',name='e_relu1')
    return e_relu1
    
def get_cnn_feat(expression):
    
    #embed= mx.symbol.Embedding(data=expression,input_dim=72000,output_dim=300,name='embed')
    pass

def symbol_vgg(data):
    # group 1
    
    conv1_1 = mx.symbol.Convolution(
        data=data, kernel=(3, 3), pad=(1, 1), num_filter=64, name="conv1_1")
    relu1_1 = mx.symbol.Activation(data=conv1_1, act_type="relu", name="relu1_1")
    #bn1_1 = mx.symbol.BatchNorm(data=relu1_1,eps=2e-5,momentum=0.9,fix_gammaa=False,name='bn1_1')
    conv1_2 = mx.symbol.Convolution(
        data=relu1_1, kernel=(3, 3), pad=(1, 1), num_filter=64, name="conv1_2")
    relu1_2 = mx.symbol.Activation(data=conv1_2, act_type="relu", name="relu1_2")
    #bn1_1 = mx.symbol.BatchNorm(data=relu1_2,eps=2e-5,momentum=0.9,fix_gammaa=False,name='bn1_2')
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
    
    return relu4_3

def symbol_vgg_bn(data):
    conv1_1 = mx.symbol.Convolution(
        data=data, kernel=(3, 3), pad=(1, 1), num_filter=64, name="conv1_1")
    bn1_1 = mx.symbol.BatchNorm(data=conv1_1,eps=2e-5,momentum=0.9,fix_gamma=False,name='bn1_1')
    relu1_1 = mx.symbol.Activation(data=bn1_1, act_type="relu", name="relu1_1")
    
    conv1_2 = mx.symbol.Convolution(
        data=relu1_1, kernel=(3, 3), pad=(1, 1), num_filter=64, name="conv1_2")
    bn1_2 = mx.symbol.BatchNorm(data=conv1_2,eps=2e-5,momentum=0.9,fix_gamma=False,name='bn1_2')
    relu1_2 = mx.symbol.Activation(data=bn1_2, act_type="relu", name="relu1_2")

    pool1 = mx.symbol.Pooling(
        data=relu1_2, pool_type="max", kernel=(2, 2), stride=(2, 2), name="pool1")
    # group 2
    conv2_1 = mx.symbol.Convolution(
        data=pool1, kernel=(3, 3), pad=(1, 1), num_filter=128, name="conv2_1")
    bn2_1 = mx.symbol.BatchNorm(data=conv2_1,eps=2e-5,momentum=0.9,fix_gamma=False,name='bn2_1')
    relu2_1 = mx.symbol.Activation(data=bn2_1, act_type="relu", name="relu2_1")
    conv2_2 = mx.symbol.Convolution(
        data=relu2_1, kernel=(3, 3), pad=(1, 1), num_filter=128, name="conv2_2")
    bn2_2 = mx.symbol.BatchNorm(data=conv2_2,eps=2e-5,momentum=0.9,fix_gamma=False,name='bn2_2')
    relu2_2 = mx.symbol.Activation(data=bn2_2, act_type="relu", name="relu2_2")
    pool2 = mx.symbol.Pooling(
        data=relu2_2, pool_type="max", kernel=(2, 2), stride=(2, 2), name="pool2")
    # group 3
    conv3_1 = mx.symbol.Convolution(
        data=pool2, kernel=(3, 3), pad=(1, 1), num_filter=256, name="conv3_1")
    bn3_1 = mx.symbol.BatchNorm(data=conv3_1,eps=2e-5,momentum=0.9,fix_gamma=False,name='bn3_1')
    relu3_1 = mx.symbol.Activation(data=bn3_1, act_type="relu", name="relu3_1")
    conv3_2 = mx.symbol.Convolution(
        data=relu3_1, kernel=(3, 3), pad=(1, 1), num_filter=256, name="conv3_2")
    bn3_2 = mx.symbol.BatchNorm(data=conv3_2,eps=2e-5,momentum=0.9,fix_gamma=False,name='bn3_2')
    relu3_2 = mx.symbol.Activation(data=bn3_2, act_type="relu", name="relu3_2")
    conv3_3 = mx.symbol.Convolution(
        data=relu3_2, kernel=(3, 3), pad=(1, 1), num_filter=256, name="conv3_3")
    bn3_3 = mx.symbol.BatchNorm(data=conv3_3,eps=2e-5,momentum=0.9,fix_gamma=False,name='bn3_3')
    relu3_3 = mx.symbol.Activation(data=bn3_3, act_type="relu", name="relu3_3")
    pool3 = mx.symbol.Pooling(
        data=relu3_3, pool_type="max", kernel=(2, 2), stride=(2, 2), \
        pooling_convention="full", name="pool3")
    # group 4
    conv4_1 = mx.symbol.Convolution(
        data=pool3, kernel=(3, 3), pad=(1, 1), num_filter=512, name="conv4_1")
    bn4_1 = mx.symbol.BatchNorm(data=conv4_1,eps=2e-5,momentum=0.9,fix_gamma=False,name='bn4_1')
    relu4_1 = mx.symbol.Activation(data=bn4_1, act_type="relu", name="relu4_1")
    conv4_2 = mx.symbol.Convolution(
        data=relu4_1, kernel=(3, 3), pad=(1, 1), num_filter=512, name="conv4_2")
    bn4_2 = mx.symbol.BatchNorm(data=conv4_2,eps=2e-5,momentum=0.9,fix_gamma=False,name='bn4_2')
    relu4_2 = mx.symbol.Activation(data=bn4_2, act_type="relu", name="relu4_2")
    conv4_3 = mx.symbol.Convolution(
        data=relu4_2, kernel=(3, 3), pad=(1, 1), num_filter=512, name="conv4_3")
    bn4_3 = mx.symbol.BatchNorm(data=conv4_3,eps=2e-5,momentum=0.9,fix_gamma=False,name='bn4_3')
    relu4_3 = mx.symbol.Activation(data=bn4_3, act_type="relu", name="relu4_3")
    pool4 = mx.symbol.Pooling(
        data=relu4_3, pool_type="max", kernel=(2, 2), stride=(2, 2), name="pool4")
    # group 5
    conv5_1 = mx.symbol.Convolution(
        data=pool4, kernel=(3, 3), pad=(1, 1), num_filter=512, name="conv5_1")
    bn5_1 = mx.symbol.BatchNorm(data=conv5_1,eps=2e-5,momentum=0.9,fix_gamma=False,name='bn5_1')
    relu5_1 = mx.symbol.Activation(data=bn5_1, act_type="relu", name="relu5_1")
    conv5_2 = mx.symbol.Convolution(
        data=relu5_1, kernel=(3, 3), pad=(1, 1), num_filter=512, name="conv5_2")
    bn5_2 = mx.symbol.BatchNorm(data=conv5_2,eps=2e-5,momentum=0.9,fix_gamma=False,name='bn5_2')
    relu5_2 = mx.symbol.Activation(data=bn5_2, act_type="relu", name="relu5_2")
    conv5_3 = mx.symbol.Convolution(
        data=relu5_2, kernel=(3, 3), pad=(1, 1), num_filter=512, name="conv5_3")
    bn5_3 = mx.symbol.BatchNorm(data=conv5_3,eps=2e-5,momentum=0.9,fix_gamma=False,name='bn5_3')
    relu5_3 = mx.symbol.Activation(data=bn5_3, act_type="relu", name="relu5_3")
    pool5 = mx.symbol.Pooling(
        data=relu5_3, pool_type="max", kernel=(2, 2), stride=(2, 2),name='pool5')
    return pool5,pool4,pool3,pool2


def residual_unit(data, num_filter, stride, dim_match,bottle_neck,name):
    if bottle_neck:
        #print('bottle neck')
        bn1 = mx.sym.BatchNorm(data=data, fix_gamma=False, eps=eps,name=name + '_bn1')
        act1 = mx.sym.Activation(data=bn1, act_type='relu', name=name + '_relu1')
        conv1 = mx.sym.Convolution(data=act1, num_filter=int(num_filter * 0.25), kernel=(1, 1), stride=(1, 1), pad=(0, 0),
                                   no_bias=True, workspace=workspace, name=name + '_conv1')
        bn2 = mx.sym.BatchNorm(data=conv1, fix_gamma=False, eps=eps, name=name + '_bn2')
        act2 = mx.sym.Activation(data=bn2, act_type='relu', name=name + '_relu2')
        conv2 = mx.sym.Convolution(data=act2, num_filter=num_filter, kernel=(3, 3), stride=stride, pad=(1, 1),
                                   no_bias=True, workspace=workspace, name=name + '_conv2')
        if dim_match:
            shortcut = data
        else:
            shortcut = mx.sym.Convolution(data=act1, num_filter=num_filter, kernel=(1, 1), stride=stride, no_bias=True,
                                          workspace=workspace, name=name + '_sc')
        sum = mx.sym.ElementWiseSum(*[conv2, shortcut], name=name + '_plus')
    else:
        
        bn1 = mx.sym.BatchNorm(data=data, fix_gamma=False, eps=eps,name=name + '_bn1')
        act1 = mx.sym.Activation(data=bn1, act_type='relu', name=name + '_relu1')
        conv1 = mx.sym.Convolution(data=act1, num_filter=num_filter, kernel=(1, 1), stride=(1, 1), pad=(0, 0),
                                   no_bias=True, workspace=workspace, name=name + '_conv1')
        bn2 = mx.sym.BatchNorm(data=conv1, fix_gamma=False, eps=eps, name=name + '_bn2')
        act2 = mx.sym.Activation(data=bn2, act_type='relu', name=name + '_relu2')
        conv2 = mx.sym.Convolution(data=act2, num_filter=num_filter, kernel=(3, 3), stride=stride, pad=(1, 1),
                                   no_bias=True, workspace=workspace, name=name + '_conv2')
        bn3 = mx.sym.BatchNorm(data=conv2, fix_gamma=False, eps=eps,name=name + '_bn3')
        act3 = mx.sym.Activation(data=bn3, act_type='relu', name=name + '_relu3')
        conv3 = mx.sym.Convolution(data=act3, num_filter=num_filter, kernel=(1, 1), stride=(1, 1), pad=(0, 0), no_bias=True,
                                   workspace=workspace, name=name + '_conv3')
        if dim_match:
            shortcut = data
        else:
            shortcut = mx.sym.Convolution(data=act1, num_filter=num_filter, kernel=(1, 1), stride=stride, no_bias=True,
                                          workspace=workspace, name=name + '_sc')
        sum = mx.sym.ElementWiseSum(*[conv3, shortcut], name=name + '_plus')
    return sum
def residual_att_unit(data,express,ratio,num_filter,stride,dim_match,bottle_neck,name,deform=False):
    excitation = mx.sym.FullyConnected(data=express,num_hidden=int(ratio*num_filter),name=name+'_excitation1')
    excitation = mx.sym.Activation(data=excitation,act_type='relu',name=name+'_relu')
    excitation = mx.sym.FullyConnected(data=excitation,num_hidden=num_filter,name=name+'_excitation2')
    att = mx.sym.Activation(data=excitation,act_type='sigmoid',name=name+'_attention')
    att = mx.sym.reshape(data=att,shape=(-1,num_filter,1,1))
    if bottle_neck:
        #print('bottle neck')
        bn1 = mx.sym.BatchNorm(data=data, fix_gamma=False, eps=eps,name=name + '_bn1')
        act1 = mx.sym.Activation(data=bn1, act_type='relu', name=name + '_relu1')
        conv1 = mx.sym.Convolution(data=act1, num_filter=int(num_filter * 0.25), kernel=(1, 1), stride=(1, 1), pad=(0, 0),
                                   no_bias=True, workspace=workspace, name=name + '_conv1')
        bn2 = mx.sym.BatchNorm(data=conv1, fix_gamma=False, eps=eps, name=name + '_bn2')
        act2 = mx.sym.Activation(data=bn2, act_type='relu', name=name + '_relu2')
        conv2 = mx.sym.Convolution(data=act2, num_filter=num_filter, kernel=(3, 3), stride=stride, pad=(1, 1),
                                   no_bias=True, workspace=workspace, name=name + '_conv2')
        conv2 = mx.sym.broadcast_mul(conv2,att,name=name+'_merge')
        if dim_match:
            shortcut = data
        else:
            shortcut = mx.sym.Convolution(data=act1, num_filter=num_filter, kernel=(1, 1), stride=stride, no_bias=True,
                                          workspace=workspace, name=name + '_sc')
        sum_ = mx.sym.ElementWiseSum(*[conv2, shortcut], name=name + '_plus')
    else:
        
        bn1 = mx.sym.BatchNorm(data=data, fix_gamma=False, eps=eps,name=name + '_bn1')
        act1 = mx.sym.Activation(data=bn1, act_type='relu', name=name + '_relu1')
        conv1 = mx.sym.Convolution(data=act1, num_filter=num_filter, kernel=(1, 1), stride=(1, 1), pad=(0, 0),
                                   no_bias=True, workspace=workspace, name=name + '_conv1')
        bn2 = mx.sym.BatchNorm(data=conv1, fix_gamma=False, eps=eps, name=name + '_bn2')
        act2 = mx.sym.Activation(data=bn2, act_type='relu', name=name + '_relu2')
        conv2 = mx.sym.Convolution(data=act2, num_filter=num_filter, kernel=(3, 3), stride=stride, pad=(1, 1),
                                   no_bias=True, workspace=workspace, name=name + '_conv2')
        bn3 = mx.sym.BatchNorm(data=conv2, fix_gamma=False, eps=eps,name=name + '_bn3')
        act3 = mx.sym.Activation(data=bn3, act_type='relu', name=name + '_relu3')
        conv3 = mx.sym.Convolution(data=act3, num_filter=num_filter, kernel=(1, 1), stride=(1, 1), pad=(0, 0), no_bias=True,
                                   workspace=workspace, name=name + '_conv3')
        conv3 = mx.sym.broadcast_mul(conv3,att,name=name+'_merge')
        if dim_match:
            shortcut = data
        else:
            shortcut = mx.sym.Convolution(data=act1, num_filter=num_filter, kernel=(1, 1), stride=stride, no_bias=True,
                                          workspace=workspace, name=name + '_sc')
        sum_ = mx.sym.ElementWiseSum(*[conv3, shortcut], name=name + '_plus')
    if deform:
        offset = mx.sym.Convolution(data=sum_,num_filter=72, pad=(2, 2), kernel=(3, 3), stride=(1, 1), dilate=(2, 2), cudnn_off=True,name=name+'_offset')
        sum_ = mx.contrib.symbol.DeformableConvolution(data=sum_, offset=offset,num_filter=512, pad=(2, 2), kernel=(3, 3), num_deformable_group=4,
                                                       stride=(1, 1), dilate=(2, 2), no_bias=True,name=name+'_deform')
    return sum_

def symbol_resnet(data,units,filter_list,bottle_neck):
    

    # res1
    data_bn = mx.sym.BatchNorm(data=data, fix_gamma=True, eps=eps, use_global_stats=True, name='bn_data')
    conv0 = mx.sym.Convolution(data=data_bn, num_filter=64, kernel=(7, 7), stride=(2, 2), pad=(3, 3),
                               no_bias=True, name="conv0", workspace=workspace)
    bn0 = mx.sym.BatchNorm(data=conv0, fix_gamma=False, eps=eps,name='bn0')
    relu0 = mx.sym.Activation(data=bn0, act_type='relu', name='relu0')
    pool0 = mx.symbol.Pooling(data=relu0, kernel=(3, 3), stride=(2, 2), pad=(1, 1), pool_type='max', name='pool0')

    # res2
    unit = residual_unit(data=pool0, num_filter=filter_list[0], stride=(1, 1), dim_match=False,bottle_neck=bottle_neck,name='stage1_unit1')
    for i in range(2, units[0] + 1):
        unit = residual_unit(data=unit, num_filter=filter_list[0], stride=(1, 1), dim_match=True, bottle_neck=bottle_neck,name='stage1_unit%s' % i)
    res2 = unit
    # res3
    unit = residual_unit(data=unit, num_filter=filter_list[1], stride=(2, 2), dim_match=False,bottle_neck=bottle_neck,name='stage2_unit1')
    for i in range(2, units[1] + 1):
        unit = residual_unit(data=unit, num_filter=filter_list[1], stride=(1, 1), dim_match=True,bottle_neck=bottle_neck,name='stage2_unit%s' % i)
    res3 = unit
    # res4
    unit = residual_unit(data=unit, num_filter=filter_list[2], stride=(2, 2), dim_match=False,bottle_neck=bottle_neck,name='stage3_unit1')
    for i in range(2, units[2] + 1):
        unit = residual_unit(data=unit, num_filter=filter_list[2], stride=(1, 1), dim_match=True,bottle_neck=bottle_neck,name='stage3_unit%s' % i)
    res4 = unit
    unit = residual_unit(data=unit, num_filter=filter_list[3], stride=(2, 2), dim_match=False,bottle_neck=bottle_neck,name='stage4_unit1')
    for i in range(2, units[3] + 1):
        unit = residual_unit(data=unit, num_filter=filter_list[3], stride=(1, 1), dim_match=True,bottle_neck=bottle_neck,name='stage4_unit%s' % i)
    res5 = unit
    return res5,res4,res3,res2

def vgg_shortcut(data,units,filter_list):
    """implement later"""
    pass

def symbol_Inception_v3(data):
    
    def Conv(data, num_filter, kernel=(1, 1), stride=(1, 1), pad=(0, 0), name=None, suffix=''):
        conv = mx.sym.Convolution(data=data, num_filter=num_filter, kernel=kernel, stride=stride, pad=pad, no_bias=True, name='%s%s_conv2d' %(name, suffix))
        bn = mx.sym.BatchNorm(data=conv, name='%s%s_batchnorm' %(name, suffix), fix_gamma=True)
        act = mx.sym.Activation(data=bn, act_type='relu', name='%s%s_relu' %(name, suffix))
        return act
    def Inception7A(data,
                num_1x1,
                num_3x3_red, num_3x3_1, num_3x3_2,
                num_5x5_red, num_5x5,
                pool, proj,
                name):
        tower_1x1 = Conv(data, num_1x1, name=('%s_conv' % name))
        tower_5x5 = Conv(data, num_5x5_red, name=('%s_tower' % name), suffix='_conv')
        tower_5x5 = Conv(tower_5x5, num_5x5, kernel=(5, 5), pad=(2, 2), name=('%s_tower' % name), suffix='_conv_1')
        tower_3x3 = Conv(data, num_3x3_red, name=('%s_tower_1' % name), suffix='_conv')
        tower_3x3 = Conv(tower_3x3, num_3x3_1, kernel=(3, 3), pad=(1, 1), name=('%s_tower_1' % name), suffix='_conv_1')
        tower_3x3 = Conv(tower_3x3, num_3x3_2, kernel=(3, 3), pad=(1, 1), name=('%s_tower_1' % name), suffix='_conv_2')
        pooling = mx.sym.Pooling(data=data, kernel=(3, 3), stride=(1, 1), pad=(1, 1), pool_type=pool, name=('%s_pool_%s_pool' % (pool, name)))
        cproj = Conv(pooling, proj, name=('%s_tower_2' %  name), suffix='_conv')
        concat = mx.sym.Concat(*[tower_1x1, tower_5x5, tower_3x3, cproj], name='ch_concat_%s_chconcat' % name)
        return concat
    def Inception7B(data,
                num_3x3,
                num_d3x3_red, num_d3x3_1, num_d3x3_2,
                pool,
                name):
        tower_3x3 = Conv(data, num_3x3, kernel=(3, 3), pad=(1, 1), stride=(2, 2), name=('%s_conv' % name))
        tower_d3x3 = Conv(data, num_d3x3_red, name=('%s_tower' % name), suffix='_conv')
        tower_d3x3 = Conv(tower_d3x3, num_d3x3_1, kernel=(3, 3), pad=(1, 1), stride=(1, 1), name=('%s_tower' % name), suffix='_conv_1')
        tower_d3x3 = Conv(tower_d3x3, num_d3x3_2, kernel=(3, 3), pad=(1, 1), stride=(2, 2), name=('%s_tower' % name), suffix='_conv_2')
        pooling = mx.symbol.Pooling(data=data, kernel=(3, 3), stride=(2, 2), pad=(1,1), pool_type="max", name=('max_pool_%s_pool' % name))
        concat = mx.sym.Concat(*[tower_3x3, tower_d3x3, pooling], name='ch_concat_%s_chconcat' % name)
        return concat
    def Inception7C(data,
                num_1x1,
                num_d7_red, num_d7_1, num_d7_2,
                num_q7_red, num_q7_1, num_q7_2, num_q7_3, num_q7_4,
                pool, proj,
                name):
        tower_1x1 = Conv(data=data, num_filter=num_1x1, kernel=(1, 1), name=('%s_conv' % name))
        tower_d7 = Conv(data=data, num_filter=num_d7_red, name=('%s_tower' % name), suffix='_conv')
        tower_d7 = Conv(data=tower_d7, num_filter=num_d7_1, kernel=(1, 7), pad=(0, 3), name=('%s_tower' % name), suffix='_conv_1')
        tower_d7 = Conv(data=tower_d7, num_filter=num_d7_2, kernel=(7, 1), pad=(3, 0), name=('%s_tower' % name), suffix='_conv_2')
        tower_q7 = Conv(data=data, num_filter=num_q7_red, name=('%s_tower_1' % name), suffix='_conv')
        tower_q7 = Conv(data=tower_q7, num_filter=num_q7_1, kernel=(7, 1), pad=(3, 0), name=('%s_tower_1' % name), suffix='_conv_1')
        tower_q7 = Conv(data=tower_q7, num_filter=num_q7_2, kernel=(1, 7), pad=(0, 3), name=('%s_tower_1' % name), suffix='_conv_2')
        tower_q7 = Conv(data=tower_q7, num_filter=num_q7_3, kernel=(7, 1), pad=(3, 0), name=('%s_tower_1' % name), suffix='_conv_3')
        tower_q7 = Conv(data=tower_q7, num_filter=num_q7_4, kernel=(1, 7), pad=(0, 3), name=('%s_tower_1' % name), suffix='_conv_4')
        pooling = mx.sym.Pooling(data=data, kernel=(3, 3), stride=(1, 1), pad=(1, 1), pool_type=pool, name=('%s_pool_%s_pool' % (pool, name)))
        cproj = Conv(data=pooling, num_filter=proj, kernel=(1, 1), name=('%s_tower_2' %  name), suffix='_conv')
        # concat
        concat = mx.sym.Concat(*[tower_1x1, tower_d7, tower_q7, cproj], name='ch_concat_%s_chconcat' % name)
        return concat
    def Inception7D(data,
                num_3x3_red, num_3x3,
                num_d7_3x3_red, num_d7_1, num_d7_2, num_d7_3x3,
                pool,
                name):
        tower_3x3 = Conv(data=data, num_filter=num_3x3_red, name=('%s_tower' % name), suffix='_conv')
        tower_3x3 = Conv(data=tower_3x3, num_filter=num_3x3, kernel=(3, 3), pad=(1,1), stride=(2, 2), name=('%s_tower' % name), suffix='_conv_1')
        tower_d7_3x3 = Conv(data=data, num_filter=num_d7_3x3_red, name=('%s_tower_1' % name), suffix='_conv')
        tower_d7_3x3 = Conv(data=tower_d7_3x3, num_filter=num_d7_1, kernel=(1, 7), pad=(0, 3), name=('%s_tower_1' % name), suffix='_conv_1')
        tower_d7_3x3 = Conv(data=tower_d7_3x3, num_filter=num_d7_2, kernel=(7, 1), pad=(3, 0), name=('%s_tower_1' % name), suffix='_conv_2')
        tower_d7_3x3 = Conv(data=tower_d7_3x3, num_filter=num_d7_3x3, kernel=(3, 3), stride=(2, 2),pad=(1,1),name=('%s_tower_1' % name), suffix='_conv_3')
        pooling = mx.sym.Pooling(data=data, kernel=(3, 3), stride=(2, 2),pad=(1,1),pool_type=pool, name=('%s_pool_%s_pool' % (pool, name)))
        # concat
        concat = mx.sym.Concat(*[tower_3x3, tower_d7_3x3, pooling], name='ch_concat_%s_chconcat' % name)
        return concat
    # stage 1
    conv = Conv(data, 32, kernel=(3, 3), stride=(2, 2),pad=(1,1), name="conv")
    conv_1 = Conv(conv, 32, kernel=(3, 3), pad=(1,1),name="conv_1")
    conv_2 = Conv(conv_1, 64, kernel=(3, 3), pad=(1, 1), name="conv_2")
    
    pool = mx.sym.Pooling(data=conv_2, kernel=(3, 3), stride=(2, 2), pad=(1,1),pool_type="max", name="pool")
    # stage 2
    conv_3 = Conv(pool, 80, kernel=(1, 1), name="conv_3")
    conv_4 = Conv(conv_3, 192, kernel=(3, 3),pad=(1,1),name="conv_4")
    pool1 = mx.sym.Pooling(data=conv_4, kernel=(3, 3), stride=(2, 2), pad=(1,1),pool_type="max", name="pool1")
    
    
    in3a = Inception7A(pool1, 64,
                       64, 96, 96,
                       48, 64,
                       "avg", 32, "mixed")
    in3b = Inception7A(in3a, 64,
                       64, 96, 96,
                       48, 64,
                       "avg", 64, "mixed_1")
    in3c = Inception7A(in3b, 64,
                       64, 96, 96,
                       48, 64,
                       "avg", 64, "mixed_2")
    in3d = Inception7B(in3c, 384,
                       64, 96, 96,
                       "max", "mixed_3")
    # stage 4
    in4a = Inception7C(in3d, 192,
                       128, 128, 192,
                       128, 128, 128, 128, 192,
                       "avg", 192, "mixed_4")
    in4b = Inception7C(in4a, 192,
                       160, 160, 192,
                       160, 160, 160, 160, 192,
                       "avg", 192, "mixed_5")
    in4c = Inception7C(in4b, 192,
                       160, 160, 192,
                       160, 160, 160, 160, 192,
                       "avg", 192, "mixed_6")
    in4d = Inception7C(in4c, 192,
                       192, 192, 192,
                       192, 192, 192, 192, 192,
                       "avg", 192, "mixed_7")
    in4e = Inception7D(in4d, 192, 320,
                       192, 192, 192, 192,
                       "max", "mixed_8")
    #print(in4e.infer_shape(data=(1,3,640,640))[1])
    return in4e,in3d,pool1,pool

def get_symbol_train(seq_len):
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
    #print('xx')
    network = cfg.NETWORK
    num_classes = cfg.NUM_CLASSES
    data = mx.symbol.Variable(name="data")
    expression = mx.symbol.Variable(name='expression')
    label = mx.symbol.Variable(name="label")
    if network=='vgg16':
        c5,c4,c3,_ = symbol_vgg(data)
    elif network=='vgg16_bn':
        c5,c4,c3,_ = symbol_vgg_bn(data)
    elif network.startswith('resnet'):
        #yi fan hou xi
        num_layers = int(network.split('_')[-1])
        if num_layers >= 50:
            filter_list = [64, 256, 512, 1024, 2048]
            bottle_neck = True
        else:
            filter_list = [64, 64, 128, 256, 512]
            bottle_neck = False
        #num_stages = 4
        if num_layers == 18:
            units = [2, 2, 2, 2]
        elif num_layers == 34:
            units = [3, 4, 6, 3]
        elif num_layers == 50:
            units = [3, 4, 6, 3]
        elif num_layers == 101:
            units = [3, 4, 23, 3]
        elif num_layers == 152:
            units = [3, 8, 36, 3]
        elif num_layers == 200:
            units = [3, 24, 36, 3]
        elif num_layers == 269:
            units = [3, 30, 48, 8]
        else:
            raise ValueError("no experiments done on num_layers {}, you can do it yourself".format(num_layers))
        c5,c4,c3,_=symbol_resnet(data,units,filter_list,bottle_neck)
    elif network=='inceptionv3':
        c5,c4,c3,_=symbol_Inception_v3(data)
    rnn_feat = get_rnn_feat(seq_len,expression)
    c5 = residual_att_unit(data=c5,express=rnn_feat,ratio=0.75,num_filter=512,stride=(1,1),bottle_neck=False,dim_match=False,name='c5',deform=False)
    c4 = residual_att_unit(data=c4,express=rnn_feat,ratio=0.5,num_filter=256,stride=(1,1),bottle_neck=False,dim_match=False,name='c4',deform=False)
    c3 = residual_att_unit(data=c3,express=rnn_feat,ratio=0.25,num_filter=128,stride=(1,1),bottle_neck=False,dim_match=False,name='c3',deform=False)
    P6 = mx.symbol.Convolution(data=c5,num_filter=256,kernel=(3,3),stride=(2,2),pad=(1,1),name='P6')
    p6_relu = mx.symbol.Activation(data=P6,act_type='relu',name='p6_relu')
    P7 = mx.symbol.Convolution(data=p6_relu,num_filter=256,kernel=(3,3),stride=(2,2),pad=(1,1),name='P7')
    P5 = mx.symbol.Convolution(data=c5,num_filter=256,kernel=(1,1),stride=(1,1),pad=(0,0),name='P5')
    
    P5_topdown = mx.symbol.Deconvolution(data=P5,num_filter=256,kernel=(4,4),stride=(2,2),pad=(1,1),name='P5_topdown')
    P4_lateral = mx.symbol.Convolution(data=c4,num_filter=256,kernel=(1,1),stride=(1,1),pad=(0,0),name='P4_lateral')
    P4 = mx.sym.elemwise_add(P4_lateral,P5_topdown,name='P4')
    
    P4_topdown = mx.symbol.Deconvolution(data=P4,num_filter=256,kernel=(4,4),stride=(2,2),pad=(1,1),name='P4_topdown')
    P3_lateral = mx.symbol.Convolution(data=c3,num_filter=256,kernel=(1,1),stride=(1,1),pad=(0,0),name='P3_lateral')
    P3 = mx.sym.elemwise_add(P3_lateral, P4_topdown,name='P3')
    #specific parameters
    from_layers = [P7,P6,P5,P4,P3]
    sizes = [[0.01, .1], [.2,.3], [.4, .5], [.6, .7],[.9,1.]]
    ratios = [[1,], [1,], [1,], [1,],[1,]]
    normalizations = [20, -1, -1, -1,-1]
    steps = [ x / 640.0 for x in [4, 8, 16, 32,32]]
    num_channels = [256]

    loc_preds, cls_preds, anchor_boxes = multibox_layer(from_layers,\
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
        name="detection", nms_threshold=cfg.NMS_THRESHOLD, force_suppress=cfg.FORCE_SUPPRESS,
        variances=(0.1, 0.1, 0.2, 0.2), nms_topk=cfg.NMS_TOPK)
    det = mx.symbol.MakeLoss(data=det, grad_scale=0, name="det_out")

    # group output
    out = mx.symbol.Group([cls_prob, loc_loss, cls_label, det])
    return out,('expression','data',),('label',)

def get_symbol(network,num_classes=20, nms_thresh=0.5, force_suppress=False, nms_topk=400):
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
    net = get_symbol_train(network,num_classes)
    cls_preds = net.get_internals()["multibox_cls_pred_output"]
    loc_preds = net.get_internals()["multibox_loc_pred_output"]
    anchor_boxes = net.get_internals()["multibox_anchors_output"]

    cls_prob = mx.symbol.SoftmaxActivation(data=cls_preds, mode='channel', \
        name='cls_prob')
    out = mx.contrib.symbol.MultiBoxDetection(*[cls_prob, loc_preds, anchor_boxes], \
        name="detection", nms_threshold=nms_thresh, force_suppress=force_suppress,
        variances=(0.1, 0.1, 0.2, 0.2), nms_topk=nms_topk)
    return out
    
if __name__=='__main__':
    symbol,data_name,label_name = get_symbol_train(10)
    print(symbol.list_arguments())
    
    
