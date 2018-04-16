"""

Inception + BN, suitable for images with around 224 x 224

Reference:

Sergey Ioffe and Christian Szegedy. Batch normalization: Accelerating deep
network training by reducing internal covariate shift. arXiv preprint
arXiv:1502.03167, 2015.

"""

import mxnet as mx
from transfer import *


eps = 2e-5 # in order to match tornadomeet's teacher model
bn_mom = 0.9
fix_gamma = False


def ConvFactory(data, num_filter, kernel, stride=(1, 1), pad=(0, 0), name=None, suffix=''):
    conv = mx.symbol.Convolution(data=data, num_filter=num_filter, kernel=kernel, stride=stride, pad=pad,
                                 name='conv_%s%s' % (name, suffix))
    bn = mx.symbol.BatchNorm(data=conv, fix_gamma=fix_gamma, eps=eps, momentum=bn_mom, name='bn_%s%s' % (name, suffix))
    act = mx.symbol.Activation(data=bn, act_type='relu', name='relu_%s%s' % (name, suffix))
    return act

def ConvFactory2(data, num_filter, kernel, stride=(1, 1), pad=(0, 0), name=None, suffix=''):
    conv = mx.symbol.Convolution(data=data, num_filter=num_filter, kernel=kernel, stride=stride, pad=pad,
                                 name='conv_%s%s' % (name, suffix))
    bn = mx.symbol.BatchNorm(data=conv, fix_gamma=fix_gamma, eps=eps, momentum=bn_mom, name='bn_%s%s' % (name, suffix))
    act = mx.symbol.Activation(data=bn, act_type='relu', name='relu_%s%s' % (name, suffix))
    return act, bn

def InceptionFactoryA(data, num_1x1, num_3x3red, num_3x3, num_d3x3red, num_d3x3, pool, proj, name):
    # 1x1
    c1x1 = ConvFactory(data=data, num_filter=num_1x1, kernel=(1, 1), name=('%s_1x1' % name))
    # 3x3 reduce + 3x3
    c3x3r = ConvFactory(data=data, num_filter=num_3x3red, kernel=(1, 1), name=('%s_3x3' % name), suffix='_reduce')
    c3x3 = ConvFactory(data=c3x3r, num_filter=num_3x3, kernel=(3, 3), pad=(1, 1), name=('%s_3x3' % name))
    # double 3x3 reduce + double 3x3
    cd3x3r = ConvFactory(data=data, num_filter=num_d3x3red, kernel=(1, 1), name=('%s_double_3x3' % name),
                         suffix='_reduce')
    cd3x3 = ConvFactory(data=cd3x3r, num_filter=num_d3x3, kernel=(3, 3), pad=(1, 1), name=('%s_double_3x3_0' % name))
    cd3x3 = ConvFactory(data=cd3x3, num_filter=num_d3x3, kernel=(3, 3), pad=(1, 1), name=('%s_double_3x3_1' % name))
    # pool + proj
    pooling = mx.symbol.Pooling(data=data, kernel=(3, 3), stride=(1, 1), pad=(1, 1), pool_type=pool,
                                name=('%s_pool_%s_pool' % (pool, name)))
    cproj = ConvFactory(data=pooling, num_filter=proj, kernel=(1, 1), name=('%s_proj' % name))
    # concat
    concat = mx.symbol.Concat(*[c1x1, c3x3, cd3x3, cproj], name='ch_concat_%s_chconcat' % name)
    return concat

def InceptionFactoryA2(data, num_1x1, num_3x3red, num_3x3, num_d3x3red, num_d3x3, pool, proj, name):
    # 1x1
    c1x1, c1x1_bn = ConvFactory2(data=data, num_filter=num_1x1, kernel=(1, 1), name=('%s_1x1' % name))
    # 3x3 reduce + 3x3
    c3x3r = ConvFactory(data=data, num_filter=num_3x3red, kernel=(1, 1), name=('%s_3x3' % name), suffix='_reduce')
    c3x3, c3x3_bn = ConvFactory2(data=c3x3r, num_filter=num_3x3, kernel=(3, 3), pad=(1, 1), name=('%s_3x3' % name))
    # double 3x3 reduce + double 3x3
    cd3x3r = ConvFactory(data=data, num_filter=num_d3x3red, kernel=(1, 1), name=('%s_double_3x3' % name),
                         suffix='_reduce')
    cd3x3 = ConvFactory(data=cd3x3r, num_filter=num_d3x3, kernel=(3, 3), pad=(1, 1), name=('%s_double_3x3_0' % name))
    cd3x3, cd3x3_bn = ConvFactory2(data=cd3x3, num_filter=num_d3x3, kernel=(3, 3), pad=(1, 1),
                                     name=('%s_double_3x3_1' % name))
    # pool + proj
    pooling = mx.symbol.Pooling(data=data, kernel=(3, 3), stride=(1, 1), pad=(1, 1), pool_type=pool,
                                name=('%s_pool_%s_pool' % (pool, name)))
    cproj, cproj_bn = ConvFactory2(data=pooling, num_filter=proj, kernel=(1, 1), name=('%s_proj' % name))
    # concat
    concat = mx.symbol.Concat(*[c1x1, c3x3, cd3x3, cproj], name='ch_concat_%s_chconcat' % name)
    concat_bn = mx.symbol.Concat(*[c1x1_bn, c3x3_bn, cd3x3_bn, cproj_bn],
                                   name='ch_concat_%s_chconcat_bn' % name)
    return concat, concat_bn


def InceptionFactoryB(data, num_3x3red, num_3x3, num_d3x3red, num_d3x3, name):
    # 3x3 reduce + 3x3
    c3x3r = ConvFactory(data=data, num_filter=num_3x3red, kernel=(1, 1), name=('%s_3x3' % name), suffix='_reduce')
    c3x3 = ConvFactory(data=c3x3r, num_filter=num_3x3, kernel=(3, 3), pad=(1, 1), stride=(2, 2), name=('%s_3x3' % name))
    # double 3x3 reduce + double 3x3
    cd3x3r = ConvFactory(data=data, num_filter=num_d3x3red, kernel=(1, 1), name=('%s_double_3x3' % name),
                         suffix='_reduce')
    cd3x3 = ConvFactory(data=cd3x3r, num_filter=num_d3x3, kernel=(3, 3), pad=(1, 1), stride=(1, 1),
                        name=('%s_double_3x3_0' % name))
    cd3x3 = ConvFactory(data=cd3x3, num_filter=num_d3x3, kernel=(3, 3), pad=(1, 1), stride=(2, 2),
                        name=('%s_double_3x3_1' % name))
    # pool + proj
    pooling = mx.symbol.Pooling(data=data, kernel=(3, 3), stride=(2, 2), pad=(1, 1), pool_type="max",
                                name=('max_pool_%s_pool' % name))
    # concat
    concat = mx.symbol.Concat(*[c3x3, cd3x3, pooling], name='ch_concat_%s_chconcat' % name)
    return concat


def inception_bn(num_classes=1000, config=None):
    # data
    data = mx.symbol.Variable(name="data")
    # match tornadomeet's teacher model
    data = mx.sym.BatchNorm(data=data, fix_gamma=True, eps=2e-5, momentum=bn_mom, name='bn_data')
    # stage 1
    conv1 = ConvFactory(data=data, num_filter=64, kernel=(7, 7), stride=(2, 2), pad=(3, 3), name='1')
    pool1 = mx.symbol.Pooling(data=conv1, kernel=(3, 3), stride=(2, 2), name='pool_1', pool_type='max')
    # stage 2
    conv2red = ConvFactory(data=pool1, num_filter=64, kernel=(1, 1), stride=(1, 1), name='2_red')
    conv2 = ConvFactory(data=conv2red, num_filter=192, kernel=(3, 3), stride=(1, 1), pad=(1, 1), name='2')
    pool2 = mx.symbol.Pooling(data=conv2, kernel=(3, 3), stride=(2, 2), name='pool_2', pool_type='max')
    # stage 2
    in3a = InceptionFactoryA(pool2, 64, 64, 64, 64, 96, "avg", 32, '3a')
    in3b = InceptionFactoryA(in3a, 64, 64, 96, 64, 96, "avg", 64, '3b')
    in3c = InceptionFactoryB(in3b, 128, 160, 64, 96, '3c')
    # stage 3
    in4a = InceptionFactoryA(in3c, 224, 64, 96, 96, 128, "avg", 128, '4a')
    in4b = InceptionFactoryA(in4a, 192, 96, 128, 96, 128, "avg", 128, '4b')
    in4c = InceptionFactoryA(in4b, 160, 128, 160, 128, 160, "avg", 128, '4c')
    in4d = InceptionFactoryA(in4c, 96, 128, 192, 160, 192, "avg", 128, '4d')
    in4e = InceptionFactoryB(in4d, 128, 192, 192, 256, '4e')
    # stage 4
    in5a = InceptionFactoryA(in4e, 352, 192, 320, 160, 224, "avg", 128, '5a')
    in5b, in5b_bn = InceptionFactoryA2(in5a, 352, 192, 320, 192, 224, "max", 128, '5b')
    # global avg pooling
    avg = mx.symbol.Pooling(data=in5b, kernel=(7, 7), stride=(1, 1), name="global_pool", pool_type='avg')
    # linear classifier
    flatten = mx.symbol.Flatten(data=avg, name='flatten')
    fc1 = mx.symbol.FullyConnected(data=flatten, num_hidden=num_classes, name='fc1')
    softmax = mx.symbol.SoftmaxOutput(data=fc1, name='softmax')

    if config.kt:
        if config.kt_type == 'kd':
            teacher_label = mx.symbol.Variable('teacher_label')
            kd_loss = kd(fc1, teacher_label, config.kd_t, config.kt_weight, "")
            return mx.symbol.Group([softmax, kd_loss])
        if config.kt_type == 'mmd' and config.mmd_kernel == 'poly':
            teacher_label = mx.symbol.Variable('teacher_label')
            mmd_loss = mmd_poly(in5b_bn, teacher_label, 0.0, config.kt_weight, "")
            return mx.symbol.Group([softmax, mmd_loss])
        if config.kt_type == 'kd+mmd' and config.mmd_kernel == 'poly':
            teacher_label_1 = mx.symbol.Variable('teacher_label_1')
            teacher_label_2 = mx.symbol.Variable('teacher_label_2')
            kd_loss = kd(fc1, teacher_label_1, config.kd_t, config.kt_weight[0], "")
            mmd_loss = mmd_poly(in5b_bn, teacher_label_2, 0.0, config.kt_weight[1], "")
            return mx.symbol.Group([softmax, kd_loss, mmd_loss])
    return softmax
