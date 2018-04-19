# mxnet version: https://github.com/huangzehao/incubator-mxnet-bk
mxnet_path = './incubator-mxnet-bk/python/'
gpu_list = [0, 1, 2, 3, 4, 5, 6, 7]
dataset = "imagenet"
model_prefix = "inception_bn_64_8_poly_0_50"
network = "inception_bn"
depth = 50
model_load_prefix = model_prefix
model_load_epoch = 0
retrain = False

# data
data_dir = 'imagenet'
batch_size = 64
batch_size *= len(gpu_list)
kv_store = 'device'

# optimizer
lr = 0.1
wd = 0.0001
momentum = 0.9
lr_step = [30, 60, 90]
lr_factor = 0.1
begin_epoch = model_load_epoch if retrain else 0
num_epoch = 100
frequent = 50

# network config
num_classes = 1000

# knowledge transfer
kt = True
kt_type = 'mmd' # 'kd' or 'kd+mmd'
teacher_prefix = 'teacher_model/resnet-101'
teacher_epoch = 0
teacher_symbol = 'fc1_output' if kt_type == 'kd' else 'bn1_output'
kt_label_names = ('teacher_label',)
kt_label_shapes = [('teacher_label', (batch_size, num_classes))] if kt_type == 'kd' else \
                  [('teacher_label', (batch_size, 2048, 7, 7))]
mmd_kernel = 'poly'
kt_weight = 16 if kt_type == 'kd' else 50
kd_t = 4

if len(kt_type.split('+')) > 1:
    teacher_symbol = ['fc1_output', 'bn1_output']
    kt_label_names = ('teacher_label_1', 'teacher_label_2',)
    kt_label_shapes = [('teacher_label_1', (batch_size, num_classes)), ('teacher_label_2', (batch_size, 2048, 7, 7))]
    kt_weight = [16, 50]
