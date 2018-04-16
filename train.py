import logging, os
import sys

import config

sys.path.insert(0, config.mxnet_path)
import mxnet as mx
from core.scheduler import multi_factor_scheduler
from core.solver import Solver
from core.metric import *
from data import *
from symbol import *


def main(config):
    # log file
    log_dir = "./log"
    if not os.path.exists(log_dir):
        os.mkdir(log_dir)
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s %(name)s %(levelname)s %(message)s',
                        datefmt='%m-%d %H:%M',
                        filename='{}/{}.log'.format(log_dir, config.model_prefix),
                        filemode='a')
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter('%(name)s %(levelname)s %(message)s')
    console.setFormatter(formatter)
    logging.getLogger('').addHandler(console)
    # model folder
    model_dir = "./model"
    if not os.path.exists(model_dir):
        os.mkdir(model_dir)

    # set up environment
    devs = [mx.gpu(int(i)) for i in config.gpu_list]
    kv = mx.kvstore.create(config.kv_store)

    # set up iterator and symbol
    # iterator
    train, val, num_examples = imagenet_iterator(data_dir=config.data_dir,
                                                 batch_size=config.batch_size,
                                                 kv=kv)
    data_names = ('data',)
    label_names = ('softmax_label',)
    data_shapes = [('data', (config.batch_size, 3, 224, 224))]
    label_shapes = [('softmax_label', (config.batch_size,))]

    symbol = eval(config.network)(num_classes=config.num_classes, config=config)

    # train
    epoch_size = max(int(num_examples / config.batch_size / kv.num_workers), 1)
    if config.lr_step is not None:
        lr_scheduler = multi_factor_scheduler(config.begin_epoch, epoch_size, step=config.lr_step,
                                              factor=config.lr_factor)
    else:
        lr_scheduler = None

    optimizer_params = {'learning_rate': config.lr,
                        'lr_scheduler': lr_scheduler,
                        'wd': config.wd,
                        'momentum': config.momentum}
    optimizer = "nag"
    eval_metric = ['acc']
    if config.dataset == "imagenet":
        eval_metric.append(mx.metric.create('top_k_accuracy', top_k=5))

    # knowledge transfer
    teacher_module = None
    if config.kt:
        eval_metric=[KTAccMetric()]
        if config.dataset == 'imagenet':
            eval_metric.append(KTTopkAccMetric(top_k=5))
        if len(config.kt_type.split('+')) > 1:
            logging.info('knowledge transfer training by {} with weight {}'.format(config.kt_type.split('+')[0], config.kt_weight[0]))
            logging.info('knowledge transfer training by {} with weight {}'.format(config.kt_type.split('+')[1], config.kt_weight[1]))
        else:
            logging.info('knowledge transfer training by {} with weight {}'.format(config.kt_type, config.kt_weight))
        label_names += config.kt_label_names
        label_shapes += config.kt_label_shapes
        logging.info('loading teacher model from {}-{:04d}'.format(config.teacher_prefix, config.teacher_epoch))
        teacher_symbol, teacher_arg_params, teacher_aux_params = mx.model.load_checkpoint(config.teacher_prefix, config.teacher_epoch)
        if len(config.kt_type.split('+')) > 1:
            teacher_symbol = mx.symbol.Group([teacher_symbol.get_internals()[config.teacher_symbol[0]],
                                              teacher_symbol.get_internals()[config.teacher_symbol[1]]])
        else:
            teacher_symbol = teacher_symbol.get_internals()[config.teacher_symbol]
        teacher_module = mx.module.Module(teacher_symbol, context=devs)
        teacher_module.bind(data_shapes=data_shapes, for_training=False, grad_req='null')
        teacher_module.set_params(teacher_arg_params, teacher_aux_params)

    solver = Solver(symbol=symbol,
                    data_names=data_names,
                    label_names=label_names,
                    data_shapes=data_shapes,
                    label_shapes=label_shapes,
                    logger=logging,
                    context=devs)
    epoch_end_callback = mx.callback.do_checkpoint("./model/" + config.model_prefix)
    batch_end_callback = mx.callback.Speedometer(config.batch_size, config.frequent)
    arg_params = None
    aux_params = None
    if config.retrain:
        logging.info('retrain from {}-{:04d}'.format(config.model_load_prefix, config.model_load_epoch))
        _, arg_params, aux_params = mx.model.load_checkpoint("model/{}".format(config.model_load_prefix),
                                                            config.model_load_epoch)
    initializer = mx.init.Xavier(rnd_type='gaussian', factor_type='in', magnitude=2)

    solver.fit(train_data=train,
               eval_data=val,
               eval_metric=eval_metric,
               epoch_end_callback=epoch_end_callback,
               batch_end_callback=batch_end_callback,
               initializer=initializer,
               arg_params=arg_params,
               aux_params=aux_params,
               optimizer=optimizer,
               optimizer_params=optimizer_params,
               begin_epoch=config.begin_epoch,
               num_epoch=config.num_epoch,
               kvstore=kv,
               teacher_modules=teacher_module)


if __name__ == '__main__':
    main(config)
