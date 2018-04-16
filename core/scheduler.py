import sys
import config

sys.path.insert(0, config.mxnet_path)
import mxnet as mx


def multi_factor_scheduler(begin_epoch, epoch_size, step, factor=0.1):
    step_ = [epoch_size * (x - begin_epoch) for x in step if x - begin_epoch > 0]
    return mx.lr_scheduler.MultiFactorScheduler(step=step_, factor=factor) if len(step_) else None
