import os
import sys
import config

sys.path.insert(0, config.mxnet_path)
import mxnet as mx


def imagenet_iterator(data_dir, batch_size, kv):
    train = mx.io.ImageRecordIter(
            path_imgrec         = os.path.join(data_dir, "train.rec"),
            label_width         = 1,
            data_name           = 'data',
            label_name          = 'softmax_label',
            data_shape          = (3, 224, 224),
            batch_size          = batch_size,
            pad                 = 0,
            fill_value          = 127,
            facebook_aug        = True,
            max_random_area     = 1.0,
            min_random_area     = 0.08,
            max_aspect_ratio    = 4.0 / 3.0,
            min_aspect_ratio    = 3.0 / 4.0,
            brightness          = 0.4,
            contrast            = 0.4,
            saturation          = 0.4,
            pca_noise           = 0.1,
            scale               = 1,
            inter_method        = 2,
            rand_mirror         = True,
            shuffle             = True,
            shuffle_chunk_size  = 4096,
            preprocess_threads  = 24,
            prefetch_buffer     = 16,
            num_parts           = kv.num_workers,
            part_index          = kv.rank)

    val = mx.io.ImageRecordIter(
            path_imgrec         = os.path.join(data_dir, "val.rec"),
            label_width         = 1,
            data_name           = 'data',
            label_name          = 'softmax_label',
            resize              = 256,
            batch_size          = batch_size,
            data_shape          = (3, 224, 224),
            scale               = 1,
            inter_method        = 2,
            rand_crop           = False,
            rand_mirror         = False,
            num_parts           = kv.num_workers,
            part_index          = kv.rank)

    num_examples = 1281167
    return train, val, num_examples
