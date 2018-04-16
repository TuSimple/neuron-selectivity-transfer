import mxnet as mx
import numpy as np


class KTAccMetric(mx.metric.EvalMetric):
    def __init__(self):
        super(KTAccMetric, self).__init__('accuracy')

    def update(self, labels, preds):
        pred_cf = preds[0]
        label_cf = labels[0]

        pred_label = mx.ndarray.argmax_channel(pred_cf).asnumpy().astype('int32')
        label = label_cf.asnumpy().astype('int32')

        self.sum_metric += (pred_label.flat == label.flat).sum()
        self.num_inst += len(pred_label.flat)


class KTTopkAccMetric(mx.metric.EvalMetric):
    """Calculate top k predictions accuracy"""

    def __init__(self, **kwargs):
        super(KTTopkAccMetric, self).__init__('top_k_accuracy')
        try:
            self.top_k = kwargs['top_k']
        except KeyError:
            self.top_k = 1
        assert (self.top_k > 1), 'Please use Accuracy if top_k is no more than 1'
        self.name += '_%d' % self.top_k

    def update(self, labels, preds):

        pred_label = preds[0]
        label = labels[0]
        assert (len(pred_label.shape) <= 2), 'Predictions should be no more than 2 dims'
        pred_label = np.argsort(pred_label.asnumpy().astype('float32'), axis=1)
        label = label.asnumpy().astype('int32')
        num_samples = pred_label.shape[0]
        num_dims = len(pred_label.shape)
        if num_dims == 1:
            self.sum_metric += (pred_label.flat == label.flat).sum()
        elif num_dims == 2:
            num_classes = pred_label.shape[1]
            top_k = min(num_classes, self.top_k)
            for j in range(top_k):
                self.sum_metric += (pred_label[:, num_classes - 1 - j].flat == label.flat).sum()
        self.num_inst += num_samples
