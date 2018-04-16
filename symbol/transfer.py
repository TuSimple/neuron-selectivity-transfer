import mxnet as mx


def kd(student_hard_logits, teacher_hard_logits, temperature, weight_lambda, prefix):
    student_soft_logits = student_hard_logits / temperature
    teacher_soft_logits = teacher_hard_logits / temperature
    teacher_soft_labels = mx.symbol.SoftmaxActivation(teacher_soft_logits, name="teacher%s_soft_labels" % prefix)
    kd_loss = mx.symbol.SoftmaxOutput(data=student_soft_logits, label=teacher_soft_labels,
                                      grad_scale=weight_lambda, name="%skd_loss" % prefix)
    return kd_loss


def mmd_poly(student_fm, teacher_fm, c, weight_lambda, prefix):
    student_mmd_l2norm = mx.symbol.L2Normalization(student_fm, eps=1e-10, mode='spatial',
                                                   name="student%s_mmd_l2norm" % prefix)
    student_mmd_vector = mx.symbol.Reshape(student_mmd_l2norm, shape=(0, 0, -1), name="student%s_mmd_vector" % prefix)
    student_mmd_vector_T = mx.symbol.transpose(student_mmd_vector, axes=(0, 2, 1),
                                               name="student%s_mmd_vector_T" % prefix)
    student_mmd_matrix = mx.symbol.batch_dot(lhs=student_mmd_vector, rhs=student_mmd_vector_T,
                                             name="student%s_mmd_matrix" % prefix)
    student_mmd_matrix = student_mmd_matrix + c
    student_mmd_square = mx.symbol.square(student_mmd_matrix, name="student%s_mmd_square" % prefix)
    student_mmd_mean = mx.symbol.mean(student_mmd_square, axis=(1, 2), name="student%s_mmd_mean" % prefix)

    teacher_mmd_l2norm = mx.symbol.L2Normalization(teacher_fm, eps=1e-10, mode='spatial',
                                                   name="teacher%s_mmd_l2norm" % prefix)
    teacher_mmd_vector = mx.symbol.Reshape(teacher_mmd_l2norm, shape=(0, 0, -1), name="teacher%s_mmd_vector" % prefix)
    teacher_mmd_vector_T = mx.symbol.transpose(teacher_mmd_vector, axes=(0, 2, 1),
                                               name="teacher%s_mmd_vector_T" % prefix)

    ts_mmd_matrix = mx.symbol.batch_dot(lhs=student_mmd_vector, rhs=teacher_mmd_vector_T,
                                        name="ts%s_mmd_matrix" % prefix)
    ts_mmd_matrix = ts_mmd_matrix + c
    ts_mmd_square = mx.symbol.square(ts_mmd_matrix, name="ts%s_mmd_square" % prefix)
    ts_mmd_mean = mx.symbol.mean(ts_mmd_square, axis=(1, 2), name="ts%s_mmd_mean" % prefix)

    mmd_diff = student_mmd_mean - 2 * ts_mmd_mean
    mmd_loss = mx.symbol.MakeLoss(mmd_diff, grad_scale=weight_lambda, name="mmd%s_loss" % prefix)
    return mmd_loss
