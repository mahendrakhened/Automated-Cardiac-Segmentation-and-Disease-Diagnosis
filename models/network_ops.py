import tensorflow as tf

def AdamOptimizer(lr):
    return tf.train.AdamOptimizer(learning_rate = lr)

def BN_ELU_Conv(inputs, n_filters, filter_size=3,
                dropout_rate=0.2, is_training=tf.constant(False,dtype=tf.bool)):
    initializer = tf.contrib.layers.variance_scaling_initializer(factor=2.0, 
                mode='FAN_IN', uniform=False, seed=None, dtype=tf.float32)
    regularizer = tf.contrib.layers.l2_regularizer(scale=1.0)

    l = tf.contrib.layers.batch_norm(inputs, activation_fn=tf.nn.elu, 
                        is_training=is_training)                          
    l = tf.layers.conv2d(inputs=l,
                        filters=n_filters,
                        kernel_size=[filter_size, filter_size],
                        padding="same",
                        activation=None,
                        kernel_initializer=initializer,
                        bias_initializer=tf.zeros_initializer(),
                        kernel_regularizer=regularizer
                        )                          
    l = tf.layers.dropout(l, rate=(dropout_rate), training=is_training)                          
    return l

def TransitionDown(inputs, n_filters, dropout_rate=0.2, 
                    is_training=tf.constant(False,dtype=tf.bool)):
    l = BN_ELU_Conv(inputs, n_filters, filter_size=1, dropout_rate=dropout_rate,
                     is_training=is_training)
    l = tf.layers.max_pooling2d(l, pool_size=[2, 2], strides=2)
    return l

def ResidualTransitionUp(skip_connection, block_to_upsample, n_filters_keep,
                     is_training=tf.constant(False,dtype=tf.bool)):
    """
    Performs upsampling on block_to_upsample by a factor 2 and adds it with the skip_connection 
    """
    initializer = tf.contrib.layers.variance_scaling_initializer(factor=2.0, 
                mode='FAN_IN', uniform=False, seed=None, dtype=tf.float32)
    regularizer = tf.contrib.layers.l2_regularizer(scale=1.0)
    l = tf.layers.conv2d_transpose(block_to_upsample, filters=n_filters_keep,
                                     kernel_size = (3,3), 
                                     strides = (2,2), 
                                     padding = 'SAME',
                                    kernel_initializer=initializer,
                                    bias_initializer=tf.zeros_initializer(),
                                    kernel_regularizer=regularizer
                                     ) 

    skip_connection_shape = tf.shape(skip_connection)
    l = tf.image.resize_image_with_crop_or_pad(l, skip_connection_shape[1], skip_connection_shape[2])
    l = tf.add(l, skip_connection)
    return l


def SpatialWeightedCrossEntropy(logits, targets, weight_map):
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels = targets,logits = logits)
    weighted_cross_entropy = tf.multiply(cross_entropy, weight_map)
    mean_weighted_cross_entropy = tf.reduce_mean(weighted_cross_entropy)
    return mean_weighted_cross_entropy

def dice_multiclass(output, target, loss_type='sorensen', axis=[0,1,2], smooth=1e-5):
    """Soft dice (Sorensen or Jaccard) coefficient for comparing the similarity
    of two batch of data, usually be used for binary image segmentation
    i.e. labels are binary. The coefficient between 0 to 1, 1 means totally match.

    Parameters
    -----------
    output : tensor
        A distribution with shape: [batch_size, ....], (any dimensions).
    target : tensor
        A distribution with shape: [batch_size, ....], (any dimensions).
    loss_type : string
        ''jaccard'' or ''sorensen'', default is ''jaccard''.
    axis : list of integer
        All dimensions are reduced, default ''[1,2,3]''.
    smooth : float
        This small value will be added to the numerator and denominator.
        If both output and target are empty, it makes sure dice is 1.
        If either output or target are empty (all pixels are background), dice = '''smooth/(small_value + smooth)'',
        then if smooth is very small, dice close to 0 (even the image values lower than the threshold),
        so in this case, higher smooth can have a higher dice.

    Examples
    ---------
    >>> outputs = tl.act.pixel_wise_softmax(network.outputs)
    >>> dice_loss = 1 - tl.cost.dice_coe(outputs, y_)

    References
    -----------
    """
    inse = tf.reduce_sum(output * target, axis=axis)
    if loss_type == 'jaccard':
        l = tf.reduce_sum(output * output, axis=axis)
        r = tf.reduce_sum(target * target, axis=axis)
    elif loss_type == 'sorensen':
        l = tf.reduce_sum(output, axis=axis)
        r = tf.reduce_sum(target, axis=axis)
    else:
        raise Exception("Unknow loss_type")

    dice = (2. * inse + smooth) / (l + r + smooth)
    ##Attention: Return dice/jaccard score of all the classes in the batch if axis=0
    # dice = tf.reduce_mean(dice, axis=0)
    return dice

def dice_hard_coe(output, target, threshold=0.5, axis=(1, 2, 3), smooth=1e-5):
    """Non-differentiable Dice coefficient for comparing the similarity
    of two batch of data, usually be used for binary image segmentation i.e. labels are binary.
    The coefficient between 0 to 1, 1 if totally match.

    Parameters
    -----------
    output : tensor
        A distribution with shape: [batch_size, ....], (any dimensions).
    target : tensor
        The target distribution, format the same with `output`.
    threshold : float
        The threshold value to be true.
    axis : tuple of integer
        All dimensions are reduced, default ``(1,2,3)``.
    smooth : float
        This small value will be added to the numerator and denominator, see ``dice_coe``.

    References
    -----------
    Source : http://tensorlayer.readthedocs.io/en/latest/_modules/tensorlayer/cost.html#dice_coe

    """

    last_dim_idx = target.get_shape().ndims - 1
    num_class = tf.shape(target)[last_dim_idx]
    output =  tf.cast(output, tf.uint8)
    output = tf.one_hot(output, num_class)
    output = tf.cast(output > threshold, dtype=tf.float32)
    target = tf.cast(target > threshold, dtype=tf.float32)
    inse = tf.reduce_sum(tf.multiply(output, target), axis=axis)
    l = tf.reduce_sum(output, axis=axis)
    r = tf.reduce_sum(target, axis=axis)
    hard_dice = (2. * inse + smooth) / (l + r + smooth)
    ##
    hard_dice = tf.reduce_mean(hard_dice)
    return hard_dice

def Dice2Class(output, target, main_class, smooth = 1.0, name = 'dice_score'):
    last_dim_idx = target.get_shape().ndims - 1
    num_class = tf.shape(target)[last_dim_idx]
    predictions = tf.one_hot(output, num_class)
    preds_unrolled = tf.reshape(predictions,[-1,num_class])[:,main_class]
    targets_unrolled = tf.reshape(target,[-1,num_class])[:,main_class]
    intersection = tf.reduce_sum(preds_unrolled*targets_unrolled)
    ret_val = (2.0*intersection)/(tf.reduce_sum(preds_unrolled)
     + tf.reduce_sum(targets_unrolled) + smooth)
    ret_val = tf.identity(ret_val,name = 'dice_score')
    return ret_val

class MetricStream(object):
    def __init__(self,op):
        self.op = op
        count = tf.constant(1.0)
        self.sum = tf.Variable([0.0,0], name = op.name[:-2] + '_sum', trainable = False)
        self.avg = tf.Variable(0.0, name = op.name[:-2] + '_avg', trainable = False)
        self.accumulate = tf.assign_add(self.sum,[self.op,count])
        self.reset = tf.assign(self.sum,[0.0,0.0])
        self.stats = tf.assign(self.avg,self.sum[0]/(0.001 + self.sum[1]))


