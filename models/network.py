
from __future__ import division
import numpy as np
import sys, os
import tensorflow as tf
from network_ops import *

class FCMultiScaleResidualDenseNet(object):
    def __init__(self, 
                inputs, 
                targets, 
                weight_maps,
                batch_class_weights,
                num_class,  
                n_pool = 3, 
                n_feat_first_layer = [12, 12, 12], 
                growth_rate = 12,
                n_layers_per_block = [2, 3, 4, 5, 4, 3, 2], 
                weight_decay = 5e-6, 
                dropout_rate = 0.2, 
                optimizer = AdamOptimizer(1e-4),
                metrics_list = ['sW_CE_loss', 'mBW_Dice_loss', 'L2_loss', 'Total_loss', 'avgDice_score',
                                'Dice_class_1', 'Dice_class_2', 'Dice_class_3'],
                metrics_to_optimize_on = ['Total_loss']
                ):

        self.inputs = inputs
        self.targets = targets
        self.weight_maps = weight_maps
        self.batch_class_weights = batch_class_weights
        self.n_pool = n_pool
        self.n_feat_first_layer = n_feat_first_layer
        self.growth_rate = growth_rate
        self.n_layers_per_block = n_layers_per_block
        self.weight_decay = weight_decay
        self.dropout_rate = dropout_rate
        self.optimizer = optimizer
        self.initializer = tf.contrib.layers.variance_scaling_initializer(factor=2.0, 
                            mode='FAN_IN', uniform=False, seed=None, dtype=tf.float32)
        self.regularizer =  tf.contrib.layers.l2_regularizer(scale=1.0)
        self.is_training = tf.placeholder(tf.bool)
        self.num_channels = inputs.get_shape()[-1].value
        self.num_classes = num_class
        self.one_hot_labels = tf.one_hot(self.targets, depth=self.num_classes, axis=3)

        self.metrics_list = metrics_list
        self.metrics_to_optimize_on = metrics_to_optimize_on
        self.stats_ops = {}
        self.inference_ops = {}
        self.accumulate_ops = []
        self.reset_ops = []

        self.NetworkForwardPass(n_feat_first_layer = self.n_feat_first_layer,
                                n_pool = self.n_pool,
                                growth_rate = self.growth_rate,
                                n_layers_per_block = self.n_layers_per_block,
                                dropout_rate = self.dropout_rate
                        )                       
        self.Metrics()
        self.Optimize()
        self.Summaries()
        self.CountVariables()

    def NetworkForwardPass(self, n_feat_first_layer, n_pool, growth_rate, n_layers_per_block, dropout_rate):
        if type(n_layers_per_block) == list:
            assert (len(n_layers_per_block) == 2 * n_pool + 1)
        elif type(n_layers_per_block) == int:
            n_layers_per_block = [n_layers_per_block] * (2 * n_pool + 1)
        else:
            raise ValueError        

        inputs = self.inputs
        targets = tf.cast(self.targets, dtype=tf.float32)
        tf.summary.image('inputs',inputs, max_outputs = 4, collections = ['per_100_steps'])
        tf.summary.image('ground_truth',targets[:,:,:,None], max_outputs = 4, collections = ['per_100_steps'])

        stack = []
        # Inception Block
        l = tf.layers.conv2d(
                inputs=inputs,
                filters=n_feat_first_layer[0],
                kernel_size=[3, 3],
                padding="same",
                activation=None,
                kernel_initializer=self.initializer,
                bias_initializer=tf.zeros_initializer(),
                kernel_regularizer=self.regularizer
                )
        stack.append(l)

        l = tf.layers.conv2d(
                inputs=inputs,
                filters=n_feat_first_layer[1],
                kernel_size=[5, 5],
                padding="same",
                activation=None,
                kernel_initializer=self.initializer,
                bias_initializer=tf.zeros_initializer(),
                kernel_regularizer=self.regularizer
                )
        stack.append(l)

        l = tf.layers.conv2d(
                inputs=inputs,
                filters=n_feat_first_layer[2],
                kernel_size=[7, 7],
                padding="same",
                activation=None,
                kernel_initializer=self.initializer,
                bias_initializer=tf.zeros_initializer(),
                kernel_regularizer=self.regularizer
                )
        stack.append(l)

        n_filters = sum(n_feat_first_layer)
        stack = tf.concat(stack, 3)
        print("First Layer shape ", stack.get_shape().as_list())   
        skip_connection_list = []

        #######################
        #   Downsampling path   #
        #######################
        for i in range(n_pool):
            for j in range(n_layers_per_block[i]):
                l = BN_ELU_Conv(stack, growth_rate, dropout_rate=dropout_rate, is_training=self.is_training)
                stack = tf.concat([stack, l], 3)
                n_filters += growth_rate
            print("DB_Down:", i, " shape ", stack.get_shape().as_list())        

            proj_l = BN_ELU_Conv(stack, growth_rate*n_layers_per_block[i+1],
                        filter_size=1, dropout_rate=dropout_rate, is_training=self.is_training)
            print("DB Projection Layer:", " shape ", proj_l.get_shape().as_list())
            skip_connection_list.append(proj_l)
            stack = TransitionDown(stack, n_filters, dropout_rate=dropout_rate, is_training=self.is_training)
            print("TD:", i, " shape ", stack.get_shape().as_list()) 

        skip_connection_list = skip_connection_list[::-1]
        block_to_upsample = []

        #*********** Bottle-Neck Layer **********#
        proj_l = BN_ELU_Conv(stack, growth_rate*n_layers_per_block[n_pool],
                            filter_size=1, dropout_rate=dropout_rate, is_training=self.is_training)
        print("Bottleneck Projection Layer:", " shape ", proj_l.get_shape().as_list())

        for j in range(n_layers_per_block[n_pool]):
            l = BN_ELU_Conv(stack, growth_rate, dropout_rate=dropout_rate, is_training=self.is_training)
            block_to_upsample.append(l)
            stack = tf.concat([stack, l],3)

        block_to_upsample = tf.concat(block_to_upsample, axis=3)    
        block_to_upsample = tf.add(block_to_upsample, proj_l)
        print("Bottleneck:", " shape ", block_to_upsample.get_shape().as_list())

        #######################
        #   Upsampling path   #
        #######################

        for i in range(n_pool):
            n_filters_keep = growth_rate * n_layers_per_block[n_pool + i]
            stack = ResidualTransitionUp(skip_connection_list[i], block_to_upsample, n_filters_keep,
                     is_training=self.is_training)
            print("TU:", i, " shape ", stack.get_shape().as_list())  

            proj_l = BN_ELU_Conv(stack, growth_rate*n_layers_per_block[n_pool + i +1],
                                filter_size=1, dropout_rate=dropout_rate, is_training=self.is_training)
            print("Transition Up Projection Layer:", " shape ", proj_l.get_shape().as_list())
            block_to_upsample = []
            for j in range(n_layers_per_block[n_pool + i + 1]):
                l = BN_ELU_Conv(stack, growth_rate, dropout_rate=dropout_rate, is_training=self.is_training)
                block_to_upsample.append(l)
                stack = tf.concat([stack, l],3)
            block_to_upsample = tf.concat(block_to_upsample, axis=3)    
            block_to_upsample = tf.add(block_to_upsample, proj_l)    
            print("DB_Up:", i, " shape ", block_to_upsample.get_shape().as_list())
        print("Final Stack:", i, " shape ", stack.get_shape().as_list())

        #####################
        #       Outputs     #
        #####################
        self.logits = tf.layers.conv2d(inputs=block_to_upsample, 
                                        filters = self.num_classes,
                                        kernel_size=[1, 1],
                                        padding="same",
                                        activation=None,
                                        kernel_initializer=self.initializer,
                                        bias_initializer=tf.zeros_initializer(),
                                        kernel_regularizer=self.regularizer
                                        )

        self.posteriors =  tf.nn.softmax(self.logits)    

        print("Final Softmax Layer:", " shape ", self.posteriors.get_shape().as_list())
        # self.predictions = tf.cast(tf.argmax(self.logits,3), tf.float32)
        self.predictions = tf.argmax(self.logits,3)
        print("Predictions:", " shape ", self.predictions.get_shape().as_list())
        tf.summary.image('predictions', tf.cast(self.predictions[:,:,:,None], tf.float32), max_outputs = 4, collections = ['per_100_steps'])

    def Optimize(self):
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            self.train_op = self.optimizer.minimize(self.loss_op, global_step=tf.train.get_global_step())

    def Metrics(self):
        for metric_name in self.metrics_list:
            metric_implemented = False
            if metric_name == 'sW_CE_loss':
                metric_implemented = True
                with tf.variable_scope(metric_name):
                    self.inference_ops[metric_name] = sW_CE_loss = \
                         SpatialWeightedCrossEntropy(self.logits, self.one_hot_labels, self.weight_maps) 
                    metric_obj = MetricStream(sW_CE_loss)

                    tf.summary.scalar('per-step', metric_obj.op,collections = ['per_step'])
                    tf.summary.scalar('epoch-avg', metric_obj.avg,collections = ['per_epoch'])                   

            elif metric_name == 'mBW_Dice_loss':
                metric_implemented = True
                with tf.variable_scope(metric_name):
                    class_weights = self.batch_class_weights
                    dice_scores = dice_multiclass(self.posteriors, self.one_hot_labels)
                    avg_dice_score = tf.divide(tf.reduce_sum(class_weights* dice_scores), tf.reduce_sum(class_weights))
                    mBW_Dice_loss = tf.subtract(1.0, avg_dice_score)
                    self.inference_ops[metric_name] = mBW_Dice_loss
                    metric_obj = MetricStream(mBW_Dice_loss)
                    tf.summary.scalar('per-step', metric_obj.op,collections = ['per_step'])
                    tf.summary.scalar('epoch-avg',metric_obj.avg,collections = ['per_epoch'])

            elif metric_name == 'L2_loss':
                metric_implemented = True
                with tf.variable_scope(metric_name):
                    L2_loss = tf.contrib.layers.apply_regularization(tf.contrib.layers.l2_regularizer(scale=self.weight_decay),
                                tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))
                    self.inference_ops[metric_name] = L2_loss
                    metric_obj = MetricStream(L2_loss)
                    tf.summary.scalar('per-step', metric_obj.op,collections = ['per_step'])
                    tf.summary.scalar('epoch-avg', metric_obj.avg,collections = ['per_epoch'])       

            elif metric_name == 'Total_loss':
                metric_implemented = True
                with tf.variable_scope(metric_name):
                    # Total_loss: Spatially Weighted Cross-Entropy + mini-Batch Weighted dice loss + L2
                    eta=1.
                    gamma=1.
                    class_weights = self.batch_class_weights
                    dice_scores = dice_multiclass(self.posteriors, self.one_hot_labels)
                    avg_dice_score = tf.divide(tf.reduce_sum(class_weights* dice_scores), tf.reduce_sum(class_weights))
                    dice_loss = tf.subtract(1.0, avg_dice_score)
                    cross_entropy_loss = SpatialWeightedCrossEntropy(self.logits, self.one_hot_labels, self.weight_maps)
                    l2_loss = tf.contrib.layers.apply_regularization(tf.contrib.layers.l2_regularizer(scale=self.weight_decay),
                        tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))
                    Total_loss = eta*dice_loss + gamma*cross_entropy_loss + l2_loss 

                    self.inference_ops[metric_name] = Total_loss
                    metric_obj = MetricStream(Total_loss)
                    tf.summary.scalar('per-step',metric_obj.op, collections = ['per_step'])
                    tf.summary.scalar('epoch-avg',metric_obj.avg, collections = ['per_epoch'])
                    
            elif metric_name == 'CE_loss':
                metric_implemented = True
                with tf.variable_scope(metric_name):
                    CE_Loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels = self.one_hot_labels, logits = self.logits))
                    self.inference_ops[metric_name] = CE_Loss
                    #class weights should be in float
                    metric_obj = MetricStream(CE_Loss)

                    tf.summary.scalar('per-step',metric_obj.op,collections = ['per_step'])
                    tf.summary.scalar('epoch-avg',metric_obj.avg,collections = ['per_epoch'])   

            elif metric_name == 'Dice_loss':
                metric_implemented = True
                with tf.variable_scope(metric_name):
                    class_weights = tf.constant([1.0]*self.num_classes, dtype=tf.float32)   

                    dice_scores = dice_multiclass(self.posteriors, self.one_hot_labels)
                    avg_dice_score = tf.divide(tf.reduce_sum(class_weights* dice_scores), tf.reduce_sum(class_weights))
                    Dice_loss = tf.subtract(1.0, avg_dice_score)
                    self.inference_ops[metric_name] = Dice_loss
                    metric_obj = MetricStream(Dice_loss)
                    tf.summary.scalar('per-step',metric_obj.op,collections = ['per_step'])
                    tf.summary.scalar('epoch-avg',metric_obj.avg,collections = ['per_epoch'])

            elif metric_name == 'avgDice_score':
                metric_implemented = True
                with tf.variable_scope(metric_name):
                    avgDice_score = dice_hard_coe(self.predictions, self.one_hot_labels)
                    self.inference_ops[metric_name] = avgDice_score
                    metric_obj = MetricStream(avgDice_score)
                    tf.summary.scalar('per-step',metric_obj.op,collections = ['per_step'])
                    tf.summary.scalar('epoch-avg',metric_obj.avg,collections = ['per_epoch'])

            elif metric_name == 'Dice_class_1':
                metric_implemented = True
                with tf.variable_scope(metric_name):
                    self.inference_ops[metric_name] = dice_score = \
                        Dice2Class(self.predictions,  self.one_hot_labels, main_class=1)

                    self.inference_ops[metric_name] = dice_score
                    metric_obj = MetricStream(dice_score)
                    tf.summary.scalar('per-step',metric_obj.op,collections = ['per_step'])
                    tf.summary.scalar('epoch-avg',metric_obj.avg,collections = ['per_epoch'])

            elif metric_name == 'Dice_class_2':
                metric_implemented = True
                with tf.variable_scope(metric_name):
                    self.inference_ops[metric_name] = dice_score = \
                        Dice2Class(self.predictions,  self.one_hot_labels, main_class=2)

                    self.inference_ops[metric_name] = dice_score
                    metric_obj = MetricStream(dice_score)
                    tf.summary.scalar('per-step',metric_obj.op,collections = ['per_step'])
                    tf.summary.scalar('epoch-avg',metric_obj.avg,collections = ['per_epoch'])

            elif metric_name == 'Dice_class_3':
                metric_implemented = True
                with tf.variable_scope(metric_name):
                    self.inference_ops[metric_name] = dice_score = \
                        Dice2Class(self.predictions,  self.one_hot_labels, main_class=3)
                    self.inference_ops[metric_name] = dice_score
                    metric_obj = MetricStream(dice_score)
                    tf.summary.scalar('per-step',metric_obj.op,collections = ['per_step'])
                    tf.summary.scalar('epoch-avg',metric_obj.avg,collections = ['per_epoch'])                    
            else:
                print('Error : ' + metric_name + ' is not implemented')
            
            if metric_implemented == True: 
                try: 
                    self.stats_ops[metric_name] = metric_obj.stats
                    self.accumulate_ops.append(metric_obj.accumulate)
                    self.reset_ops.append(metric_obj.reset)
                except AttributeError:
                    pass           
        self.loss_op = sum([self.inference_ops[metric] for metric in self.metrics_to_optimize_on])

    def Summaries(self):
        self.summary_ops = {}       
        self.summary_ops['1step'] = tf.summary.merge_all(key = 'per_step')
        self.summary_ops['100steps'] = tf.summary.merge_all(key = 'per_100_steps')  
        self.summary_ops['1epoch'] = tf.summary.merge_all(key = 'per_epoch')

    def CountVariables(self):    
        total_parameters = 0
        #iterating over all variables
        for variable in tf.trainable_variables():  
            local_parameters=1
            shape = variable.get_shape()  #getting shape of a variable
            for i in shape:
                local_parameters*=i.value  #mutiplying dimension values
            total_parameters+=local_parameters
        print('Total Number of Trainable Parameters:', total_parameters)


if __name__ == '__main__':
    # Set Environment
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    num_channels = 1
    num_class = 4
    inputs = tf.placeholder(tf.float32, shape=(None, None, None, num_channels))
    targets = tf.placeholder(tf.uint8, shape = (None, None, None))
    weight_maps = tf.placeholder(tf.float32, shape=[None, None, None])
    batch_class_weights = tf.placeholder(tf.float32)

    # define the net
    print('Defining the network')
    net = FCMultiScaleResidualDenseNet(inputs,
                          targets, 
                          weight_maps,
                          batch_class_weights,
                          num_class=num_class)
