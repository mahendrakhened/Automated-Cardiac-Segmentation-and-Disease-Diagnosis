import argparse
import os, sys, shutil
import numpy as np 
import time
from estimator import *
from config import *
sys.path.insert(0,'../models/')
from network import *
from network_ops import *



if __name__ == '__main__':
    conf = conf()
    inputs = tf.placeholder(tf.float32, shape=(None, None, None, conf.num_channels))
    targets = tf.placeholder(tf.uint8, shape = (None, None, None))
    weight_maps = tf.placeholder(tf.float32, shape=[None, None, None])
    batch_class_weights = tf.placeholder(tf.float32)

    # define the net
    print('Defining the network', conf.run_name)
    # model = FCMultiScaleResidualDenseNet(inputs,
    #                         targets, 
    #                         weight_maps,
    #                         batch_class_weights,
    #                         num_class=conf.num_class,
    #                         n_pool = 3, 
    #                         n_feat_first_layer = [12, 12, 12], 
    #                         growth_rate = 12,
    #                         n_layers_per_block = [2, 3, 4, 5, 4, 3, 2], 
    #                         weight_decay = 5e-6, 
    #                         dropout_rate = 0.2, 
    #                         optimizer = AdamOptimizer(conf.learning_rate),
    #                         metrics_list = ['sW_CE_loss', 'mBW_Dice_loss', 'L2_loss', 'Total_loss', 'avgDice_score',
    #                                         'Dice_class_1', 'Dice_class_2', 'Dice_class_3'],
    #                         metrics_to_optimize_on = ['Total_loss']
    #                       )

    model = FCMultiScaleResidualDenseNet(inputs,
                            targets, 
                            weight_maps,
                            batch_class_weights,
                            num_class=conf.num_class,
                            n_pool = 3, 
                            n_feat_first_layer = [16, 16, 16], 
                            growth_rate = 16,
                            n_layers_per_block = [2, 3, 4, 5, 4, 3, 2], 
                            weight_decay = 5e-6, 
                            dropout_rate = 0.2, 
                            optimizer = AdamOptimizer(conf.learning_rate),
                            metrics_list = ['sW_CE_loss', 'mBW_Dice_loss', 'L2_loss', 'Total_loss', 'avgDice_score',
                                            'Dice_class_1', 'Dice_class_2', 'Dice_class_3'],
                            metrics_to_optimize_on = ['Total_loss']
                          )

    # initialise the estimator with the net
    print('Preparing the estimator..')
    trainer = Estimator(model = model,
                        conf = conf
                        )
    # iterate for the number of epochs
    for epoch in range(int(trainer.numKeeper.counts['epoch']+1), conf.num_epochs):
        print('\n\n$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$')
        print('Training @ Epoch : ' + str(epoch))
        trainer.Fit(steps=-1)
        print('\n---------------------------------------------------')
        print('Validating @ Epoch : ' + str(epoch))
        trainer.Evaluate(steps=-1 )
        trainer.numKeeper.counts['epoch'] = trainer.numKeeper.counts['epoch'] + 1
        trainer.numKeeper.UpdateCounts(trainer.summary_manager.counts)
        trainer.SaveModel(os.path.join(conf.output_dir,conf.run_name,'models','latest.ckpt'))




