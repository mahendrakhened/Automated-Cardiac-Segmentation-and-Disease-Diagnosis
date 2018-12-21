from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensorflow.python import debug as tf_debug 
import numpy as np 
import os, sys
import pprint
import time, argparse
import collections
sys.path.insert(0,'../data_loaders/')
from hdf5_loader import DataIterator
from helpers.utils import imshow, progress_bar


class Estimator(object):
    """docstring for Estimator"""
    def __init__(self, model, conf):
        self.model = model
        self.conf = conf
        print ('Initializing iterators for dataloaders')
        self.train_iterator = DataIterator(conf.data_path, conf.transformation_params, 
                                mode = 'train', batch_size = conf.batch_size, num_threads=4)
        self.valid_iterator = DataIterator(conf.data_path, conf.transformation_params,
                                mode = 'valid', batch_size = conf.batch_size, num_threads=4)
        print ('Training & Validation Batches', self.train_iterator.steps, self.valid_iterator.steps)

        print('preparing summary manager')
        self.summary_manager = SummaryManager(conf.summary_dir, tf.get_default_graph(), conf.freq_list)

        print('Initialising the numbers manager')
        self.numKeeper = NumbersKeeper()

        #initalising NumbersKeeper
        counts_initializer = {'epoch' : 0, 'avgDice_score' : 0,
                              'Dice_class_1' : 0, 'Dice_class_2' : 0, 'Dice_class_3' : 0}

        counts_initializer['train'] = self.summary_manager.counts['train']
        counts_initializer['valid'] = self.summary_manager.counts['valid']

        self.numKeeper.InitNumpyDict(counts_initializer)

        print('Defining the session')
        sess_config = tf.ConfigProto()
        sess_config.allow_soft_placement = True
        sess_config.gpu_options.allow_growth = True
        self.sess = tf.Session(config = sess_config)
        # self.sess = tf_debug.TensorBoardDebugWrapperSession(self.sess, "localhost:7000")
        self.sess.run(tf.global_variables_initializer())        
        try:
            self.sess.run(tf.assert_variables_initialized())
        except tf.errors.FailedPreconditionError:
            raise RuntimeError('Not all variables initialized')

        self.saver = tf.train.Saver(tf.global_variables())
        if conf.load_pretrained_model_from: 
            if conf.resume_training:
                print('Restoring model from: ' + str(conf.load_pretrained_model_from))
                self.saver.restore(self.sess, conf.load_pretrained_model_from)
                self.numKeeper.UpdateCounts(self.sess.run(self.numKeeper.tf_counts))
                print("\nWhile restoring : ")
                print(self.sess.run(self.numKeeper.tf_counts))
                print("\n")

                self.summary_manager.UpdateCounts(self.numKeeper.counts)
                print('Epochs completed : ' + str(self.numKeeper.counts['epoch']))
                print('Best average Dice: ' + str(self.numKeeper.counts['avgDice_score']))
            else:
                print('Restoring model from: ' + str(conf.load_pretrained_model_from))
                self.saver.restore(self.sess, conf.load_pretrained_model_from)

    def SaveModel(self, save_path):
        self.sess.run(self.numKeeper.AssignNpToTfVariables(self.numKeeper.counts))
        model_dir = os.path.split(save_path)[0]
        if not os.path.isdir(model_dir): 
            os.makedirs(model_dir)
        self.saver.save(self.sess, save_path)

    def Fit(self, steps=1000):
        self.train_iterator.Reset()
        self.summary_manager.mode = 'train'
        time.sleep(5.0)
        feed = None
 
        train_ops = [   self.model.logits,
                        self.model.summary_ops['1step'],
                        self.model.inference_ops,
                        self.model.accumulate_ops,
                        self.model.train_op,
                    ]
        
        count = 0
        step_sec = 0
        if steps <= 0:
            steps = self.train_iterator.steps
        while (count < steps):
            start_time = time.time()

            # fetch inputs batches and verify if they are numpy.ndarray and run all the ops
            # g_time = time.time()
            input_batch, target_batch, weight_batch, batch_class_weights, _ = self.train_iterator.GetNextBatch()
            # print("time taken to get a batch : " + str(time.time()-g_time) + 's')

            if type(input_batch) is np.ndarray:

                feed = {self.model.inputs : input_batch,
                        self.model.targets : target_batch,
                        self.model.weight_maps : weight_batch,
                        self.model.batch_class_weights : batch_class_weights,
                        self.model.is_training : True}
                input_batch, target_batch, weight_batch, batch_class_weights = None, None, None, None
                # i_time = time.time()
                outputs, summary, _, _, _ = self.sess.run(train_ops, feed_dict = feed)
                # print("time taken to for inference: " + str(time.time()-i_time) + 's')
                # print("\n")
                self.summary_manager.AddSummary(summary, "train", "per_step")
                progress_bar(count%steps+1, steps, step_sec)

                outputs, summary = None, None

                # add summaries regularly for every 100 steps or no. of steps to finish an epoch
                if self.train_iterator.steps <= 100:
                    save_step = self.train_iterator.steps //2
                else:
                    save_step = 100

                if (count + 1)% save_step == 0:
                    summary = self.sess.run(self.model.summary_ops['100steps'], feed_dict = feed) 
                    self.summary_manager.AddSummary(summary, "train", "per_100_steps")

                if (count + 1) % 250 == 0:
                    print('Avg metrics : ')
                    pprint.pprint(self.sess.run(self.model.stats_ops), width = 1)
                count = count + 1 
              
            stop_time = time.time()
            step_sec = stop_time - start_time
            if self.train_iterator.iter_over == True:
                # print('\nIteration over')
                self.train_iterator.Reset()
                time.sleep(4)

        print('\nAvg metrics for epoch : ')
        pprint.pprint(self.sess.run(self.model.stats_ops), width = 1)
        summary = self.sess.run(self.model.summary_ops['1epoch'], feed_dict = feed)
        self.sess.run(self.model.reset_ops)
        self.summary_manager.AddSummary(summary,"train","per_epoch")
        summary = None


    def Evaluate(self, steps=250):
        #set mode and wait for the threads to populate the queue
        self.valid_iterator.Reset()
        self.summary_manager.mode = 'valid'
        time.sleep(5.0)
        
        feed = None
        valid_ops = [self.model.logits,
                     self.model.summary_ops['1step'],
                     self.model.inference_ops,
                     self.model.accumulate_ops]

        #iterate the validation step, until count = steps
        count = 0
        step_sec = 0
        if steps <= 0:
            steps = self.valid_iterator.steps
        while (count < steps):
            start_time = time.time()
            time.sleep(0.4)


            input_batch, target_batch, weight_batch, batch_class_weights, _ = self.valid_iterator.GetNextBatch()

            if type(input_batch) is np.ndarray:

                feed = {self.model.inputs : input_batch,
                        self.model.targets : target_batch,
                        self.model.weight_maps : weight_batch,
                        self.model.batch_class_weights : batch_class_weights,
                        self.model.is_training : False}
                input_batch, target_batch, weight_batch, batch_class_weights = None, None, None, None

                
                outputs, summary, _, _ = self.sess.run(valid_ops, feed_dict = feed) 
                self.summary_manager.AddSummary(summary,"valid", "per_step")
                progress_bar(count%steps+1, steps, step_sec)
                outputs, summary = None, None

                # add summaries regularly for every 100 steps or no. of steps to finish an epoch
                if self.valid_iterator.steps <= 100:
                    save_step = self.valid_iterator.steps //2
                else:
                    save_step = 100
                
                if (count+1) % save_step == 0:
                    summary = self.sess.run(self.model.summary_ops['100steps'], feed_dict = feed)
                    self.summary_manager.AddSummary(summary,"valid", "per_100_steps")

                if (count+1) % 250 == 0:
                    print('Avg metrics : ')
                    pprint.pprint(self.sess.run(self.model.stats_ops), width = 1)

                count = count + 1 
            stop_time = time.time()
            step_sec = stop_time - start_time
            if self.valid_iterator.iter_over == True:
                # print('\nIteration over')
                self.valid_iterator.Reset()
                time.sleep(5)

        print('\nAvg metrics for epoch : ')
        metrics = self.sess.run(self.model.stats_ops)
        pprint.pprint(metrics, width=1)
        if (metrics['avgDice_score'] > self.numKeeper.counts['avgDice_score']):
            self.numKeeper.counts['avgDice_score'] = metrics['avgDice_score']
            self.numKeeper.UpdateCounts(self.summary_manager.counts)
            print('Saving best model for all classes!')
            self.SaveModel(os.path.join(self.conf.output_dir,self.conf.run_name,'best_model','latest.ckpt'))

        if (metrics['Dice_class_1'] > self.numKeeper.counts['Dice_class_1']):
            self.numKeeper.counts['Dice_class_1'] = metrics['Dice_class_1']
            self.numKeeper.UpdateCounts(self.summary_manager.counts)
            print('Saving best model for class 1!')
            self.SaveModel(os.path.join(self.conf.output_dir,self.conf.run_name,'best_model_class1','latest.ckpt'))

        if (metrics['Dice_class_2'] > self.numKeeper.counts['Dice_class_2']):
            self.numKeeper.counts['Dice_class_2'] = metrics['Dice_class_2']
            self.numKeeper.UpdateCounts(self.summary_manager.counts)
            print('Saving best model for class 2!')
            self.SaveModel(os.path.join(self.conf.output_dir,self.conf.run_name,'best_model_class2','latest.ckpt'))

        if (metrics['Dice_class_3'] > self.numKeeper.counts['Dice_class_3']):
            self.numKeeper.counts['Dice_class_3'] = metrics['Dice_class_3']
            self.numKeeper.UpdateCounts(self.summary_manager.counts)
            print('Saving best model for class 3!')
            self.SaveModel(os.path.join(self.conf.output_dir,self.conf.run_name,'best_model_class3','latest.ckpt'))

        print('Current best average Dice: ' + str(self.numKeeper.counts['avgDice_score']))
        summary = self.sess.run(self.model.summary_ops['1epoch'], feed_dict = feed)
        self.sess.run(self.model.reset_ops)
        self.summary_manager.AddSummary(summary, "valid", "per_epoch")
        summary = None
    
class SummaryManager(object):
    def __init__(self, summary_dir, model_graph, freq_list, mode='train'):
        self.summary_dir = summary_dir
        self.model_graph = model_graph
        self.mode = mode

        # Create the different directories to save summaries of train, valid 
        self.train_dir = self.CheckCreateFolder(os.path.join(summary_dir,'train'))
        self.valid_dir = self.CheckCreateFolder(os.path.join(summary_dir,'valid'))

        #Create different summary writers for train, valid
        self.trainWriter = tf.summary.FileWriter(self.train_dir,model_graph)
        self.validWriter = tf.summary.FileWriter(self.valid_dir,model_graph)

        # Initialise the counts
        self.counts = {'train': {},'valid': {}}

        # Initialise the counts
        self.InitCounts(freq_list)

    def UpdateCounts(self, count_dict):
        for mode, val1 in count_dict.items():
            if isinstance(val1,collections.Mapping):
                for freq, val in count_dict[mode].items():
                    self.counts[mode][freq] = val

    def InitCounts(self, freq_list):
        for each in freq_list:
            self.counts['train'][each] = 0
            self.counts['valid'][each] = 0

    def CheckCreateFolder(self,path):
        if not os.path.exists(path):
            os.makedirs(path)
        return path

    def AddSummary(self,summary,mode,time_step_name):
        if self.mode == 'train':
            self.trainWriter.add_summary(summary,self.counts[mode][time_step_name])
            self.counts[mode][time_step_name] += 1
        elif self.mode == 'valid':
            self.validWriter.add_summary(summary,self.counts[mode][time_step_name])
            self.counts[mode][time_step_name] += 1

        else:
            print("summary not added !!!, accepted values for variable mode are 'train', 'valid' ")
            raise 1


class NumbersKeeper(object):
    def __init__(self):
        self.counts = {'train':{},'valid':{}}
        self.tf_counts = {'train':{},'valid':{}}

    def UpdateCounts(self, count_dict):
        for mode, val1 in count_dict.items():
            if isinstance(val1,collections.Mapping):
                for freq, val in count_dict[mode].items():
                    self.counts[mode][freq] = val
            else:
                self.counts[mode] = val1

    def InitNumpyDict(self,count_dict):
        for mode,val1 in count_dict.items():
            if isinstance(val1,collections.Mapping):
                for freq, val in count_dict[mode].items():
                    self.counts[mode][freq] = val
            else:
                self.counts[mode] = val1

        self.InitTfVariables(count_dict)

    def InitTfVariables(self,count_dict):
        # print(count_dict)
        for mode,val1 in count_dict.items():
            if isinstance(val1,collections.Mapping):
                for freq, val in count_dict[mode].items():
                    self.tf_counts[mode][freq] = tf.Variable(val,name=mode+freq,trainable=False)
            else:
                self.tf_counts[mode] = tf.Variable(val1,dtype=tf.float32,name=mode,trainable=False)


    def AssignNpToTfVariables(self,count_dict):
        assign_ops = []
        for mode,val1 in count_dict.items():
            if isinstance(val1,collections.Mapping):
                for freq, val in count_dict[mode].items():
                    assign_ops.append(tf.assign(self.tf_counts[mode][freq],val))
            else:
                assign_ops.append(tf.assign(self.tf_counts[mode],val1))

        return assign_ops

if __name__ == '__main__':
    pass
