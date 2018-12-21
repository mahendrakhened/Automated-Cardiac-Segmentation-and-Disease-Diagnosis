from __future__ import division
import numpy as np
import os, shutil, sys
import SimpleITK as sitk
import glob
from datetime import datetime
import time
import re, nibabel
import pandas as pd

from config import *
from test_utils import *

sys.path.insert(0,'../models/')
from network import *
from network_ops import *


if __name__ == "__main__":
    # Set Environment
    conf = conf()

    save_dir = os.path.join(conf.output_dir, conf.run_name, 'predictions{}'.format(time.strftime("%Y%m%d_%H%M%S")))
    if os.path.exists(save_dir):
        shutil.rmtree(save_dir)
    os.makedirs(save_dir)
    # If prediction to be made on saving criteria for best class of the model. Here the 
    # classes are 1,2,3 for RV, MYO and LV.

    # models = ['best_model/latest.ckpt', 'best_model_class1/latest.ckpt', 'best_model_class2/latest.ckpt',
    #           'best_model_class3/latest.ckpt']
    # Prediction based on the model with best dice score for MYO class on held out validation set
    models = ['best_model_class2/latest.ckpt']

    for model in models:   
        saved_model_dir = os.path.join(save_dir, model.split('/')[0]) 
        os.makedirs(saved_model_dir)
        model_path = os.path.join(conf.output_dir, conf.run_name, model)
        # gt_available=False -> Implies Ground Truth available for benchmarking on validation or test set
        # Then the metrics reported in the paper are calculated
        # gt_available = True
        # final_test_data_path = ['../../processed_acdc_dataset/dataset/test_set',
        #                         '../../processed_acdc_dataset/dataset/validation_set']

        # Actual ACDC testing dataset 
        final_test_data_path = ['../../ACDC_DataSet/testing']
        gt_available = False

        inputs = tf.placeholder(tf.float32, shape=(None, None, None, conf.num_channels))
        targets = tf.placeholder(tf.uint8, shape = (None, None, None))
        weight_maps = tf.placeholder(tf.float32, shape=[None, None, None])
        batch_class_weights = tf.placeholder(tf.float32)


        # define the network
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
        print('Preparing the Tester..')
        tester = Tester(model, conf, model_path)
        results = []
        patients = []
        for test_data in final_test_data_path:
            base_folder = os.path.basename(test_data)
            patient_folders = next(os.walk(test_data))[1]   
            patient_folders = sorted(patient_folders, key=lambda x: re.findall('\d+', x))
            for patient in patient_folders:
                os.makedirs(os.path.join(saved_model_dir, base_folder, patient))
            for i in range(len(patient_folders)):
                print("\n-----------------------------------------------------------------------")
                print("Working on " + patient_folders[i])
                # if patient_folders[i] == 'patient147':
                patient_data = get_test_data_acdc(test_data, patient_folders[i], 
                                                gt_available=gt_available)
                # print (img_file_list[0])
                print("-----------------------------------------------------------------------\n")
                i_time = time.time()
                result = tester.LoadAndPredict(patient_data, None, None, outputs_shape = None,
                                        preProcessList = ['roi_detect', 'normalize', 'roi_crop'],
                                        postProcessList = ['glcc', 'glcc_2D', 'pad_patch'],
                                        # preProcessList = ['crop_pad_4d', 'normalize'],
                                        # postProcessList = ['glcc', 'glcc_2D'],
                                        crf = None,
                                        patch_size=(128, 128),
                                        save_path = os.path.join(saved_model_dir, base_folder, patient_folders[i])
                                        )
                print("Time taken for Prediction: " + str(time.time()-i_time) + 's')
                print("\n")
                results.append(result)
                patients.append(patient_folders[i])


        tf.reset_default_graph()        
        if gt_available:
            HEADER = ["Name", "ED_dice_RV", "ED_jaccard_RV", "ED_Avg_Hausdorff_RV", "ED_Hausdorff_RV",
                      "ED_dice_MYO", "ED_jaccard_MYO", "ED_Avg_Hausdorff_MYO", "ED_Hausdorff_MYO", 
                      "ED_dice_LV", "ED_jaccard_LV", "ED_Avg_Hausdorff_LV", "ED_Hausdorff_LV", 
                      "ES_dice_RV", "ES_jaccard_RV", "ES_Avg_Hausdorff_RV", "ES_Hausdorff_RV",
                      "ES_dice_MYO","ES_jaccard_MYO", "ES_Avg_Hausdorff_MYO", "ES_Hausdorff_MYO",
                      "ES_dice_LV", "ES_jaccard_LV", "ES_Avg_Hausdorff_LV", "ES_Hausdorff_LV"]


            table = [[n,] + r for r, n in zip(results, patients)]
            df = pd.DataFrame(table, columns=HEADER)
            # TODO: Compute statistics 

            # Select the features:
            START_COL = 1
            END_COL = 25
            features = list(df.columns[np.r_[START_COL:END_COL]])
            print ('Statistics\n')
            print ('\nAverage\n')
            print (np.mean(df[features]))
            print ('\nStandard deviation\n')
            print (np.std(df[features]))
            print ('\nBest\n')
            print (np.max(df[features]))
            print ('\nWorst\n')
            print (np.min(df[features]))
            df.to_csv(os.path.join(saved_model_dir, "results_{}.csv".format(time.strftime("%Y%m%d_%H%M%S"))), index=False)

            print ("Writing report to file")
            target = open(os.path.join(saved_model_dir, "report.txt"), 'w')
            target.write('\nAverage\n')
            target.write(str(np.mean(df[features])))
            target.write("\n")

            target.write('\nStandard deviation\n')
            target.write(str(np.std(df[features])))
            target.write("\n")

            target.write('\nBest Dice worst HD\n')
            target.write(str(np.max(df[features])))
            target.write("\n")

            target.write('\nWorst Dice, best HD\n')
            target.write(str(np.min(df[features])))
            target.write("\n")
            target.close() 