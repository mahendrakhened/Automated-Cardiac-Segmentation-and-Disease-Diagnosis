# Imports
from __future__ import division
import numpy as np
import os, sys, shutil, re
import random, time
import scipy.ndimage as snd
import random
from datetime import datetime
import glob
import matplotlib.pyplot as plt
import SimpleITK as sitk
import pandas as pd 
from collections import OrderedDict
import nibabel as nib
import tensorflow as tf

sys.path.append("../helpers")
from utils import *
sys.path.insert(0,'../data_loaders/')
from data_augmentation import *
sys.path.insert(0,'../data_preprocess/')
from acdc_data_preparation import extract_roi_stddev
rng = np.random.RandomState(40)

class Tester(object):
    """docstring for Tester"""
    def __init__(self, model, conf, model_path=None):
        self.model = model
        self.conf = conf

        print('Defining the session')
        sess_config = tf.ConfigProto()
        sess_config.allow_soft_placement = True
        sess_config.gpu_options.allow_growth = True
        self.sess = tf.Session(config = sess_config)
        self.sess.run(tf.global_variables_initializer())        
        try:
            self.sess.run(tf.assert_variables_initialized())
        except tf.errors.FailedPreconditionError:
            raise RuntimeError('Not all variables initialized')

        self.saver = tf.train.Saver(tf.global_variables())
        if model_path:
            print('Restoring model from: ' + str(model_path))
            self.saver.restore(self.sess, model_path)

        self.binary_opening_filter = sitk.BinaryMorphologicalOpeningImageFilter()
        self.binary_opening_filter.SetKernelRadius(1)

        self.binary_closing_filter = sitk.BinaryMorphologicalClosingImageFilter()
        self.binary_closing_filter.SetKernelRadius(1)

        self.erosion_filter = sitk.BinaryErodeImageFilter()
        self.erosion_filter.SetKernelRadius(1)

        self.dilation_filter = sitk.BinaryDilateImageFilter()
        self.dilation_filter.SetKernelRadius(1)

    def LoadAndPreprocessVolume(self, patient_data, preProcessList, patch_size=(128, 128), max_size=(256, 256)):
        """
        TODO: Check
        """

        for each in preProcessList:
            if each == 'crop_pad_4d':
                # print("Crop or pad image to max supported size and get 4D volume", max_size)
                # TODO: Bugfix: output is not same as input for some shapes
                vol_4d, pixel_spacing = patient_data['4D'], patient_data['4D_hdr'].get_zooms()
                self.input_img = {}
                self.input_img['shape'] = vol_4d.shape
                self.input_img['spacing'] = pixel_spacing
                self.input_img['resized'] = False

            elif each == 'normalize':
                # print("Applying Slicewise standardization with adding bias and later clipping-[0,1]...")               
                # vol_4d = slicewise_normalization(vol_4d, scheme='minmax')
                vol_4d = slicewise_normalization(vol_4d, scheme='zscore')
                # vol_4d = slicewise_normalization(vol_4d, scheme='truncated_zscore')
                # vol_4d = slicewise_normalization(vol_4d, scheme='None')
                # vol_4d = phasewise_normalization(vol_4d, scheme='zscore')

            elif each == 'roi_detect':
                # print("ROI Detection")
                # Do ROI Extraction from 4-D Data (3D+time): Specifc to Cine-MRI cardiac preprocessing
                # Form 4D-Matrix from 2D files and do fourier-hough analysis 
                vol_4d, pixel_spacing = patient_data['4D'], patient_data['4D_hdr'].get_zooms()
                print (vol_4d.shape)
                self.input_img = {}
                self.input_img['shape'] = vol_4d.shape
                self.input_img['spacing'] = pixel_spacing
                self.input_img['resized'] = False
                roi_center, radii = extract_roi_stddev(vol_4d, self.input_img['spacing'])
                self.input_img['roi_center'] = roi_center
                self.input_img['roi_max_radius'] = radii[1]         
                self.input_img['pad_params'] = None

            elif each == 'roi_crop':
                # print("Cropping a patch centered around ROI", patch_size)
                roi_center = self.input_img['roi_center'] 
                max_radius = self.input_img['roi_max_radius']
                # print (vol_4d.shape, (roi_center,radii))
                # boResizeReqd = CheckIfResizeRequired(vol_4d[:,:, 0, 0], max_radius, patch_size, max_size)
                # print (boResizeReqd)
                boResizeReqd = False #TODO: Check with ACDC
                self.input_img['resized'] = boResizeReqd
                out_vol = np.zeros(patch_size+vol_4d.shape[2:])
                for phase in range(vol_4d.shape[3]):
                    for slice in range(vol_4d.shape[2]):
                        image = vol_4d[:,:, slice, phase]
                        if not boResizeReqd:
                            # Center around roi
                            vol_2d, pad_params = extract_patch(image, roi_center, patch_size)
                        else:
                            vol_2d, pad_params = extract_patch(image, roi_center, max_size)
                            vol_2d = resize_sitk_2D(vol_2d, patch_size)                
                        out_vol[:,:,slice, phase] = vol_2d
                self.input_img['pad_params'] = pad_params
                vol_4d = out_vol

        return vol_4d


    def LoadAndPreprocessSlice(self, img_files_path_list, preProcessList, patch_size=(128, 128), max_size=(256, 256)):
        """
        TODO: Check
        """

        for each in preProcessList:
            if each == 'crop_pad_4d':
                # print("Crop or pad image to max supported size and get 4D volume", max_size)
                vol_4d, pixel_spacing = get_4D_volume_of_fixed_shape(img_files_path_list, max_size)
                self.input_img = {}
                self.input_img['shape'] = vol_4d.shape
                self.input_img['spacing'] = pixel_spacing
                self.input_img['resized'] = False

            elif each == 'normalize':
                # print("Applying Slicewise standardization with adding bias and later clipping-[0,1]...")               
                # vol_4d = slicewise_normalization(vol_4d, scheme='minmax')
                vol_4d = slicewise_normalization(vol_4d, scheme='zscore')

            elif each == 'roi_detect':
                # print("ROI Detection")
                # Do ROI Extraction from 4-D Data (3D+time): Specifc to Cine-MRI cardiac preprocessing
                # Form 4D-Matrix from 2D files and do fourier-hough analysis 
                vol_4d, pixel_spacing = get_4D_volume(img_files_path_list)
                print (vol_4d.shape)
                self.input_img = {}
                self.input_img['shape'] = vol_4d.shape
                self.input_img['spacing'] = pixel_spacing
                self.input_img['resized'] = False
                roi_center, radii = extract_roi_stddev(vol_4d, self.input_img['spacing'])
                self.input_img['roi_center'] = roi_center
                self.input_img['roi_max_radius'] = radii[1]         
                self.input_img['pad_params'] = None

            elif each == 'roi_crop':
                # print("Cropping a patch centered around ROI", patch_size)
                roi_center = self.input_img['roi_center'] 
                max_radius = self.input_img['roi_max_radius']
                # print (vol_4d.shape, (roi_center,radii))
                boResizeReqd = CheckIfResizeRequired(vol_4d[:,:, 0, 0], max_radius, patch_size, max_size)
                # print (boResizeReqd)
                self.input_img['resized'] = boResizeReqd
                out_vol = np.zeros(patch_size+vol_4d.shape[2:])
                for phase in range(vol_4d.shape[3]):
                    for slice in range(vol_4d.shape[2]):
                        image = vol_4d[:,:, slice, phase]
                        if not boResizeReqd:
                            # Center around roi
                            vol_2d, pad_params = extract_patch(image, roi_center, patch_size)
                        else:
                            vol_2d, pad_params = extract_patch(image, roi_center, max_size)
                            vol_2d = resize_sitk_2D(vol_2d, patch_size)                
                        out_vol[:,:,slice, phase] = vol_2d
                self.input_img['pad_params'] = pad_params
                vol_4d = out_vol

        return vol_4d           

    def PostProcessVolume(self, output_sitk_img, postProcessList, roi_mask_path):
        """
        Do the list of post-processing tasks
        """
        # print("Post processing the outputs ...")
        for each in postProcessList:
            if each == 'binary_opening':
                output_sitk_img = self.binary_opening_filter.Execute(output_sitk_img)
            
            elif each == 'binary_erosion':
                output_sitk_img = self.erosion_filter.Execute(output_sitk_img)

            elif each == 'binary_dilation':
                output_sitk_img = self.dilation_filter.Execute(output_sitk_img)

            elif each == 'binary_closing':
                output_sitk_img = self.binary_closing_filter.Execute(output_sitk_img)
            
            elif each == 'glcc':
                output_sitk_img = getLargestConnectedComponent(output_sitk_img)

            elif each == 'glcc_2D':
                output_sitk_img = getLargestConnectedComponent_2D(output_sitk_img)
            
            elif each == 'maskroi':
                output_sitk_img = maskROIInPrediction(output_sitk_img, roi_mask_path)

        return output_sitk_img

    def PostPadding(self, seg_post_3d, postProcessList, max_size=(256, 256)):
        """
        Handle : Resizing or post padding operations to get back image to original shape
        """
        # Resize and Pad the output 3d volume to its original dimension 
        # If the ROI extraction resized the image then upsample to the original shape
        boResized = self.input_img['resized']
        if boResized:
            # Upsample the image to max_size = (256, 256)
            seg_resized = np.zeros((seg_post_3d.shape[0], max_size[0], max_size[1]), dtype=np.uint8)
            for slice in range(seg_post_3d.shape[0]):
                seg_resized[slice] = resize_sitk_2D(seg_post_3d[slice], max_size, interpolator=sitk.sitkNearestNeighbor) 
            seg_post_3d = seg_resized

        if 'pad_patch' in postProcessList:
            seg_post_3d = pad_3Dpatch(seg_post_3d, self.input_img['pad_params'])
        # print (seg_post_3d.shape)
        return swapaxes_to_xyz(seg_post_3d)

    def CalDice(self, pred, gt, class_lbl=1):
        """
        Calculate dice score
        """
        # intersection = np.sum((pred == class_lbl) & (gt == class_lbl))
        # dice_4d = 2.0 *(intersection)/(np.sum(pred == class_lbl) + np.sum(gt == class_lbl))
        dices = []
        for i in range(pred.shape[3]):
            labelPred=sitk.GetImageFromArray(pred[:,:,:,i], isVector=False)
            labelTrue=sitk.GetImageFromArray(gt[:,:,:,i], isVector=False)
            dicecomputer=sitk.LabelOverlapMeasuresImageFilter()
            dicecomputer.Execute(labelTrue==class_lbl,labelPred==class_lbl)
            dice=dicecomputer.GetDiceCoefficient()
            dices.append(dice)
        # print (np.mean(dices), dice_4d)
        return np.mean(dices)

    def Predict(self, input_vol, outputs_shape = None, postProcessList = [], crf = None, roi_mask_path = None):
        """
        Infer phase-wise- 3D batches
        """
        x_dim = input_vol.shape[0]
        y_dim = input_vol.shape[1]
        n_slices = input_vol.shape[2]
        n_phases = input_vol.shape[3]
        n_class = self.conf.num_class
        n_channel = self.conf.num_channels
        n_batch = self.conf.prediction_batch_size
        in_shape = self.input_img['shape']
        seg_4d_post = np.zeros(in_shape, dtype=np.uint8)

        # Do inference phase-wise for 4-D data
        for phase in range(n_phases):
            # Intialization
            s_time = time.time()
            output_volume = np.zeros([n_slices, x_dim, y_dim])
            out_posteriors = np.zeros([n_slices, x_dim, y_dim, n_class])
            img_batch = np.zeros([n_batch, x_dim, y_dim, n_channel])
            from_idx = 0
            while True:
                to_idx = from_idx + n_batch
                if not to_idx > n_slices:
                    for i in range(n_batch):
                        img_batch[i,:,:,0] = input_vol[:,:,from_idx + i, phase]
                    outputs, posteriors = self.sess.run([self.model.predictions, self.model.posteriors], 
                        feed_dict ={self.model.inputs: img_batch, self.model.is_training:False})
                    # print (img_batch.shape, outputs.shape)
                    output_volume[from_idx:to_idx] = outputs
                    out_posteriors[from_idx:to_idx] = posteriors
                    from_idx = to_idx
                else:
                    if from_idx < n_slices:
                        img_batch = np.zeros([n_slices-from_idx, x_dim, y_dim, n_channel])
                        for i in range(n_slices-from_idx):
                            img_batch[i,:,:,0] = input_vol[:,:,from_idx + i, phase]
                        outputs, posteriors = self.sess.run([self.model.predictions, self.model.posteriors],
                            feed_dict ={self.model.inputs: img_batch, self.model.is_training:False})
                        # print (img_batch.shape, outputs.shape)
                        output_volume[from_idx:] = outputs
                        out_posteriors[from_idx:] = posteriors
                        from_idx = n_slices
                    else:
                        break

            output_sitk_img = sitk.GetImageFromArray(np.uint8(output_volume))
            output_sitk_img = self.PostProcessVolume(output_sitk_img, postProcessList, roi_mask_path)

            # if crf is not None:
            #   output_sitk_img = doCRF(output_sitk_img, out_posteriors)

            # Resize and Pad the output 3d volume to its original dimension 
            seg_post_3d = sitk.GetArrayFromImage(output_sitk_img)
            seg_post_3d = self.PostPadding(seg_post_3d, postProcessList)
            # print (seg_post_3d.shape)
            seg_4d_post[:,:,:,phase] = seg_post_3d
            # plt.imshow(seg_4d_post[:,:,5,phase],cmap='gray')
            # plt.savefig(str(phase)+'.png')            
            progress_bar(phase%n_phases+1, n_phases,time.time()-s_time)
        return seg_4d_post

    def CheckSizeAndSaveSlice(self, seg_4D, img_files_path_list, seg_files_path_list, 
                            save_path, extension='.png', scale=1, class_lbl=None, enable_IPG=False):
        """
        TODO:
        """
        folder_path = os.path.dirname(img_files_path_list[0])
        print (folder_path)
        file_names = [os.path.basename(file_path) for file_path in img_files_path_list]     
        sorted_file_names = sorted(file_names, 
            key=lambda x: tuple(int(i) for i in re.findall('\d+', x)[1:]))
        sorted_file_path = [os.path.join(folder_path, file) for file in sorted_file_names]
        x_dim, y_dim, n_slice, n_phase = seg_4D.shape
        iter_cnt = 0
        for slice in range(n_slice):
            s_time = time.time()
            for phase in range(n_phase):
                seg_2D = np.uint8(seg_4D[:,:,slice,phase])
                img = sitk.ReadImage(sorted_file_path[iter_cnt])
                img_array = sitk.GetArrayFromImage(img)
                if img_array[0].shape != (x_dim, y_dim):
                    # print (img_array[0].shape, (x_dim, y_dim))
                    # print ('Shapes of DCM & Pred not matching: Resizing to DCM')
                    seg_2D = resize_image_with_crop_or_pad(seg_2D,
                             img_array[0].shape[0], img_array[0].shape[1], pad_mode='constant')

                if seg_files_path_list or enable_IPG:
                    plt.figure()
                    plt.subplot(1,3,1)
                    plt.title('Input_SA_{}_ph{}'.format(slice, phase))
                    plt.imshow(img_array[0], cmap='gray')
                    # For plotting ROI center, max_radii and extracted patch
                    roi_center = self.input_img.get ('roi_center', None) 
                    if roi_center:
                        roi_max_radius = self.input_img['roi_max_radius']
                        plt.plot([roi_center[1]], [roi_center[0]], marker='+', markersize=6, color="red")
                        circle =plt.Circle((roi_center[1], roi_center[0]), roi_max_radius, color='r', fill=False)
                        ax = plt.gca()
                        ax.add_artist(circle)
                        # Create a Rectangle patch of size 128*128
                        rect = patches.Rectangle((roi_center[1]-64, roi_center[0]-64), 128,128,linewidth=1,edgecolor='b',facecolor='none')
                        # Add the patch to the Axes
                        ax.add_patch(rect)

                    plt.subplot(1,3,2)
                    plt.title('Pred_SA_{}_ph{}'.format(slice, phase))
                    plt.imshow(seg_2D, cmap='gray')
                    plt.subplot(1,3,3)
                    plt.title('GT_SA_{}_ph{}'.format(slice, phase))
                    gt_img_array = np.zeros_like(seg_2D)

                    gt_path = os.path.join(folder_path, 
                            sorted_file_names[iter_cnt].split('.')[0]+extension)
                    gt_img = sitk.ReadImage(gt_path)
                    gt_img_array = sitk.GetArrayFromImage(gt_img)
                    gt_img_array = gt_img_array[:,:,0] 
                    # gt_img_array[np.where(gt_img_array>0)] = 1
                    # print (np.unique(seg_2D))

                    plt.imshow(gt_img_array, cmap='gray')

                    if not os.path.exists(save_path+'/IPG'):
                        os.makedirs(save_path+'/IPG')
                    plt.savefig(os.path.join(save_path, 'IPG',
                                'IPG_'+sorted_file_names[iter_cnt].split('.')[0]+extension))
                    plt.close('all')
                if class_lbl:
                    seg_2D = (seg_2D==class_lbl)
                    seg_2D = np.uint8(seg_2D)
                sitk_img = sitk.GetImageFromArray(seg_2D*scale)
                sitk.WriteImage(sitk_img, os.path.join(save_path, 
                            sorted_file_names[iter_cnt].split('.')[0]+extension))
                iter_cnt+=1
            utils.progress(slice%n_slice+1, n_slice, time.time()-s_time)
                # break
        if seg_files_path_list:     
            pred_file_path_list = glob.glob(save_path + "/*_SA*_ph*.png")
            calcSegmentationMetrics(pred_file_path_list, seg_files_path_list, class_label=class_lbl,
                extension=extension, result_path = save_path+'/IPG')    
        return  

    def CheckSizeAndSaveVolume(self, seg_4D, patient_data, save_path):
        """
        TODO:
        """ 
        prefix = patient_data['pid']
        suffix = '4D'
        seg_4D = np.swapaxes(seg_4D, 0, 1)
        save_nii(seg_4D, patient_data['4D_affine'], patient_data['4D_hdr'], save_path, prefix, suffix)
        suffix = 'ED'
        ED_phase_n = int(patient_data['ED'])
        ED_pred = seg_4D[:,:,:,ED_phase_n]
        save_nii(ED_pred, patient_data['3D_affine'], patient_data['3D_hdr'], save_path, prefix, suffix)

        suffix = 'ES'
        ES_phase_n = int(patient_data['ES'])
        ES_pred = seg_4D[:,:,:,ES_phase_n]
        save_nii(ES_pred, patient_data['3D_affine'], patient_data['3D_hdr'], save_path, prefix, suffix)

        ED_GT = patient_data.get('ED_GT', None)
        results = []
        if ED_GT is not None:
            quality = computeQualityMeasures(ED_pred.T, ED_GT.T, class_label=1)     
            results.extend(quality.values())    
            quality = computeQualityMeasures(ED_pred.T, ED_GT.T, class_label=2)
            results.extend(quality.values())
            quality = computeQualityMeasures(ED_pred.T, ED_GT.T, class_label=3)
            results.extend(quality.values())
        ES_GT = patient_data.get('ES_GT', None)
        if ES_GT is not None:
            quality = computeQualityMeasures(ES_pred.T, ES_GT.T, class_label=1)
            results.extend(quality.values())
            quality = computeQualityMeasures(ES_pred.T, ES_GT.T, class_label=2)
            results.extend(quality.values())
            quality = computeQualityMeasures(ES_pred.T, ES_GT.T, class_label=3)
            results.extend(quality.values())
        print (results)
        return results
        
    def LoadAndPredict(self, patient_data=None, img_files_path_list=None, seg_files_path_list=None, roi_mask_path = None, 
                        patch_size=(128, 128), outputs_shape = None, preProcessList=[], 
                        postProcessList = [], crf=None, save_path = None):
        """
        TODO: Generic Prediction pipeline 
        """
        print("loading image and pre-processing ...")
        i_time = time.time()
        if img_files_path_list:
            # LV-2011 PipeLine
            input_prep_4D = self.LoadAndPreprocessSlice(img_files_path_list, preProcessList, patch_size)
        if  patient_data:
            # ACDC Pipeline
            input_prep_4D = self.LoadAndPreprocessVolume(patient_data, preProcessList, patch_size)
        print("Time taken for Pre-processing: " + str(time.time()-i_time) + 's')
        print("\n")
        # print (input_prep_4D.shape)

        i_time = time.time()
        pred_4d = self.Predict(input_prep_4D, outputs_shape, postProcessList, crf, roi_mask_path)
        print("Time taken for Inference and post-processing: " + str(time.time()-i_time) + 's')
        print("\n")
        # print (pred_4d.shape)
        if seg_files_path_list:
            # print("loading segmentation and computing Quality metrics...")
            label_myo = 1
            gt_4d = get_4D_volume(seg_files_path_list, gt=True, gt_shape=pred_4d.shape)
            dice = self.CalDice(np.uint8(pred_4d==label_myo), gt_4d)
            print("Dice for MYO: " + os.path.basename(save_path) + " : " + str(dice))

        if not save_path is None:
            if not img_files_path_list is None:
                print("Saving Predictions in PNG format ...")
                # print (pred_4d.shape)
                self.CheckSizeAndSaveSlice(pred_4d, img_files_path_list, seg_files_path_list, save_path, scale=1, class_lbl=label_myo)
            elif not patient_data is None:
                print ("Saving predictions in Nii format")
                # print (pred_4d.shape)
                results = self.CheckSizeAndSaveVolume(pred_4d, patient_data, save_path)         
        if seg_files_path_list:
            return dice
        else:
            return results




def save_nii(vol, affine, hdr, path, prefix, suffix):
    vol = nib.Nifti1Image(vol, affine, hdr)
    vol.set_data_dtype(np.uint8)
    nib.save(vol, os.path.join(path, prefix+'_'+suffix))

def load_nii(img_file, folder_path):
    """
    Function to load a 'nii' or 'nii.gz' file, The function returns
    everyting needed to save another 'nii' or 'nii.gz'
    in the same dimensional space, i.e. the affine matrix and the header
    """
    nimg = nib.load(os.path.join(folder_path, img_file))
    return nimg.get_data(), nimg.affine, nimg.header

def get_test_data_acdc(src_path, patient, gt_available=False):
    """
    Segregate the splits into LA and SA views and dump all files in a common folder
    """
    gt_file_path_list = None

    patient_data = {}
    folder_path = os.path.join(src_path, patient)
    config_file_path = os.path.join(folder_path, 'Info.cfg')

    with open(config_file_path) as f_in:
        for line in f_in:
            l = line.rstrip().split(": ")
            patient_data[l[0]] = l[1]

    # Read patient Number
    m = re.match("patient(\d{3})", patient)
    patient_No = int(m.group(1))
    # Read Diastole frame Number
    ED_frame_No = int(patient_data['ED'])
    ed_img = "patient%03d_frame%02d.nii.gz" %(patient_No, ED_frame_No)
    ed, affine, hdr  = load_nii(ed_img, folder_path)
    # Read Systole frame Number
    ES_frame_No = int(patient_data['ES'])
    es_img = "patient%03d_frame%02d.nii.gz" %(patient_No, ES_frame_No)
    es, _, _  = load_nii(es_img, folder_path)

    img_4d = "patient%03d_4d.nii.gz" %(patient_No)

    img_4d_data, affine_4d, hdr_4d  = load_nii(img_4d, folder_path)
    patient_data['pid'] = patient
    patient_data['4D'] = np.swapaxes(img_4d_data, 0, 1)
    patient_data['4D_affine'] = affine_4d
    patient_data['4D_hdr'] = hdr_4d

    patient_data['ED_VOL'] = ed
    patient_data['ES_VOL'] = es
    patient_data['3D_affine'] = affine
    patient_data['3D_hdr'] = hdr


    if gt_available:
        ed_gt, _, _  = load_nii("patient%03d_frame%02d_gt.nii.gz" %(patient_No, ED_frame_No), folder_path)
        es_gt, _, _  = load_nii("patient%03d_frame%02d_gt.nii.gz" %(patient_No, ES_frame_No), folder_path)
        patient_data['ED_GT'] = ed_gt
        patient_data['ES_GT'] = es_gt

    return patient_data 

def calcSegmentationMetrics(pred_files_path_list, seg_files_path_list, class_label=1, extension='.png', result_path = './'):

    """
    TODO:
    """
    #Metric Dictionary 
    HEADER = ["Name", "dice", "jaccard", "Hausdorff"]
    metrics = {}

    # Sort files
    folder_path = os.path.dirname(pred_files_path_list[0])
    # print (folder_path)
    file_names = [os.path.basename(file_path) for file_path in pred_files_path_list]    
    sorted_pred_file_names = sorted(file_names, 
        key=lambda x: tuple(int(i) for i in re.findall('\d+', x)[1:]))
    sorted_pred_file_path_list = [os.path.join(folder_path, file) for file in sorted_pred_file_names]

    folder_path = os.path.dirname(seg_files_path_list[0])
    file_names = [os.path.basename(file_path) for file_path in seg_files_path_list]     
    sorted_seg_file_names = sorted(file_names, 
        key=lambda x: tuple(int(i) for i in re.findall('\d+', x)[1:]))
    sorted_seg_file_path_list = [os.path.join(folder_path, file) for file in sorted_seg_file_names]


    # Read the files
    res = []
    for pred_file, seg_file in zip(sorted_pred_file_path_list, sorted_seg_file_path_list):
        if os.path.basename(pred_file) != os.path.basename(seg_file):
            raise ValueError("The two files don't have the same name"
                             " {}, {}.".format(os.path.basename(pred_file),
                                               os.path.basename(seg_file)))
        # Reag Prediction Groundtruth
        if extension =='.png':
            pred = sitk.ReadImage(pred_file)
            pred_array = sitk.GetArrayFromImage(pred)
            # print (pred_array.shape)
            # seg_array = pred_array[:,:,0] 
            # pred_array[np.where(pred_array>0)] = 1
            seg = sitk.ReadImage(seg_file)

            seg_array = sitk.GetArrayFromImage(seg)
            # print (seg_array.shape)
            seg_array = seg_array[:,:,0] 
            seg_array[np.where(seg_array>0)] = 1
        else:
            print ("TODO")

        quality = computeQualityMeasures(pred_array, seg_array, class_label)
        # print (os.path.basename(pred_file), quality.values())
        res.append(quality.values())
    lst_name_gt = [os.path.basename(gt).split(".")[0] for gt in sorted_seg_file_names]
    result = [[n,] + r for r, n in zip(res, lst_name_gt)]
    df = pd.DataFrame(result, columns=HEADER)
    # TODO: Compute statistics 

    # Select the features:
    START_COL = 1
    END_COL = 3
    features = list(df.columns[np.r_[START_COL:END_COL]])
    # print ('Statistics\n')
    # print ('\nAverage\n')
    # print (np.mean(df[features]))
    # print ('\nBest\n')
    # print (np.max(df[features]))
    # print ('\nWorst\n')
    # print (np.min(df[features]))
    # TODO:
    # df.append(pd.DataFrame(np.mean(df[features]), columns=features), ignore_index=True)
    # df.append(pd.DataFrame(np.max(df[features]), columns=features), ignore_index=True)
    # df.append(pd.DataFrame(np.min(df[features]), columns=features), ignore_index=True)
    df.to_csv(os.path.join(result_path, "results_{}.csv".format(time.strftime("%Y%m%d_%H%M%S"))), index=False)

# https://programtalk.com/vs2/?source=python/4202/VNet/VNet.py
def computeQualityMeasures(lP, lT, class_label):
    # Get a SimpleITK Image from a numpy array. 
    # If isVector is True, then a 3D array will be treaded 
    # as a 2D vector image, otherwise it will be treaded as a 3D image

    quality=OrderedDict()
    labelPred=sitk.GetImageFromArray(lP, isVector=False)
    labelTrue=sitk.GetImageFromArray(lT, isVector=False)

    dicecomputer=sitk.LabelOverlapMeasuresImageFilter()
    dicecomputer.Execute(labelTrue==class_label,labelPred==class_label)
    quality["dice"]=dicecomputer.GetDiceCoefficient()
    quality["jaccard"]=dicecomputer.GetJaccardCoefficient()
    #    quality["dice"]=dicecomputer.GetMeanOverlap()
    # quality["dice"]=dicecomputer.GetVolumeSimilarity() 
    # quality["dice"]=dicecomputer.GetUnionOverlap()
    # quality["dice"]=dicecomputer.GetFalseNegativeError() 
    #    quality["dice"]=dicecomputer.GetFalsePositiveError () 
    # Check if both the images have non-zero pixel count?
    # Else it will throw error. Just set 0 distance if pixel count =0
    quality["avgHausdorff"]=0
    quality["Hausdorff"]=0
    # Disable Hausdorff: Takes long time to compute 
    # # READ:
    # # https://itk.org/Doxygen/html/classitk_1_1DirectedHausdorffDistanceImageFilter.html
    # # https://itk.org/SimpleITKDoxygen/html/classitk_1_1simple_1_1HausdorffDistanceImageFilter.html#a0bc838ff0d5624132abdbe089eb54705
    try:        
        if (np.count_nonzero(labelTrue) and np.count_nonzero(labelPred)):
            hausdorffcomputer=sitk.HausdorffDistanceImageFilter()
            hausdorffcomputer.Execute(labelTrue==class_label,labelPred==class_label)
            quality["avgHausdorff"]=hausdorffcomputer.GetAverageHausdorffDistance()
            quality["Hausdorff"]=hausdorffcomputer.GetHausdorffDistance()
    except Exception:
        pass
    return quality
 

def hist_match(source, template):
    """
    Adjust the pixel values of a grayscale image such that its histogram
    matches that of a target image
 
    Arguments:
    -----------
        source: np.ndarray
            Image to transform; the histogram is computed over the flattened
            array
        template: np.ndarray
            Template image; can have different dimensions to source
    Returns:
    -----------
        matched: np.ndarray
            The transformed output image
    """
 
    oldshape = source.shape
    source = source.ravel()
    template = template.ravel()
 
    # get the set of unique pixel values and their corresponding indices and
    # counts
    s_values, bin_idx, s_counts = np.unique(source, return_inverse=True,
                                            return_counts=True)
    t_values, t_counts = np.unique(template, return_counts=True)
 
    # take the cumsum of the counts and normalize by the number of pixels to
    # get the empirical cumulative distribution functions for the source and
    # template images (maps pixel value --> quantile)
    s_quantiles = np.cumsum(s_counts).astype(np.float64)
    s_quantiles /= s_quantiles[-1]
    t_quantiles = np.cumsum(t_counts).astype(np.float64)
    t_quantiles /= t_quantiles[-1]
 
    # interpolate linearly to find the pixel values in the template image
    # that correspond most closely to the quantiles in the source image
    #interp_t_values = np.zeros_like(source,dtype=float)
    interp_t_values = np.interp(s_quantiles, t_quantiles, t_values)
 
    return interp_t_values[bin_idx].reshape(oldshape)
 
def sitk_show(nda, title=None, margin=0.0, dpi=40):
    figsize = (1 + margin) * nda.shape[0] / dpi, (1 + margin) * nda.shape[1] / dpi
 
    extent = (0, nda.shape[1], nda.shape[0], 0)
    fig = plt.figure(figsize=figsize, dpi=dpi)
    ax = fig.add_axes([margin, margin, 1 - 2*margin, 1 - 2*margin])
 
    plt.set_cmap("gray")
    for k in range(0,nda.shape[2]):
        print ("printing slice "+str(k))
        ax.imshow(np.squeeze(nda[:,:,k]),extent=extent,interpolation=None)
        plt.draw()
        plt.pause(0.1)
        #plt.waitforbuttonpress()

if __name__ == '__main__':
    pass
    # pred_path = 'path/to_pred_folder'
    # seg_path  = 'path/to_ground_truth'
    # pred_files_path_list =  glob.glob(pred_path + "/*_SA*_ph*.png")
    # seg_files_path_list = glob.glob(seg_path + "/*_SA*_ph*.png")
    # calcSegmentationMetrics(pred_files_path_list, seg_files_path_list, extension='.png', result_path = './')