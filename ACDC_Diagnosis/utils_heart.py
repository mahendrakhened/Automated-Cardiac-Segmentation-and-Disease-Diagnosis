import os
import time
import nibabel as nib
import pandas as pd
import numpy as np
import scipy
from scipy import ndimage
import matplotlib
# matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import skimage
from skimage import feature
from scipy import spatial
from helpers import utils
#from loaders import data_augmentation

def heart_metrics(seg_3Dmap, voxel_size, classes=[3, 1, 2]):
    """
    Compute the volumes of each classes
    """
    # Loop on each classes of the input images
    volumes = []
    for c in classes:
        # Copy the gt image to not alterate the input
        seg_3Dmap_copy = np.copy(seg_3Dmap)
        seg_3Dmap_copy[seg_3Dmap_copy != c] = 0

        # Clip the value to compute the volumes
        seg_3Dmap_copy = np.clip(seg_3Dmap_copy, 0, 1)

        # Compute volume
        # volume = seg_3Dmap_copy.sum() * np.prod(voxel_size) / 1000.
        volume = seg_3Dmap_copy.sum() * np.prod(voxel_size)
        volumes += [volume]
    return volumes

def ejection_fraction(ed_vol, es_vol):
    """
    Calculate ejection fraction
    """
    stroke_vol = ed_vol - es_vol
    return (np.float(stroke_vol)/np.float(ed_vol))

def myocardialmass(myocardvol):
    """
    Specific gravity of heart muscle (1.05 g/ml)
    """ 
    return myocardvol*1.05

def bsa(height, weight):
    """
    Body surface Area
    """
    return np.sqrt((height*weight)/3600)

def myocardial_thickness(data_path, slices_to_skip=(0,0), myo_label=2):
    """
    Calculate myocardial thickness of mid-slices, excluding a few apex and basal slices
    since myocardium is difficult to identify
    """
    label_obj = nib.load(data_path)
    myocardial_mask = (label_obj.get_data()==myo_label)
    # pixel spacing in X and Y
    pixel_spacing = label_obj.header.get_zooms()[:2]
    assert pixel_spacing[0] == pixel_spacing[1]

    holes_filles = np.zeros(myocardial_mask.shape)
    interior_circle = np.zeros(myocardial_mask.shape)

    cinterior_circle_edge=np.zeros(myocardial_mask.shape)
    cexterior_circle_edge=np.zeros(myocardial_mask.shape)

    overall_avg_thickness= []
    overall_std_thickness= []
    for i in xrange(slices_to_skip[0], myocardial_mask.shape[2]-slices_to_skip[1]):
        holes_filles[:,:,i] = ndimage.morphology.binary_fill_holes(myocardial_mask[:,:,i])
        interior_circle[:,:,i] = holes_filles[:,:,i] - myocardial_mask[:,:,i]
        cinterior_circle_edge[:,:,i] = feature.canny(interior_circle[:,:,i])
        cexterior_circle_edge[:,:,i] = feature.canny(holes_filles[:,:,i])
        # patch = 64
        # utils.imshow(data_augmentation.resize_image_with_crop_or_pad(myocardial_mask[:,:,i], patch, patch), 
        #     data_augmentation.resize_image_with_crop_or_pad(holes_filles[:,:,i], patch, patch),
        #     data_augmentation.resize_image_with_crop_or_pad(interior_circle[:,:,i], patch,patch ), 
        #     data_augmentation.resize_image_with_crop_or_pad(cinterior_circle_edge[:,:,i], patch, patch), 
        #     data_augmentation.resize_image_with_crop_or_pad(cexterior_circle_edge[:,:,i], patch, patch), 
        #     title= ['Myocardium', 'Binary Hole Filling', 'Left Ventricle Cavity', 'Interior Contour', 'Exterior Contour'], axis_off=True)
        x_in, y_in = np.where(cinterior_circle_edge[:,:,i] != 0)
        number_of_interior_points = len(x_in)
        # print (len(x_in))
        x_ex,y_ex=np.where(cexterior_circle_edge[:,:,i] != 0)
        number_of_exterior_points=len(x_ex)
        # print (len(x_ex))
        if len(x_ex) and len(x_in) !=0:
            total_distance_in_slice=[]
            for z in xrange(number_of_interior_points):
                distance=[]
                for k in xrange(number_of_exterior_points):
                    a  = [x_in[z], y_in[z]]
                    a=np.array(a)
                    # print a
                    b  = [x_ex[k], y_ex[k]]
                    b=np.array(b)
                    # dst = np.linalg.norm(a-b)
                    dst = scipy.spatial.distance.euclidean(a, b)
                    # pdb.set_trace()
                    # if dst == 0:
                    #     pdb.set_trace()
                    distance = np.append(distance, dst)
                distance = np.array(distance)
                min_dist = np.min(distance)
                total_distance_in_slice = np.append(total_distance_in_slice,min_dist)
                total_distance_in_slice = np.array(total_distance_in_slice)

            average_distance_in_slice = np.mean(total_distance_in_slice)*pixel_spacing[0]
            overall_avg_thickness = np.append(overall_avg_thickness, average_distance_in_slice)

            std_distance_in_slice = np.std(total_distance_in_slice)*pixel_spacing[0]
            overall_std_thickness = np.append(overall_std_thickness, std_distance_in_slice)

    # print (overall_avg_thickness)
    # print (overall_std_thickness)
    # print (pixel_spacing[0])
    return (overall_avg_thickness, overall_std_thickness)