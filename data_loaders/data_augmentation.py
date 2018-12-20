# Imports
from __future__ import division
import numpy as np
import h5py, sys
from collections import namedtuple
import skimage.morphology as morph
import skimage.transform
import skimage.draw
import skimage.morphology as morph
from skimage.feature import canny
import matplotlib.pyplot as plt
import cv2
from cv2 import bilateralFilter
from skimage.restoration import denoise_tv_chambolle
from skimage.restoration import denoise_tv_bregman
import SimpleITK as sitk
import scipy.ndimage as snd

sys.path.append("../")
from helpers.utils import imshow
rng = np.random.RandomState(40)

# ********************** Weight Map Generation and mini-Batch Class weights***********#
selem = morph.disk(1)
def getEdgeEnhancedWeightMap(label, label_ids =[0,1,2,3], scale=1, edgescale=1, assign_equal_wt=False):
    shape = (0,)+label.shape[1:]
    weight_map = np.empty(shape, dtype='uint8')
    if assign_equal_wt:
        return np.ones_like(label)
    for i in range(label.shape[0]): 
        #Estimate weight maps:
        weights = np.ones(len(label_ids))
        slice_map = np.ones(label[i,:,:].shape)
        for _id in label_ids:
            class_frequency = np.sum(label[i,:,:] == label_ids[_id])
            if class_frequency:
                weights[label_ids.index(_id)] = scale*label[i,:,:].size/class_frequency
                slice_map[np.where(label[i,:,:]==label_ids.index(_id))] = weights[label_ids.index(_id)]
                edge = np.float32(morph.binary_dilation(
                    canny(np.float32(label[i,:,:]==label_ids.index(_id)),sigma=1), selem=selem))
                edge_frequency = np.sum(np.sum(edge==1.0))
                if edge_frequency:    
                    slice_map[np.where(edge==1.0)] += edgescale*label[i,:,:].size/edge_frequency
            # print (weights)
            # utils.imshow(edge, cmap='gray')
        # utils.imshow(weight_map, cmap='gray')
        weight_map = np.append(weight_map, np.expand_dims(slice_map, axis=0), axis=0)
    return np.float32(weight_map)


def GetAvgbatchClassWeights(label, scale=1, label_ids=[0,1], assign_equal_wt=False):
    """
    This function calulates the class weights for a batch of data
    Args:
    label: [batch_size,H,W]
    return:
    [class1_weight, ..., class2_weight] 
    """
    batch_size = label.shape[0]
    batch_weights = np.zeros((batch_size, len(label_ids)))
    if assign_equal_wt:
        return np.ones(len(label_ids), dtype=np.uint8)
    pixel_cnt = label[0,:,:].size
    eps = 0.001
    for i in range(batch_size): 
        for _id in label_ids:
            batch_weights[i, label_ids.index(_id)] = scale*pixel_cnt/np.float(np.sum(label[i,:,:] == label_ids[_id])+eps)
            # print (np.uint8(np.mean(batch_weights+1, axis=0)))
    return np.float32(np.mean(batch_weights+1, axis=0)) 

#**************************************Data Preprocessing functions********************
def PreProcessData(file_name, data, mode, transformation_params, Alternate=True):
    """
    Preprocess the image, ground truth (label) and return  along with its corresponding weight map
    """
    image = data['image'][:]
    label = data['label'][:] 
    roi = data['roi_center'][:]
    roi_radii = data['roi_radii'][:]
    pixel_spacing = data['pixel_spacing'][:]
    n_labels = transformation_params['n_labels']
    max_radius = roi_radii[1]
    patch_size = transformation_params[mode]['patch_size']
    max_size = transformation_params.get('data_crop_pad', (256, 256))

    # print (image.shape, pixel_spacing)

    if transformation_params['full_image']:
        # Dont do any ROI crop or augmentation
        # Just make sure that all the images are of fixed size
        # By cropping or Padding
        if image.ndim == 2:
            image = np.expand_dims(image, axis=2)
        patch_image = resize_image_with_crop_or_pad_3D(normalize(image), max_size[0], max_size[1])
        patch_label = resize_image_with_crop_or_pad_3D(label[:,:,None], max_size[0], max_size[1])
    else: 
        # If to alternate randomly between training with and without augmentation
        if Alternate:
            boAlternate = rng.randint(2) > 0
        else:
            boAlternate = False

        if not transformation_params['data_augmentation'] or boAlternate:
            # Check if the roi fits patch_size else resize the image to patch dimension
            if CheckImageFitsInPatch(image, roi, max_radius, patch_size):
                # Center around roi
                patch_image = crop_img_patch_from_roi(normalize(image), roi, patch_size)
                patch_label = crop_img_patch_from_roi(label, roi, patch_size)
                # If patch size does not fit then pad or crop
                patch_image = resize_image_with_crop_or_pad_3D(patch_image[:,:, None], patch_size[0], patch_size[1])
                patch_label = resize_image_with_crop_or_pad_3D(patch_label[:,:, None], patch_size[0], patch_size[1])
                # print (patch_image.shape, patch_label.shape)
            else:
                patch_image  = crop_img_patch_from_roi(normalize(image), roi, max_size)
                patch_label = crop_img_patch_from_roi(label, roi, max_size)
                patch_image = resize_sitk_2D(patch_image, patch_size)
                patch_label = resize_sitk_2D(patch_label, patch_size, interpolator=sitk.sitkNearestNeighbor)
        else:
            random_params = sample_augmentation_parameters(transformation_params[mode])
            # print (random_params)
            patch_image, patch_label, _ = roi_patch_transform_norm(data, transformation_params[mode], 
                                        n_labels, random_augmentation_params=random_params,
                                        uniform_scale=False, random_denoise=False, denoise=False)

            if transformation_params['data_deformation'] and (rng.randint(2) > 0)\
            and (transformation_params[mode] != 'valid'):
                patch_image, patch_label = produceRandomlyDeformedImage(patch_image, patch_label[:,:,None])                       

    # Expand dimensions to feed to network
    if patch_image.ndim == 2:
        patch_image = np.expand_dims(patch_image, axis=2) 
    if patch_label.ndim == 3:
        patch_label = np.squeeze(patch_label, axis=2) 
    patch_image = np.expand_dims(patch_image, axis=0)
    patch_label = np.expand_dims(patch_label, axis=0)
    # print (patch_image.shape, patch_label.shape)
    # TODO: Check post nrmalization effects
    # patch_image = normalize(patch_image, scheme='zscore')
    weight_map = getEdgeEnhancedWeightMap(patch_label, label_ids=range(n_labels), scale=1, edgescale=1,  assign_equal_wt=False)
    return (patch_image, patch_label, weight_map)

# Functions
def normalize(image, scheme='zscore'):
    # Do Image Normalization:
    if scheme == 'zscore':
        image = normalize_zscore(image, z=0.5, offset=0)
    elif scheme == 'minmax':
        image = normalize_minmax(image)
    elif scheme == 'truncated_zscore':
        image = normalize_zscore(image, z=2, offset=0.5, clip=True)
    else:
        image = image
    return image

def normalize_with_given_mean_std(image, mean, std):
    # Do Image Normalization, given mean and std
    return (image - mean) / std 

def normalize_zscore(data, z=2, offset=0.5, clip=False):
    """
    Normalize contrast across volume
    """
    mean = np.mean(data)
    std = np.std(data)
    img = ((data - mean) / (2 * std * z) + offset) 
    if clip:
        # print ('Before')
        # print (np.min(img), np.max(img))
        img = np.clip(img, -0.0, 1.0)
        # print ('After clip')
        # print (np.min(img), np.max(img))
    return img

def normalize_minmax(data):
    """
    Normalize contrast across volume
    """
    _min = np.float(np.min(data))
    _max = np.float(np.max(data))
    if (_max-_min)!=0:
        img = (data - _min) / (_max-_min)
    else:
        img = np.zeros_like(data)            
    return img

def slicewise_normalization(img_data4D, scheme='minmax'):
    """
    Do slice-wise normalization for the 4D image data(3D+ Time)
    """
    x_dim, y_dim, n_slices, n_phases = img_data4D.shape

    data_4d = np.zeros([x_dim, y_dim, n_slices, n_phases])
    for slice in range(n_slices):
        for phase in range(n_phases):
            data_4d[:,:,slice, phase] = normalize(img_data4D[:,:,slice, phase], scheme)
    return data_4d

def phasewise_normalization(img_data4D, scheme='minmax'):
    """
    Do slice-wise normalization for the 4D image data(3D+ Time)
    """
    x_dim, y_dim, n_slices, n_phases = img_data4D.shape

    data_4d = np.zeros([x_dim, y_dim, n_slices, n_phases])
    for phase in range(n_phases):
        data_4d[:,:,:, phase] = normalize(img_data4D[:,:,:, phase], scheme)
    return data_4d

def CheckImageFitsInPatch(image, roi, max_radius, patch_size):
    boFlag = True
    max_radius += max_radius*.05
    if (max_radius > patch_size[0]/2) or (max_radius > patch_size[1]/2)\
        or (image.shape[0]>=512) or (image.shape[1]>=512):
        # print('The patch wont fit the roi: resize image', image.shape, max_radius, patch_size[0]/2)
        boFlag = False
    return boFlag

def swapaxes_to_xyz(vol):
    return np.swapaxes(np.swapaxes(vol,0,2),0,1)

def crop_img_patch_from_roi(image_2D, roi_center, patch_size=(128,128)):
    """
    This code extracts a patch of defined size from the given center point of the image
    and returns parameters for padding and translation to original location of the image
    Args:
    2D: Volume:  Y, X
    """

    cols, rows = image_2D.shape
    patch_cols = np.uint16(max(0, roi_center[0] - patch_size[0]/2))
    patch_rows = np.uint16(max(0, roi_center[1] - patch_size[0]/2))
    # print (patch_cols,patch_cols+patch_size[0], patch_rows,patch_rows+patch_size[0])
    patch_img = image_2D[patch_cols:patch_cols+patch_size[0], patch_rows:patch_rows+patch_size[0]]
    return patch_img

def extract_patch(image_2D, roi_center, patch_size=(128, 128)):
    """
    This code extracts a patch of defined size from the given center point of the image
    and returns parameters for padding and translation to original location of the image
    Args:
    2D: Volume: X ,Y
    """
    cols, rows = image_2D.shape

    patch_cols = np.uint16(max(0, roi_center[0] - patch_size[0]/2))
    patch_rows = np.uint16(max(0, roi_center[1] - patch_size[0]/2))
    patch_img = image_2D[patch_cols:patch_cols+patch_size[0], patch_rows:patch_rows+patch_size[0]]
    if patch_img.shape != patch_size:
        # Pad the image with appropriately to patch_size
        print ('Not supported yet: Patch_size is bigger than the input image')
    # patch_img = np.expand_dims(patch_img, axis=1)
    pad_params = {'rows': rows, 'cols': cols, 'tx': patch_rows, 'ty': patch_cols}
    return patch_img, pad_params

def pad_3Dpatch(patch_3D, pad_params):
    """
    This code does padding and translation to original location of the image
    Args:
    3D: Volume: Batch_size, X, Y
    Used for predicted ground truth
    """
    if patch_3D.dtype != 'float32':
        dtype = 'uint8'
    else:
        dtype = 'float32'
    patch_3D = patch_3D.astype(dtype)
    M = np.float32([[1,0, pad_params['tx']],[0, 1, pad_params['ty']]])
    padded_patch = np.empty(((0,)+(pad_params['cols'], pad_params['rows'])), dtype=np.float32)

    for i in range(patch_3D.shape[0]):
        # import pdb; pdb.set_trace()
        patch = cv2.warpAffine(patch_3D[i],M,(pad_params['rows'], pad_params['cols']))
        patch = np.expand_dims(patch, axis=0)
        padded_patch = np.append(padded_patch, patch, axis=0)
    return padded_patch.astype(dtype)

def resize_sitk_2D(image_array, outputSize=None, interpolator=sitk.sitkLinear):
    """
    Resample 2D images Image:
    For Labels use nearest neighbour
    For image use 
    sitkNearestNeighbor = 1,
    sitkLinear = 2,
    sitkBSpline = 3,
    sitkGaussian = 4,
    sitkLabelGaussian = 5, 
    """
    image = sitk.GetImageFromArray(image_array) 
    inputSize = image.GetSize()
    inputSpacing = image.GetSpacing()
    outputSpacing = [1.0, 1.0]
    if outputSize:
        outputSpacing[0] = inputSpacing[0] * (inputSize[0] /outputSize[0]);
        outputSpacing[1] = inputSpacing[1] * (inputSize[1] / outputSize[1]);
    else:
        # If No outputSize is specified then resample to 1mm spacing
        outputSize = [0.0, 0.0]
        outputSize[0] = int(inputSize[0] * inputSpacing[0] / outputSpacing[0] + .5)
        outputSize[1] = int(inputSize[1] * inputSpacing[1] / outputSpacing[1] + .5)
    resampler = sitk.ResampleImageFilter()
    resampler.SetSize(outputSize)
    resampler.SetOutputSpacing(outputSpacing)
    resampler.SetOutputOrigin(image.GetOrigin())
    resampler.SetOutputDirection(image.GetDirection())
    resampler.SetInterpolator(interpolator)
    resampler.SetDefaultPixelValue(0)
    image = resampler.Execute(image)
    resampled_arr = sitk.GetArrayFromImage(image)
    return resampled_arr

def produceRandomlyDeformedImage(image, label, numcontrolpoints=2, stdDef=15):
    sitkImage=sitk.GetImageFromArray(image, isVector=False)
    sitklabel=sitk.GetImageFromArray(label, isVector=False)

    transfromDomainMeshSize=[numcontrolpoints]*sitkImage.GetDimension()

    tx = sitk.BSplineTransformInitializer(sitkImage,transfromDomainMeshSize)


    params = tx.GetParameters()

    paramsNp=np.asarray(params,dtype=float)
    paramsNp = paramsNp + np.random.randn(paramsNp.shape[0])*stdDef
    #remove z deformations! The resolution in z is too bad in case of 3D or its channels in 2D
    paramsNp[0:int(len(params)/3)]=0 #remove z deformations! The resolution in z is too bad in case of 3D or its channels

    params=tuple(paramsNp)
    tx.SetParameters(params)
    # print (sitkImage.GetSize(), sitklabel.GetSize(), transfromDomainMeshSize, paramsNp.shape)

    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(sitkImage)
    resampler.SetInterpolator(sitk.sitkLinear)
    resampler.SetDefaultPixelValue(0)
    resampler.SetTransform(tx)

    outimgsitk = resampler.Execute(sitkImage)

    # For Label use nearest neighbour
    resampler.SetReferenceImage(sitklabel)
    resampler.SetInterpolator(sitk.sitkLabelGaussian)
    resampler.SetDefaultPixelValue(0)
    outlabsitk = resampler.Execute(sitklabel)

    outimg = sitk.GetArrayFromImage(outimgsitk)
    outimg = outimg.astype(dtype=np.float32)

    outlbl = sitk.GetArrayFromImage(outlabsitk)
    outlbl = (outlbl).astype(dtype=np.uint8)
    return outimg, outlbl

# ********************************Augmentation Transforms**************************#
def sample_augmentation_parameters(transformation):
    # This code does random sampling from the transformation parameters
    # Random number generator
    if set(transformation.keys()) == {'patch_size', 'mm_patch_size'} or \
                    set(transformation.keys()) == {'patch_size', 'mm_patch_size', 'mask_roi'}:
        return None

    shift_x = rng.uniform(*transformation.get('translation_range_x', [0., 0.]))
    shift_y = rng.uniform(*transformation.get('translation_range_y', [0., 0.]))
    translation = (shift_x, shift_y)
    rotation = rng.uniform(*transformation.get('rotation_range', [0., 0.]))
    shear = rng.uniform(*transformation.get('shear_range', [0., 0.]))
    roi_scale = rng.uniform(*transformation.get('roi_scale_range', [1., 1.]))
    z = rng.uniform(*transformation.get('zoom_range', [1., 1.]))
    zoom = (z, z)

    if 'do_flip' in transformation:
        if type(transformation['do_flip']) == tuple:
            flip_x = rng.randint(2) > 0 if transformation['do_flip'][0] else False
            flip_y = rng.randint(2) > 0 if transformation['do_flip'][1] else False
        else:
            flip_x = rng.randint(2) > 0 if transformation['do_flip'] else False
            flip_y = False
    else:
        flip_x, flip_y = False, False

    sequence_shift = rng.randint(30) if transformation.get('sequence_shift', False) else 0

    return namedtuple('Params', ['translation', 'rotation', 'shear', 'zoom',
                                 'roi_scale',
                                 'flip_x', 'flip_y',
                                 'sequence_shift'])(translation, rotation, shear, zoom,
                                                    roi_scale,
                                                    flip_x, flip_y,
                                                    sequence_shift)

def roi_patch_transform_norm(data, transformation, nlabel, random_augmentation_params=None,
                             mm_center_location=(.5, .4), mm_patch_size=(128, 128), mask_roi=False,
                             uniform_scale=False, random_denoise=False, denoise=False, ACDC=True):

    # Input data dimension is of shape: (X,Y)
    add_noise = transformation.get('add_noise', None)
    patch_size = transformation['patch_size']
    mm_patch_size = transformation.get('mm_patch_size', mm_patch_size)
    mask_roi = transformation.get('mask_roi', mask_roi)


    image = data['image'][:]
    label = data['label'][:] 
    if image.ndim == 2:
        image = np.expand_dims(image, axis=2)
    # pixel spacing in X and Y
    pixel_spacing = data['pixel_spacing'][:]
    roi_center = data['roi_center'][:]
    roi_radii = data['roi_radii'][:]

    # Check if the roi fits patch_size else resize the image to patch dimension
    max_radius = roi_radii[1]
    if not CheckImageFitsInPatch(image, roi_center, max_radius, patch_size):
        mm_patch_size = (256, 256)


    # if random_augmentation_params=None -> sample new params
    # if the transformation implies no augmentations then random_augmentation_params remains None
    if not random_augmentation_params:
        random_augmentation_params = sample_augmentation_parameters(transformation)
        # print random_augmentation_params
    # build scaling transformation
    current_shape = image.shape[:2]

    # scale ROI radii and find ROI center in normalized patch
    if roi_center.any():
        mm_center_location = tuple(int(r * ps) for r, ps in zip(roi_center, pixel_spacing))

    # scale the images such that they all have the same scale if uniform_scale=True
    norm_rescaling = 1./ pixel_spacing[0] if uniform_scale else 1
    mm_shape = tuple(int(float(d) * ps) for d, ps in zip(current_shape, pixel_spacing))

    tform_normscale = build_rescale_transform(downscale_factor=norm_rescaling,
                                              image_shape=current_shape, target_shape=mm_shape)

    tform_shift_center, tform_shift_uncenter = build_shift_center_transform(image_shape=mm_shape,
                                                                            center_location=mm_center_location,
                                                                            patch_size=mm_patch_size)
    patch_scale = max(1. * mm_patch_size[0] / patch_size[0],
                      1. * mm_patch_size[1] / patch_size[1])
    tform_patch_scale = build_rescale_transform(patch_scale, mm_patch_size, target_shape=patch_size)

    total_tform = tform_patch_scale + tform_shift_uncenter + tform_shift_center + tform_normscale

    # build random augmentation
    if random_augmentation_params:
        augment_tform = build_augmentation_transform(rotation=random_augmentation_params.rotation,
                                                     shear=random_augmentation_params.shear,
                                                     translation=random_augmentation_params.translation,
                                                     flip_x=random_augmentation_params.flip_x,
                                                     flip_y=random_augmentation_params.flip_y,
                                                     zoom=random_augmentation_params.zoom)
        total_tform = tform_patch_scale + tform_shift_uncenter + augment_tform + tform_shift_center + tform_normscale
        # print total_tform.params
    if add_noise is not None:
        noise_type = add_noise[rng.randint(len(add_noise))]
        image = generate_noisy_image(noise_type, image)  
    # For Multi-Channel Data warp all the slices in the same manner
    n_channels = image.shape[2]
    transformed_image = np.zeros(patch_size+(n_channels,))
    for i in range(n_channels):   
        transformed_image[:,:,i] = fast_warp(normalize(image[:,:,i]), total_tform, output_shape=patch_size, mode='symmetric')
    image = transformed_image
    label = multilabel_transform(label, total_tform, patch_size, nlabel)


    if denoise:
        if random_denoise:
            image = rng.randint(2) > 0 if denoise else image
        else:  
            image = denoise_img_vol(image)

    # apply transformation to ROI and mask the images
    if roi_center.any() and roi_radii.any() and mask_roi:
        roi_scale = random_augmentation_params.roi_scale if random_augmentation_params else 1.6  # augmentation
        roi_zoom = random_augmentation_params.zoom if random_augmentation_params else pixel_spacing
        rescaled_roi_radii = (roi_scale * roi_radii[0], roi_scale * roi_radii[1])
        out_roi_radii = (int(roi_zoom[0] * rescaled_roi_radii[0] * pixel_spacing[0] / patch_scale),
                         int(roi_zoom[1] * rescaled_roi_radii[1] * pixel_spacing[1] / patch_scale))
        roi_mask = make_circular_roi_mask(patch_size, (patch_size[0] / 2, patch_size[1] / 2), out_roi_radii)
        image *= roi_mask

    if random_augmentation_params:
        if uniform_scale:
            targets_zoom_factor = random_augmentation_params.zoom[0] * random_augmentation_params.zoom[1]
        else:
            targets_zoom_factor = pixel_spacing[0]*pixel_spacing[1]*\
                                random_augmentation_params.zoom[0]*random_augmentation_params.zoom[1]
    else:
        targets_zoom_factor = pixel_spacing[0]*pixel_spacing[1]
    return image, label, targets_zoom_factor


def make_circular_roi_mask(img_shape, roi_center, roi_radii):
    mask = np.zeros(img_shape)
    rr, cc = skimage.draw.ellipse(roi_center[0], roi_center[1], roi_radii[0], roi_radii[1], img_shape)
    mask[rr, cc] = 1.
    return mask

def fast_warp(img, tf, output_shape, mode='constant', order=1):
    """
    This wrapper function is faster than skimage.transform.warp
    """
    m = tf.params  # tf._matrix is
    # TODO: check if required
    # mode='symmetric'
    return skimage.transform._warps_cy._warp_fast(img, m, output_shape=output_shape, mode=mode, order=order)


def build_rescale_transform(downscale_factor, image_shape, target_shape):
    """
    estimating the correct rescaling transform is slow, so just use the
    downscale_factor to define a transform directly. This probably isn't
    100% correct, but it shouldn't matter much in practice.
    """
    rows, cols = image_shape
    trows, tcols = target_shape
    tform_ds = skimage.transform.AffineTransform(scale=(downscale_factor, downscale_factor))
    # centering
    shift_x = cols / (2.0 * downscale_factor) - tcols / 2.0
    shift_y = rows / (2.0 * downscale_factor) - trows / 2.0
    tform_shift_ds = skimage.transform.SimilarityTransform(translation=(shift_x, shift_y))
    return tform_shift_ds + tform_ds

def build_augmentation_transform(rotation=0, shear=0, translation=(0, 0), flip_x=False, flip_y=False, zoom=(1.0, 1.0)):
    if flip_x:
        shear += 180  # shear by 180 degrees is equivalent to flip along the X-axis
    if flip_y:
        shear += 180
        rotation += 180

    tform_augment = skimage.transform.AffineTransform(scale=(1. / zoom[0], 1. / zoom[1]), rotation=np.deg2rad(rotation),
                                                      shear=np.deg2rad(shear), translation=translation)
    return tform_augment

def build_shift_center_transform(image_shape, center_location, patch_size):
    """Shifts the center of the image to a given location.
    This function tries to include as much as possible of the image in the patch
    centered around the new center. If the patch around the ideal center
    location doesn't fit within the image, we shift the center to the right so
    that it does.
    params in (i,j) coordinates !!!
    """
    if center_location[0] < 1. and center_location[1] < 1.:
        center_absolute_location = [
            center_location[0] * image_shape[0], center_location[1] * image_shape[1]]
    else:
        center_absolute_location = [center_location[0], center_location[1]]

    # Check for overlap at the edges
    center_absolute_location[0] = max(
        center_absolute_location[0], patch_size[0] / 2.0)
    center_absolute_location[1] = max(
        center_absolute_location[1], patch_size[1] / 2.0)

    center_absolute_location[0] = min(
        center_absolute_location[0], image_shape[0] - patch_size[0] / 2.0)

    center_absolute_location[1] = min(
        center_absolute_location[1], image_shape[1] - patch_size[1] / 2.0)

    # Check for overlap at both edges
    if patch_size[0] > image_shape[0]:
        center_absolute_location[0] = image_shape[0] / 2.0
    if patch_size[1] > image_shape[1]:
        center_absolute_location[1] = image_shape[1] / 2.0

    # Build transform
    new_center = np.array(center_absolute_location)
    translation_center = new_center - 0.5
    translation_uncenter = -np.array((patch_size[0] / 2.0, patch_size[1] / 2.0)) - 0.5
    return (
        skimage.transform.SimilarityTransform(translation=translation_center[::-1]),
        skimage.transform.SimilarityTransform(translation=translation_uncenter[::-1]))

def multilabel_binarize(image_2D, nlabel):
    """
    Binarize multilabel images and return stack of binary images
    Returns: Tensor of shape: Bin_Channels* Image_shape(3D tensor)
    TODO: Labels are assumed to discreet in steps from -> 0,1,2,...,nlabel-1
    """
    labels = range(nlabel)
    out_shape = (len(labels),) + image_2D.shape
    bin_img_stack = np.ones(out_shape, dtype='uint8')
    for label in labels:
        bin_img_stack[label] = np.where(image_2D == label, bin_img_stack[label], 0)
    return bin_img_stack

def multilabel_transform(img, tf, output_shape, nlabel, mode='constant', order=0):
    """
    Binarize images do apply transform on each of the binary images and take argmax while
    doing merge operation
    Order -> 0 : nearest neighbour interpolation
    """
    bin_img_stack = multilabel_binarize(img, nlabel)
    n_labels = len(bin_img_stack)
    tf_bin_img_stack = np.zeros((n_labels,) + output_shape, dtype='uint8')
    for label in range(n_labels):
        tf_bin_img_stack[label] = fast_warp(bin_img_stack[label], tf, output_shape=output_shape, mode=mode, order=order)
    # Do merge operation along the axis = 0
    return np.argmax(tf_bin_img_stack, axis=0)


def denoise_img_vol(image_vol, weight=.01):
    denoised_img = slicewise_bilateral_filter(image_vol)
    # denoised_img = denoise_tv_chambolle(image_vol, weight=weight, eps=0.0002, n_iter_max=200, multichannel=False)
    # denoised_img = denoise_tv_bregman(image_vol, weight=1./weight, max_iter=100, eps=0.001, isotropic=True)
    # print (np.linalg.norm(denoised_img-image_vol))
    return denoised_img

def generate_noisy_image(noise_typ, image):
    """
    Parameters
    ----------
    image : ndarray
        Input image data. Will be converted to float.
    mode : str
        One of the following strings, selecting the type of noise to add:

        'gauss'     Gaussian-distributed additive noise.
        'poisson'   Poisson-distributed noise generated from the data.
        's&p'       Replaces random pixels with 0 or 1.
        'speckle'   Multiplicative noise using out = image + n*image,where
                    n is uniform noise with specified mean & variance.
        noise = ['gauss', 'poisson', 's&p', 'speckle', 'denoise', 'none1', 'none2']            
    """
    if noise_typ == "gauss":
        row,col = image.shape[:2]
        mean = 0
        var = 0.0001
        sigma = var**0.5
        gauss = rng.normal(mean,sigma,(row,col))
        gauss = gauss.reshape(row,col,1)
        noisy = image + gauss
        return noisy

    elif noise_typ == "s&p":
        s_vs_p = 0.5
        amount = 0.004
        out = np.copy(image)
        # Salt mode
        num_salt = np.ceil(amount * image.size * s_vs_p)
        coords = [rng.randint(0, i - 1, int(num_salt)) for i in image.shape]
        out[coords] = 1
        # Pepper mode
        num_pepper = np.ceil(amount* image.size * (1. - s_vs_p))
        coords = [rng.randint(0, i - 1, int(num_pepper)) for i in image.shape]
        out[coords] = 0
        return out

    elif noise_typ == "poisson":
        vals = len(np.unique(image))
        vals = 2 ** np.ceil(np.log2(vals))
        noisy = rng.poisson(image * vals) / float(vals)
        return noisy

    elif noise_typ =="speckle":
        row,col = image.shape[:2]
        gauss = 0.1*rng.randn(row,col)
        gauss = gauss.reshape(row,col,1)        
        noisy = image + image * gauss
        return noisy   
    else:
        return image  

def slicewise_bilateral_filter(data_3d, d=3, sigmaColor=8, sigmaSpace=8):
    img_batch_shape = data_3d.shape[:2] +(0,)
    img_batch = np.empty(img_batch_shape, dtype='float32')
    print (img_batch.shape)
    print (data_3d.dtype)
    try:
        slices = data_3d.shape[2] 
    except Exception:
        slices = 1
        denoised_img = bilateralFilter(data_3d[:,:].astype('float32'),d, sigmaColor, sigmaSpace)
        return denoised_img
    for i in range(slices):
        denoised_img = np.expand_dims(bilateralFilter(data_3d[:,:,i].astype('float32'),
                                        d, sigmaColor, sigmaSpace), axis=2)
        img_batch = np.concatenate((img_batch, denoised_img), axis=2)
    return img_batch


def resize_image_with_crop_or_pad_3D(image, target_height, target_width):
  """Crops and/or pads an image to a target width and height.
  Resizes an image to a target width and height by either centrally
  cropping the image or padding it evenly with zeros.
  If `width` or `height` is greater than the specified `target_width` or
  `target_height` respectively, this op centrally crops along that dimension.
  If `width` or `height` is smaller than the specified `target_width` or
  `target_height` respectively, this op centrally pads with 0 along that
  dimension.
  Args:
    image: 3-D Tensor of shape `[ height, width, channels]` or
    target_height: Target height.
    target_width: Target width.
  Raises:
    ValueError: if `target_height` or `target_width` are zero or negative.
  Returns:
    Cropped and/or padded image.
  """

  # `crop_to_bounding_box` and `pad_to_bounding_box` have their own checks.
  def max_(x, y):
      return max(x, y)

  def min_(x, y):
      return min(x, y)

  def equal_(x, y):
      return x == y

  height, width, _ = image.shape
  width_diff = target_width - width
  offset_crop_width = max_(-width_diff // 2, 0)
  offset_pad_width = max_(width_diff // 2, 0)

  height_diff = target_height - height
  offset_crop_height = max_(-height_diff // 2, 0)
  offset_pad_height = max_(height_diff // 2, 0)

  # Maybe crop if needed.
  cropped = crop_to_bounding_box_3D(image, offset_crop_height, offset_crop_width,
                                 min_(target_height, height),
                                 min_(target_width, width))

  # Maybe pad if needed.
  resized = pad_to_bounding_box_3D(cropped, offset_pad_height, offset_pad_width,
                                target_height, target_width)
  return resized

def pad_to_bounding_box_3D(image, offset_height, offset_width, target_height,
                        target_width):
  """Pad `image` with zeros to the specified `height` and `width`.
  Adds `offset_height` rows of zeros on top, `offset_width` columns of
  zeros on the left, and then pads the image on the bottom and right
  with zeros until it has dimensions `target_height`, `target_width`.
  This op does nothing if `offset_*` is zero and the image already has size
  `target_height` by `target_width`.
  Args:
    image: 2-D Tensor of shape `[height, width]`
    offset_height: Number of rows of zeros to add on top.
    offset_width: Number of columns of zeros to add on the left.
    target_height: Height of output image.
    target_width: Width of output image.
  Returns:
    If `image` was a 3-D float Tensor of shape
    `[target_height, target_width, channels]`
  Raises:
    ValueError: If the shape of `image` is incompatible with the `offset_*` or
      `target_*` arguments, or either `offset_height` or `offset_width` is
      negative.
  """
  height, width, _ = image.shape

  after_padding_width = target_width - offset_width - width
  after_padding_height = target_height - offset_height - height

  assert (offset_height >= 0),"offset_height must be >= 0"    
  assert (offset_width >= 0),"width must be <= target - offset"    
  assert (after_padding_width >= 0),"height must be <= target - offset"    
  assert (after_padding_height >= 0),"height must be <= target - offset"    

  # Do not pad on the depth dimensions.
  padded = np.lib.pad(image, ((offset_height, after_padding_height),
                     (offset_width, after_padding_width), (0, 0)), 'constant')
  return padded


def crop_to_bounding_box_3D(image, offset_height, offset_width, target_height,
                         target_width):
  """Crops an image to a specified bounding box.
  This op cuts a rectangular part out of `image`. The top-left corner of the
  returned image is at `offset_height, offset_width` in `image`, and its
  lower-right corner is at
  `offset_height + target_height, offset_width + target_width`.
  Args:
    image: 3-D Tensor of shape `[height, width, channels]`.
    offset_height: Vertical coordinate of the top-left corner of the result in
                   the input.
    offset_width: Horizontal coordinate of the top-left corner of the result in
                  the input.
    target_height: Height of the result.
    target_width: Width of the result.
  Returns:
    If `image` was  a 3-D float Tensor of shape
    `[target_height, target_width, channels]`
  Raises:
    ValueError: If the shape of `image` is incompatible with the `offset_*` or
      `target_*` arguments, or either `offset_height` or `offset_width` is
      negative, or either `target_height` or `target_width` is not positive.
  """
  height, width, _ = image.shape

  assert (offset_width >= 0),"offset_width must be >= 0."    
  assert (offset_height >= 0),"offset_height must be >= 0."    
  assert (target_width > 0),"target_width must be > 0."    
  assert (target_height > 0),"target_height must be > 0." 
  assert (width >= (target_width + offset_width)),"width must be >= target + offset."    
  assert (height >= (target_height + offset_height)),"height must be >= target + offset."    
  cropped = image[offset_height: target_height+offset_height, offset_width: target_width+offset_width, :]
  return cropped


def pad_to_bounding_box(image, offset_height, offset_width, target_height,
                        target_width, pad_mode='symmetric'):
  """Pad `image` with zeros to the specified `height` and `width`.
  Adds `offset_height` rows of zeros on top, `offset_width` columns of
  zeros on the left, and then pads the image on the bottom and right
  with zeros until it has dimensions `target_height`, `target_width`.
  This op does nothing if `offset_*` is zero and the image already has size
  `target_height` by `target_width`.
  Args:
    image: 2-D Tensor of shape `[height, width]`
    offset_height: Number of rows of zeros to add on top.
    offset_width: Number of columns of zeros to add on the left.
    target_height: Height of output image.
    target_width: Width of output image.
  Returns:
    If `image` was 2-D, a 2-D float Tensor of shape
    `[target_height, target_width]`
  Raises:
    ValueError: If the shape of `image` is incompatible with the `offset_*` or
      `target_*` arguments, or either `offset_height` or `offset_width` is
      negative.
  """
  height, width = image.shape

  after_padding_width = target_width - offset_width - width
  after_padding_height = target_height - offset_height - height

  assert (offset_height >= 0),"offset_height must be >= 0"    
  assert (offset_width >= 0),"width must be <= target - offset"    
  assert (after_padding_width >= 0),"height must be <= target - offset"    
  assert (after_padding_height >= 0),"height must be <= target - offset"    

  # Do not pad on the depth dimensions.
  padded = np.lib.pad(image, ((offset_height, after_padding_height),
                     (offset_width, after_padding_width)), pad_mode)
  return padded


def crop_to_bounding_box(image, offset_height, offset_width, target_height,
                         target_width):
  """Crops an image to a specified bounding box.
  This op cuts a rectangular part out of `image`. The top-left corner of the
  returned image is at `offset_height, offset_width` in `image`, and its
  lower-right corner is at
  `offset_height + target_height, offset_width + target_width`.
  Args:
    image: 2-D Tensor of shape `[height, width]`.
    offset_height: Vertical coordinate of the top-left corner of the result in
                   the input.
    offset_width: Horizontal coordinate of the top-left corner of the result in
                  the input.
    target_height: Height of the result.
    target_width: Width of the result.
  Returns:
    If `image` was 2-D, a 2-D float Tensor of shape
    `[target_height, target_width]`
  Raises:
    ValueError: If the shape of `image` is incompatible with the `offset_*` or
      `target_*` arguments, or either `offset_height` or `offset_width` is
      negative, or either `target_height` or `target_width` is not positive.
  """
  height, width = image.shape

  assert (offset_width >= 0),"offset_width must be >= 0."    
  assert (offset_height >= 0),"offset_height must be >= 0."    
  assert (target_width > 0),"target_width must be > 0."    
  assert (target_height > 0),"target_height must be > 0." 
  assert (width >= (target_width + offset_width)),"width must be >= target + offset."    
  assert (height >= (target_height + offset_height)),"height must be >= target + offset."    
  cropped = image[offset_height: target_height+offset_height, offset_width: target_width+offset_width]
  return cropped

def resize_image_with_crop_or_pad(image, target_height, target_width, pad_mode='symmetric'):
  """Crops and/or pads an image to a target width and height.
  Resizes an image to a target width and height by either centrally
  cropping the image or padding it evenly with zeros.
  If `width` or `height` is greater than the specified `target_width` or
  `target_height` respectively, this op centrally crops along that dimension.
  If `width` or `height` is smaller than the specified `target_width` or
  `target_height` respectively, this op centrally pads with 0 along that
  dimension.
  Args:
    image: 2-D Tensor of shape `[ height, width]` or
    target_height: Target height.
    target_width: Target width.
  Raises:
    ValueError: if `target_height` or `target_width` are zero or negative.
  Returns:
    Cropped and/or padded image.
  """

  # `crop_to_bounding_box` and `pad_to_bounding_box` have their own checks.
  def max_(x, y):
      return max(x, y)

  def min_(x, y):
      return min(x, y)

  def equal_(x, y):
      return x == y

  height, width = image.shape
  width_diff = target_width - width
  offset_crop_width = max_(-width_diff // 2, 0)
  offset_pad_width = max_(width_diff // 2, 0)

  height_diff = target_height - height
  offset_crop_height = max_(-height_diff // 2, 0)
  offset_pad_height = max_(height_diff // 2, 0)

  # Maybe crop if needed.
  cropped = crop_to_bounding_box(image, offset_crop_height, offset_crop_width,
                                 min_(target_height, height),
                                 min_(target_width, width))

  # Maybe pad if needed.
  resized = pad_to_bounding_box(cropped, offset_pad_height, offset_pad_width,
                                target_height, target_width, pad_mode)

  return resized

def get_4D_volume_of_fixed_shape(img_files_path_list, max_size=(256, 256)):
    """
    For the LV 2011 challenge the dataset is in the form of 2D images.
    So group files and arrange it form a 4d volume: (X, Y, slice, time)
    and crop or pad image .
    Assumption: All the images have same pixel spacing
    Returns: 4D volume and pixel spacing   
    """
    folder_path = os.path.dirname(img_files_path_list[0])
    # print (folder_path)
    file_names = [os.path.basename(file_path) for file_path in img_files_path_list]     
    sorted_file_names = sorted(file_names, 
        key=lambda x: tuple(int(i) for i in re.findall('\d+', x)[1:]))
    sorted_file_path = [os.path.join(folder_path, file) for file in sorted_file_names]
    #print (path_list)
    sa_list = []
    ph_list = [] 
    # print file_list
    for file_name in sorted_file_names:
        pat_id = re.findall(r'\d+', file_name)[0]
        sa = re.findall(r'\d+', file_name)[1]
        ph = re.findall(r'\d+', file_name)[2]
        if not int(sa) in sa_list:
            sa_list.append(int(sa))
        if not int(ph) in ph_list:
            ph_list.append(int(ph))
    sa_list_sorted = np.sort(sa_list)
    ph_list_sorted = np.sort(ph_list)
    n_slice = len(sa_list_sorted)
    n_phase = len(ph_list_sorted)
    out_vol = np.zeros(max_size+(n_slice, n_phase))
    iter_cnt = 0
    for slice in range(n_slice):
        s_time = time.time()
        for phase in range(n_phase):
            img = sitk.ReadImage(sorted_file_path[iter_cnt])
            img_array = sitk.GetArrayFromImage(img)
            if img_array[0].shape != max_size:
                img_array = resize_image_with_crop_or_pad(img_array[0], max_size[0], max_size[1])
            out_vol[:,:,slice, phase] = img_array
            iter_cnt += 1
    return out_vol, img.GetSpacing()


def get_4D_volume(path_list, gt=False, gt_shape=None):
    #print (path_list)
    sa_list = []
    ph_list = [] 
    boImgSizeNotEq = False
    ref_img_size = sitk.ReadImage(path_list[0]).GetSize()
    # print file_list
    for path in path_list:
        file_name = os.path.basename(path)
        #print re.findall(r'\d+', file_name)
        pat_id = re.findall(r'\d+', file_name)[0]
        sa = re.findall(r'\d+', file_name)[1]
        ph = re.findall(r'\d+', file_name)[2]
        if not int(sa) in sa_list:
            sa_list.append(int(sa))
        if not int(ph) in ph_list:
            ph_list.append(int(ph))
        # Check if the sizes of all slices are equal
        img_size = sitk.ReadImage(path).GetSize()
        if img_size != ref_img_size:
            boImgSizeNotEq = True
#             print ('The Sizes donot match: the image will cropped or padded to reference slice')
    sa_list_sorted = np.sort(sa_list)
    ph_list_sorted = np.sort(ph_list)
    n_slices = len(sa_list_sorted)
    n_phases = len(ph_list_sorted)
    img = sitk.ReadImage(path_list[0])
    img_data = sitk.GetArrayFromImage(img)
#     print (img_data.shape)
    if not gt:
        x_dim, y_dim = img_data.shape[1:]      
#         print(img.GetOrigin())
#         print(img.GetSize())
#         print(img.GetSpacing())
#         print(img.GetDirection())
    else:
        x_dim, y_dim = img_data.shape[:2]
    pat_id = re.findall(r'\d+', os.path.basename(path_list[0]))[0]
    pat_dir = os.path.dirname(path_list[0])
#     print (pat_id, pat_dir) 
    if not gt:
        data_4d = np.zeros([x_dim, y_dim, n_slices, n_phases], dtype=np.uint16)
    else:
        if not gt_shape:
            data_4d = np.zeros([x_dim, y_dim, n_slices, n_phases], dtype=np.uint8)
        else:
            data_4d = np.zeros(gt_shape, dtype=np.uint8)
            x_dim, y_dim = gt_shape[:2]     
    for slice in sa_list_sorted:
        for phase in ph_list_sorted:
#             print (phase, slice)
            if not gt:
                file_path = (pat_dir + "/DET"+pat_id+"_SA"+str(slice)+"_ph"+str(phase)+".dcm")
                slice_img = sitk.ReadImage(file_path)
                img_data = sitk.GetArrayFromImage(slice_img)   
                data_4d[:,:,slice-1,phase] = resize_image_with_crop_or_pad(img_data[0,:,:], x_dim, y_dim)
            else:
                file_path = (pat_dir + "/DET"+pat_id+"_SA"+str(slice)+"_ph"+str(phase)+".png")                
                slice_img = sitk.ReadImage(file_path)
                img_data = sitk.GetArrayFromImage(slice_img)
                # Ground Truth Preprocessing: Threshold the image between (0,1)
                img_data[np.where(img_data>0)] = 1
                # print (data_4d.shape, img_data.shape)
                data_4d[:,:,slice-1, phase] = resize_image_with_crop_or_pad(img_data[:,:,0], x_dim, y_dim)
    if not gt: 
        pixel_spacing = img.GetSpacing()
        pixel_spacing += (1,) 
        return (data_4d, pixel_spacing)
    else:
        return data_4d

def getLargestConnectedComponent(itk_img):
    data = np.uint8(sitk.GetArrayFromImage(itk_img))
    c,n = snd.label(data)
    sizes = snd.sum(data, c, range(n+1))
    mask_size = sizes < (max(sizes))
    remove_voxels = mask_size[c]
    c[remove_voxels] = 0
    c[np.where(c!=0)]=1
    data[np.where(c==0)] = 0
    return sitk.GetImageFromArray(data)

def getLargestConnectedComponent_2D(itk_img):
    data = np.uint8(sitk.GetArrayFromImage(itk_img))
    for i in range(data.shape[0]):
        c,n = snd.label(data[i])
        sizes = snd.sum(data[i], c, range(n+1))
        mask_size = sizes < (max(sizes))
        remove_voxels = mask_size[c]
        c[remove_voxels] = 0
        c[np.where(c!=0)]=1
        data[i][np.where(c==0)] = 0
    return sitk.GetImageFromArray(data)

def maskROIInPrediction(img,roi_mask_path):
    # TODO: Generalize the code
    if os.path.exists(roi_mask_path):
        data = sitk.GetArrayFromImage(img)
        mask_data = sitk.GetArrayFromImage(sitk.ReadImage(roi_mask_path))
        data[np.where(mask_data == 0)] = 0
        tumor_pos = np.where(data == 2)
        data[mask_data == 1] = 1
        data[tumor_pos] = 2
        return data
    else:
        print("mask path " + roi_mask_path + " doesn't exist")
        raise 2

# def hole_fill_connected_components(pred_path='pred_path', out_path='post_pred_path'):
#     """
#     connected component analysis to remove outliers and clean predictions
#     """
#     print('Post Processing to remove connected components')
#     if not os.path.exists(out_path):
#         os.makedirs(out_path)

#     images=os.listdir(pred_path)
#     for i in sorted(images):
#         # print i
#         predicted_image = nib.load(pred_path+'/'+i)
#         affine=predicted_image.get_affine()
#         hdr = predicted_image.header
#         predicted_image = predicted_image.get_data()
#         predicted_image = utils.multilabel_split(predicted_image)
#         output_slice = np.empty((predicted_image.shape[:2]+(0,)), dtype=np.uint8)
#         for j in range(predicted_image.shape[2]):
#           n_labels = predicted_image.shape[3]
#           bin_img_stack = np.zeros(predicted_image.shape[:2]+(n_labels,), dtype='uint8')
#           for k in range(n_labels):
#             mask = predicted_image[:,:,j,k]>0
#             o_image = predicted_image[:,:,j,k]
#             connected_components, unique_number = ndimage.label(mask)
#             # print unique_number
#             sizes = ndimage.sum(mask,connected_components,range(unique_number+1))
#             # print sizes
#             Threshold = np.max(sizes)
#             # print Threshold
#             mask_size= sizes< Threshold
#             remove_voxels = mask_size[connected_components]
#             # print remove_voxels
#             connected_components[remove_voxels] = 0
#             connected_components[np.where(connected_components!=0)]=1
#             o_image[np.where(connected_components==0)] = 0

#             if k!=2 and k!=0:
#               #WARNING: Background and Myocardium labels are 0 and 2 and dont fill holes
#               fill_image = ndimage.morphology.binary_fill_holes(o_image).astype(np.uint8)
#               # fill_image = o_image
#             else:
#               fill_image = o_image
#             # print k
#             bin_img_stack[:,:,k] = fill_image
#           # utils.imshow(o_image, fill_image, bin_img_stack[:,:,2])
#           # Do merge operation along the axis = 2
#           m_image=np.argmax(bin_img_stack, axis=2)
#           # print (m_image.shape, bin_img_stack.shape, np.unique(m_image), np.unique(bin_img_stack))
#           # utils.imshow(m_image, bin_img_stack[:,:,2])
#           output_slice = np.append(output_slice, np.expand_dims(m_image, axis=2), axis=2)
#         # print output_slice.shape

#         img= nib.Nifti1Image(output_slice, affine, hdr)
#         img.set_data_dtype(np.uint8)
#         save_path = out_path+'/'+i
#         nib.save(img, save_path)


# def doCRF(img,posteriors):
#     # TODO: check
#     img_data = sitk.GetArrayFromImage(img)
#     mn, mx = getZminAndmax(img_data)
#     mn, mx = mn-5, mx+5

#     if mn < 0:
#         mn = 0
#     if mx > img_data.shape[0]:
#         mx = img_data.shape[0]

#     crfparams = {'max_iterations': 100 ,'dynamic_z': True ,'ignore_memory': True ,

#                 'pos_x_std': 3 ,'pos_y_std': 0.75,'pos_z_std': 3,'pos_w': 0.75 ,

#                 'bilateral_x_std': 60,'bilateral_y_std': 15,'bilateral_z_std': 15,

#                 'bilateral_intensity_std': 10.0,'bilateral_w': 0.25,'verbose': False}

#     pro = CRFProcessor.CRF3DProcessor(**crfparams)
#     crf_out = np.zeros(img_data.shape)
#     # print (mn, mx, np.min(img_data), np.max(img_data))
#     crf_out[mn:mx] = pro.set_data_and_run(np.uint8(img_data[mn:mx]==2), posteriors[mn:mx])

#     return sitk.GetImageFromArray(crf_out)