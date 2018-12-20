import numpy as np
import os, sys, shutil, time, re
import h5py
import skimage.morphology as morph
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib import animation
import time
import pickle
# For ROI extraction
import skimage.transform
from scipy.fftpack import fftn, ifftn
from skimage.feature import peak_local_max, canny
from skimage.transform import hough_circle 
# Nifti processing
import nibabel as nib
from collections import OrderedDict
# print sys.path
# sys.path.append("..") 
import errno
np.random.seed(42)

# Helper functions
## Heart Metrics
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
        volume = seg_3Dmap_copy.sum() * np.prod(voxel_size) / 1000.
        volumes += [volume]
    return volumes

def ejection_fraction(ed_vol, es_vol):
    """
    Calculate ejection fraction
    """
    stroke_vol = ed_vol - es_vol
    return (np.float(stroke_vol)/np.float(ed_vol))*100

def myocardialmass(myocardvol):
    """
    Specific gravity of heart muscle (1.05 g/ml)
    """ 
    return myocardvol*1.05
def imshow(*args,**kwargs):
    """ Handy function to show multiple plots in on row, possibly with different cmaps and titles
    Usage: 
    imshow(img1, title="myPlot")
    imshow(img1,img2, title=['title1','title2'])
    imshow(img1,img2, cmap='hot')
    imshow(img1,img2,cmap=['gray','Blues']) """
    cmap = kwargs.get('cmap', 'gray')
    title= kwargs.get('title','')
    if len(args)==0:
        raise ValueError("No images given to imshow")
    elif len(args)==1:
        plt.title(title)
        plt.imshow(args[0], interpolation='none')
    else:
        n=len(args)
        if type(cmap)==str:
            cmap = [cmap]*n
        if type(title)==str:
            title= [title]*n
        plt.figure(figsize=(n*5,10))
        for i in range(n):
            plt.subplot(1,n,i+1)
            plt.title(title[i])
            plt.imshow(args[i], cmap[i])
    plt.show()
    
def plot_roi(data4D, roi_center, roi_radii):
    """
    Do the animation of full heart volume
    """
    x_roi_center, y_roi_center = roi_center[0], roi_center[1]
    x_roi_radius, y_roi_radius = roi_radii[0], roi_radii[1]
    print ('nslices', data4D.shape[2])

    zslices = data4D.shape[2]
    tframes = data4D.shape[3]

    slice_cnt = 0
    for slice in [data4D[:,:,z,:] for z in range(zslices)]:
      outdata = np.swapaxes(np.swapaxes(slice[:,:,:], 0,2), 1,2)
      roi_mask = np.zeros_like(outdata[0])
      roi_mask[x_roi_center - x_roi_radius:x_roi_center + x_roi_radius,
      y_roi_center - y_roi_radius:y_roi_center + y_roi_radius] = 1

      outdata[:, roi_mask > 0.5] = 0.8 * outdata[:, roi_mask > 0.5]
      outdata[:, roi_mask > 0.5] = 0.8 * outdata[:, roi_mask > 0.5]

      fig = plt.figure(1)
      fig.canvas.set_window_title('slice_No' + str(slice_cnt))
      slice_cnt+=1
      def init_out():
          im.set_data(outdata[0])

      def animate_out(i):
          im.set_data(outdata[i])
          return im

      im = fig.gca().imshow(outdata[0], cmap='gray')
      anim = animation.FuncAnimation(fig, animate_out, init_func=init_out, frames=tframes, interval=50)
      anim.save('Cine_MRI_SAX_%d.mp4'%slice_cnt, fps=50, extra_args=['-vcodec', 'libx264'])
      plt.show()
        
def plot_4D(data4D):
    """
    Do the animation of full heart volume
    """
    print ('nslices', data4D.shape[2])
    zslices = data4D.shape[2]
    tframes = data4D.shape[3]

    slice_cnt = 0
    for slice in [data4D[:,:,z,:] for z in range(zslices)]:
      outdata = np.swapaxes(np.swapaxes(slice[:,:,:], 0,2), 1,2)
      fig = plt.figure(1)
      fig.canvas.set_window_title('slice_No' + str(slice_cnt))
      slice_cnt+=1
      def init_out():
          im.set_data(outdata[0])

      def animate_out(i):
          im.set_data(outdata[i])
          return im

      im = fig.gca().imshow(outdata[0], cmap='gray')
      anim = animation.FuncAnimation(fig, animate_out, init_func=init_out, frames=tframes, interval=50)
      plt.show()


def multilabel_split(image_tensor):
    """
    image_tensor : Batch * H * W
    Split multilabel images and return stack of images
    Returns: Tensor of shape: Batch * H * W * n_class (4D tensor)
    # TODO: Be careful: when using this code: labels need to be 
    defined, explictly before hand as this code does not handle
    missing labels
    So far, this function is okay as it considers full volume for
    finding out unique labels
    """
    labels = np.unique(image_tensor)
    batch_size = image_tensor.shape[0]
    out_shape =  image_tensor.shape + (len(labels),)
    image_tensor_4D = np.zeros(out_shape, dtype='uint8')
    for i in xrange(batch_size):
        cnt = 0
        shape =image_tensor.shape[1:3] + (len(labels),)
        temp = np.ones(shape, dtype='uint8')
        for label in labels:
            temp[...,cnt] = np.where(image_tensor[i] == label, temp[...,cnt], 0)
            cnt += 1
        image_tensor_4D[i] = temp
    return image_tensor_4D

def save_data(data, filename, out_path):
    out_filename = os.path.join(out_path, filename)
    with open(out_filename, 'wb') as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
    print ('saved to %s' % out_filename)

def load_pkl(path):
    with open(path, 'rb') as f:
        obj = pickle.load(f)
    return obj

### Stratified Sampling of data

# Refer:
# http://www.echopedia.org/wiki/Left_Ventricular_Dimensions
# https://www.creatis.insa-lyon.fr/Challenge/acdc/databases.html
# https://en.wikipedia.org/wiki/Body_surface_area
# 30 normal subjects - NOR
NORMAL = 'NOR'
# 30 patients with previous myocardial infarction 
# (ejection fraction of the left ventricle lower than 40% and several myocardial segments with abnormal contraction) - MINF
MINF = 'MINF'
# 30 patients with dilated cardiomyopathy 
# (diastolic left ventricular volume >100 mL/m2 and an ejection fraction of the left ventricle lower than 40%) - DCM
DCM = 'DCM'
# 30 patients with hypertrophic cardiomyopathy 
# (left ventricular cardiac mass high than 110 g/m2,
# several myocardial segments with a thickness higher than 15 mm in diastole and a normal ejecetion fraction) - HCM
HCM = 'HCM'
# 30 patients with abnormal right ventricle (volume of the right ventricular 
# cavity higher than 110 mL/m2 or ejection fraction of the rigth ventricle lower than 40%) - RV
RV = 'RV'
def copy(src, dest):
  """
  Copy function
  """
  try:
      shutil.copytree(src, dest, ignore=shutil.ignore_patterns())
  except OSError as e:
      # If the error was caused because the source wasn't a directory
      if e.errno == errno.ENOTDIR:
          shutil.copy(src, dest)
      else:
          print('Directory not copied. Error: %s' % e)

def read_patient_cfg(path):
  """
  Reads patient data in the cfg file and returns a dictionary
  """
  patient_info = {}
  with open(os.path.join(path, 'Info.cfg')) as f_in:
    for line in f_in:
      l = line.rstrip().split(": ")
      patient_info[l[0]] = l[1]
  return patient_info
     
def group_patient_cases(src_path, out_path, force=False):
  """ Group the patient data according to cardiac pathology""" 

  cases = sorted(next(os.walk(src_path))[1])
  dest_path = os.path.join(out_path, 'Patient_Groups')
  if force:
    shutil.rmtree(dest_path)
  if os.path.exists(dest_path):
    return dest_path  

  os.makedirs(dest_path)
  os.mkdir(os.path.join(dest_path, NORMAL))
  os.mkdir(os.path.join(dest_path, MINF))
  os.mkdir(os.path.join(dest_path, DCM))
  os.mkdir(os.path.join(dest_path, HCM))
  os.mkdir(os.path.join(dest_path, RV))

  for case in cases:
    full_path = os.path.join(src_path, case)
    copy(full_path, os.path.join(dest_path,\
        read_patient_cfg(full_path)['Group'], case))

def generate_train_validate_test_set(src_path, dest_path):
  """
  Split the data into 70:15:15 for train-validate-test set
  arg: path: input data path
  """
  SPLIT_TRAIN = 0.7
  SPLIT_VALID = 0.15

  dest_path = os.path.join(dest_path,'dataset')
  if os.path.exists(dest_path):
    shutil.rmtree(dest_path)
  os.makedirs(os.path.join(dest_path, 'train_set'))  
  os.makedirs(os.path.join(dest_path, 'validation_set'))  
  os.makedirs(os.path.join(dest_path, 'test_set'))  
  # print (src_path)
  groups = next(os.walk(src_path))[1]
  for group in groups:
    group_path = next(os.walk(os.path.join(src_path, group)))[0]
    patient_folders = next(os.walk(group_path))[1]
    np.random.shuffle(patient_folders)
    train_ = patient_folders[0:int(SPLIT_TRAIN*len(patient_folders))]
    valid_ = patient_folders[int(SPLIT_TRAIN*len(patient_folders)): 
                 int((SPLIT_TRAIN+SPLIT_VALID)*len(patient_folders))]
    test_ = patient_folders[int((SPLIT_TRAIN+SPLIT_VALID)*len(patient_folders)):]
    for patient in train_:
      folder_path = os.path.join(group_path, patient)
      copy(folder_path, os.path.join(dest_path, 'train_set', patient))

    for patient in valid_:
      folder_path = os.path.join(group_path, patient)
      copy(folder_path, os.path.join(dest_path, 'validation_set', patient))

    for patient in test_:
      folder_path = os.path.join(group_path, patient)
      copy(folder_path, os.path.join(dest_path, 'test_set', patient))

#   Fourier-Hough Transform Based ROI Extraction
def extract_roi_fft(data4D, pixel_spacing, minradius_mm=15, maxradius_mm=45, kernel_width=5, 
                center_margin=8, num_peaks=10, num_circles=20, radstep=2):
    """
    Returns center and radii of ROI region in (i,j) format
    """
    # Data shape: 
    # radius of the smallest and largest circles in mm estimated from the train set
    # convert to pixel counts

    pixel_spacing_X, pixel_spacing_Y, _,_ = pixel_spacing
    minradius = int(minradius_mm / pixel_spacing_X)
    maxradius = int(maxradius_mm / pixel_spacing_Y)

    ximagesize = data4D.shape[0]
    yimagesize = data4D.shape[1]
    zslices = data4D.shape[2]
    tframes = data4D.shape[3]
    xsurface = np.tile(range(ximagesize), (yimagesize, 1)).T
    ysurface = np.tile(range(yimagesize), (ximagesize, 1))
    lsurface = np.zeros((ximagesize, yimagesize))

    allcenters = []
    allaccums = []
    allradii = []

    for slice in range(zslices):
        ff1 = fftn([data4D[:,:,slice, t] for t in range(tframes)])
        fh = np.absolute(ifftn(ff1[1, :, :]))
        fh[fh < 0.1 * np.max(fh)] = 0.0
        image = 1. * fh / np.max(fh)
        # find hough circles and detect two radii
        edges = canny(image, sigma=3)
        hough_radii = np.arange(minradius, maxradius, radstep)
        # print hough_radii
        hough_res = hough_circle(edges, hough_radii)
        if hough_res.any():
            centers = []
            accums = []
            radii = []

            for radius, h in zip(hough_radii, hough_res):
                # For each radius, extract num_peaks circles
                peaks = peak_local_max(h, num_peaks=num_peaks)
                centers.extend(peaks)
                accums.extend(h[peaks[:, 0], peaks[:, 1]])
                radii.extend([radius] * num_peaks)
  
            # Keep the most prominent num_circles circles
            sorted_circles_idxs = np.argsort(accums)[::-1][:num_circles]

            for idx in sorted_circles_idxs:
                center_x, center_y = centers[idx]
                allcenters.append(centers[idx])
                allradii.append(radii[idx])
                allaccums.append(accums[idx])
                brightness = accums[idx]
                lsurface = lsurface + brightness * np.exp(
                    -((xsurface - center_x) ** 2 + (ysurface - center_y) ** 2) / kernel_width ** 2)

    lsurface = lsurface / lsurface.max()
    # select most likely ROI center
    roi_center = np.unravel_index(lsurface.argmax(), lsurface.shape)

    # determine ROI radius
    roi_x_radius = 0
    roi_y_radius = 0
    for idx in range(len(allcenters)):
        xshift = np.abs(allcenters[idx][0] - roi_center[0])
        yshift = np.abs(allcenters[idx][1] - roi_center[1])
        if (xshift <= center_margin) & (yshift <= center_margin):
            roi_x_radius = np.max((roi_x_radius, allradii[idx] + xshift))
            roi_y_radius = np.max((roi_y_radius, allradii[idx] + yshift))

    if roi_x_radius > 0 and roi_y_radius > 0:
        roi_radii = roi_x_radius, roi_y_radius
    else:
        roi_radii = None

    return roi_center, roi_radii

#   Stddev-Hough Transform Based ROI Extraction
def extract_roi_stddev(data4D, pixel_spacing, minradius_mm=15, maxradius_mm=45, kernel_width=5, 
                center_margin=8, num_peaks=10, num_circles=20, radstep=2):
    """
    Returns center and radii of ROI region in (i,j) format
    """
    # Data shape: 
    # radius of the smallest and largest circles in mm estimated from the train set
    # convert to pixel counts

    pixel_spacing_X, pixel_spacing_Y, _,_ = pixel_spacing
    minradius = int(minradius_mm / pixel_spacing_X)
    maxradius = int(maxradius_mm / pixel_spacing_Y)

    ximagesize = data4D.shape[0]
    yimagesize = data4D.shape[1]
    zslices = data4D.shape[2]
    tframes = data4D.shape[3]
    xsurface = np.tile(range(ximagesize), (yimagesize, 1)).T
    ysurface = np.tile(range(yimagesize), (ximagesize, 1))
    lsurface = np.zeros((ximagesize, yimagesize))

    allcenters = []
    allaccums = []
    allradii = []

    for slice in range(zslices):
        ff1 = np.array([data4D[:,:,slice, t] for t in range(tframes)])
        fh = np.std(ff1, axis=0)
        fh[fh < 0.1 * np.max(fh)] = 0.0
        image = 1. * fh / np.max(fh)
        # find hough circles and detect two radii
        edges = canny(image, sigma=3)
        hough_radii = np.arange(minradius, maxradius, radstep)
        # print hough_radii
        hough_res = hough_circle(edges, hough_radii)
        if hough_res.any():
            centers = []
            accums = []
            radii = []
            for radius, h in zip(hough_radii, hough_res):
                # For each radius, extract num_peaks circles
                peaks = peak_local_max(h, num_peaks=num_peaks)
                centers.extend(peaks)
                accums.extend(h[peaks[:, 0], peaks[:, 1]])
                radii.extend([radius] * num_peaks)
  
            # Keep the most prominent num_circles circles
            sorted_circles_idxs = np.argsort(accums)[::-1][:num_circles]

            for idx in sorted_circles_idxs:
                center_x, center_y = centers[idx]
                allcenters.append(centers[idx])
                allradii.append(radii[idx])
                allaccums.append(accums[idx])
                brightness = accums[idx]
                lsurface = lsurface + brightness * np.exp(
                    -((xsurface - center_x) ** 2 + (ysurface - center_y) ** 2) / kernel_width ** 2)

    lsurface = lsurface / lsurface.max()
    # select most likely ROI center
    roi_center = np.unravel_index(lsurface.argmax(), lsurface.shape)

    # determine ROI radius
    roi_x_radius = 0
    roi_y_radius = 0
    for idx in range(len(allcenters)):
        xshift = np.abs(allcenters[idx][0] - roi_center[0])
        yshift = np.abs(allcenters[idx][1] - roi_center[1])
        if (xshift <= center_margin) & (yshift <= center_margin):
            roi_x_radius = np.max((roi_x_radius, allradii[idx] + xshift))
            roi_y_radius = np.max((roi_y_radius, allradii[idx] + yshift))

    if roi_x_radius > 0 and roi_y_radius > 0:
        roi_radii = roi_x_radius, roi_y_radius
    else:
        roi_radii = None

    return roi_center, roi_radii


class Dataset(object):
    def __init__(self, directory, subdir):
        # type: (object, object) -> object
        self.patient_data = {}
        self.directory = directory
        self.name = subdir

    def _filename(self, file):
        return os.path.join(self.directory, self.name, file)

    def load_nii(self, img_path):
        """
        Function to load a 'nii' or 'nii.gz' file, The function returns
        everyting needed to save another 'nii' or 'nii.gz'
        in the same dimensional space, i.e. the affine matrix and the header

        Parameters
        ----------

        img_path: string
        String with the path of the 'nii' or 'nii.gz' image file name.

        Returns
        -------
        Three element, the first is a numpy array of the image values,
        the second is the affine transformation of the image, and the
        last one is the header of the image.
        """
        nimg = nib.load(self._filename(img_path))
        return nimg.get_data(), nimg.affine, nimg.header
    
    def read_patient_info_data(self):
        """
        Reads patient data in the cfg file from patient folder 
        using Info.cfg
        """
        print (self._filename('Info.cfg'))
        with open(self._filename('Info.cfg')) as f_in:
            for line in f_in:
              l = line.rstrip().split(": ")
              self.patient_data[l[0]] = l[1]

    def read_patient_data(self, mode='train', roi_detect=True):
        """
        Reads patient data in the cfg file and returns a dictionary and
        extract End diastole and End Systole image from patient folder
        using Info.cfg
        """
        self.read_patient_info_data()
        # Read patient Number
        m = re.match("patient(\d{3})", self.name)
        patient_No = int(m.group(1))
        # Read Diastole frame Number
        ED_frame_No = int(self.patient_data['ED'])
        ed_img = "patient%03d_frame%02d.nii.gz" %(patient_No, ED_frame_No)
        ed, affine, hdr  = self.load_nii(ed_img)
        # Read Systole frame Number
        ES_frame_No = int(self.patient_data['ES'])
        es_img = "patient%03d_frame%02d.nii.gz" %(patient_No, ES_frame_No)
        es, _, _  = self.load_nii(es_img)
        # Save Images:
        self.patient_data['ED_VOL'] = ed
        self.patient_data['ES_VOL'] = es
 
        # Header Info for saving    
        header_info ={'affine':affine, 'hdr': hdr}
        self.patient_data['header'] = header_info
        if mode == 'reader':
            # Read a particular volume number in 4D image
            img_4d_name = "patient%03d_4d.nii.gz"%patient_No
            # Load data
            img_4D, _, hdr = self.load_nii(img_4d_name)
            self.patient_data['4D'] = img_4D

            ed_gt, _, _  = self.load_nii("patient%03d_frame%02d_gt.nii.gz" %(patient_No, ED_frame_No))
            es_gt, _, _  = self.load_nii("patient%03d_frame%02d_gt.nii.gz" %(patient_No, ES_frame_No))
            ed_lv, ed_rv, ed_myo = heart_metrics(ed_gt, hdr.get_zooms()) 
            es_lv, es_rv, es_myo = heart_metrics(es_gt, hdr.get_zooms())
            ef_lv = ejection_fraction(ed_lv, es_lv)
            ef_rv = ejection_fraction(ed_rv, es_rv)
            heart_param = {'EDV_LV': ed_lv, 'EDV_RV': ed_rv, 'ESV_LV': es_lv, 'ESV_RV': es_rv,
                           'ED_MYO': ed_myo, 'ES_MYO': es_myo, 'EF_LV': ef_lv, 'EF_RV': ef_rv}  
            self.patient_data['HP'] = heart_param 
            self.patient_data['ED_GT'] = ed_gt
            self.patient_data['ES_GT'] = es_gt
            return

        if mode == 'train':
            ed_gt, _, _  = self.load_nii("patient%03d_frame%02d_gt.nii.gz" %(patient_No, ED_frame_No))
            es_gt, _, _  = self.load_nii("patient%03d_frame%02d_gt.nii.gz" %(patient_No, ES_frame_No))
            ed_lv, ed_rv, ed_myo = heart_metrics(ed_gt, hdr.get_zooms()) 
            es_lv, es_rv, es_myo = heart_metrics(es_gt, hdr.get_zooms())
            ef_lv = ejection_fraction(ed_lv, es_lv)
            ef_rv = ejection_fraction(ed_rv, es_rv)
            heart_param = {'EDV_LV': ed_lv, 'EDV_RV': ed_rv, 'ESV_LV': es_lv, 'ESV_RV': es_rv,
                           'ED_MYO': ed_myo, 'ES_MYO': es_myo, 'EF_LV': ef_lv, 'EF_RV': ef_rv}  
            self.patient_data['HP'] = heart_param 
            self.patient_data['ED_GT'] = ed_gt
            self.patient_data['ES_GT'] = es_gt

        if mode == 'tester':
            # Read a particular volume number in 4D image
            img_4d_name = "patient%03d_4d.nii.gz"%patient_No
            # Load data
            img_4D, _, hdr = self.load_nii(img_4d_name)
            self.patient_data['4D'] = img_4D

        if roi_detect:
            # Read a particular volume number in 4D image
            img_4d_name = "patient%03d_4d.nii.gz"%patient_No
            # Load data
            img_4D, _, hdr = self.load_nii(img_4d_name)
            c, r = extract_roi_stddev(img_4D, hdr.get_zooms()) 
            self.patient_data['roi_center'], self.patient_data['roi_radii']=c,r 
            self.patient_data['4D'] = img_4D
#             print c, r
#             plot_roi(img_4D, c,r)
            
def convert_nii_np(data_path, mode, roi_detect):
    """
    Prepare a dictionary of dataset and save it as numpy file
    """
    patient_fulldata = OrderedDict()
    print (data_path)
    patient_folders = next(os.walk(data_path))[1]
    for patient in tqdm(sorted(patient_folders)):
#         print (patient)
        dset = Dataset(data_path, patient)
        dset.read_patient_data(mode=mode, roi_detect=roi_detect)
        patient_fulldata[dset.name] = dset.patient_data
    return patient_fulldata

if __name__ == '__main__':
  start_time = time.time()
  # Path to ACDC training database
  complete_data_path = '../../ACDC_DataSet/training'
  dest_path = '../../processed_acdc_dataset'
  group_path = '../../processed_acdc_dataset/Patient_Groups'

  # Training dataset
  train_dataset = '../../processed_acdc_dataset/dataset/train_set'
  validation_dataset = '../../processed_acdc_dataset/dataset/validation_set'
  test_dataset = '../../processed_acdc_dataset/dataset/test_set'
  out_path_train = '../../processed_acdc_dataset/pickled/full_data'
  hdf5_out_path = '../../processed_acdc_dataset/hdf5_files'
  #Final Test dataset
  final_testing_dataset = '../../ACDC_DataSet/testing'
  out_path_test = '../../processed_acdc_dataset/pickled/final_test'

  # First perform stratified sampling
  group_patient_cases(complete_data_path, dest_path)
  generate_train_validate_test_set(group_path, dest_path)
  print("---Time taken to stratify the dataset %s seconds ---" % (time.time() - start_time))

  print ('ROI->ED->ES train dataset')
  if not os.path.exists(out_path_train):
      os.makedirs(out_path_train)
      os.makedirs(out_path_test)
      
  train_dataset = convert_nii_np(train_dataset, mode='train', roi_detect=True)
  save_data(train_dataset, 'train_set.pkl', out_path_train)
  print("---Processing Training dataset %s seconds ---" % (time.time() - start_time))
  validation_dataset = convert_nii_np(validation_dataset, mode='train', roi_detect=True)
  save_data(validation_dataset, 'validation_set.pkl', out_path_train)
  print("---Processing Training dataset %s seconds ---" % (time.time() - start_time))
  test_dataset = convert_nii_np(test_dataset, mode='train', roi_detect=True)
  save_data(test_dataset, 'test_set.pkl', out_path_train)
  print("---Processing Training dataset %s seconds ---" % (time.time() - start_time))

  print ('ROI->ED->ES test dataset')
  final_test_dataset = convert_nii_np(final_testing_dataset, mode='test', roi_detect=True)
  save_data(final_test_dataset, 'final_testing_data.pkl', out_path_test)
  print("---Processing final testing dataset %s seconds ---" % (time.time() - start_time))

  # Generate 2D HDF5 files
  modes = ['train_set', 'validation_set', 'test_set']
  for mode in modes: 
      if os.path.exists(os.path.join(hdf5_out_path, mode)):
          shutil.rmtree(os.path.join(hdf5_out_path, mode))
      os.makedirs(os.path.join(hdf5_out_path, mode))
      patient_data = load_pkl(os.path.join(out_path_train, mode+'.pkl'))
      for patient_id in tqdm(patient_data.keys()):
      #     print (patient_id)
          _id = patient_id[-3:]
          n_slices = patient_data[patient_id]['ED_VOL'].shape[2]
  #         print (n_slices)
          for slice in range(n_slices):
  #           ED frames
              group = patient_data[patient_id]['Group']
              slice_str ='_%02d_'%slice
              roi_center = (patient_data[patient_id]['roi_center'][1], patient_data[patient_id]['roi_center'][0])
              hp = h5py.File(os.path.join(hdf5_out_path, mode, 'P_'+_id+'_ED'+slice_str+group+'.hdf5'),'w')
              hp.create_dataset('image', data=patient_data[patient_id]['ED_VOL'][:,:,slice].T)
              hp.create_dataset('label', data=patient_data[patient_id]['ED_GT'][:,:,slice].T)
              hp.create_dataset('roi_center', data=roi_center)
              hp.create_dataset('roi_radii', data=patient_data[patient_id]['roi_radii'])
              hp.create_dataset('pixel_spacing', data=patient_data[patient_id]['header']['hdr'].get_zooms())
              hp.close()
  #           ES frames
              hp = h5py.File(os.path.join(hdf5_out_path, mode, 'P_'+_id+'_ES'+slice_str+group+'.hdf5'),'w')
              hp.create_dataset('image', data=patient_data[patient_id]['ES_VOL'][:,:,slice].T)
              hp.create_dataset('label', data=patient_data[patient_id]['ES_GT'][:,:,slice].T)
              hp.create_dataset('roi_center', data=roi_center)
              hp.create_dataset('roi_radii', data=patient_data[patient_id]['roi_radii'])
              hp.create_dataset('pixel_spacing', data=patient_data[patient_id]['header']['hdr'].get_zooms())
              hp.close()    
  print("---Time taken to generate hdf5 files %s seconds ---" % (time.time() - start_time))