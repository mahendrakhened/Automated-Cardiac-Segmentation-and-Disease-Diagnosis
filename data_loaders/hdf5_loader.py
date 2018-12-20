# Imports
from __future__ import division
import numpy as np
import os, sys
import h5py
import random, time
import weakref
import threading
from datetime import datetime
import glob
sys.path.append("../")
# Custom
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from helpers.utils import imshow
from data_augmentation import * 
random.seed(999)


class DataIterator(object):
    def __init__(self, data_path, transformation_params, mode='train', 
        batch_size = 2, num_threads = 4, max_imgs_in_ram = 500):
        self.data_path = data_path
        self.mode = mode
        self.batch_size = batch_size
        self.num_threads = num_threads
        self.iter_over = False
        self.image_volume = np.array([])
        self.label_volume = np.array([])
        self.weight_volume = np.array([])
        self.file_name = []
        self.n_imgs_in_ram = 0
        self.num_imgs_obt = 0
        self.max_imgs_in_ram = max_imgs_in_ram
        self.files_accessed = []
        self.file_access_lock = threading.Lock()
        self.data_access_lock = threading.Lock()
        self.done_event = threading.Event()
        self.iter_over_for_thread = {}
        self.GetFilePaths()
        self.transformation_params = transformation_params

        for t_i in range(0,num_threads):
            t = threading.Thread(target = ThreadWorker, args = (weakref.ref(self),))
            t.setDaemon(True)
            t.start()
            self.iter_over_for_thread[t.name] = False

    def GetFilePaths(self):
        # Comment to train on ES or ED phase instead on joint ED+ES model
        if self.mode == 'train': 
            train_files = [os.path.join(self.data_path,'train_set',f) 
                        for f in os.listdir(os.path.join(self.data_path,'train_set'))]
            self.files = train_files

            # valid_files = [os.path.join(self.data_path,'validation_set',f) 
            #             for f in os.listdir(os.path.join(self.data_path,'validation_set'))]
            # self.files = train_files + valid_files

            # Uncomment to train on only ES phase images  
            # self.files = glob.glob(os.path.join(self.data_path,'train_set', '*ES*'))

            # Uncomment to train on only ED phase images  
            # self.files = glob.glob(os.path.join(self.data_path,'train_set', '*ED*'))

        elif self.mode == 'valid':  
            valid_files = [os.path.join(self.data_path,'validation_set',f) 
                        for f in os.listdir(os.path.join(self.data_path,'validation_set'))]
            self.files = valid_files
            # test_files = [os.path.join(self.data_path,'test_set',f) 
            #             for f in os.listdir(os.path.join(self.data_path,'test_set'))]
            # self.files = test_files

            # Uncomment to train on only ES phase images  
            # self.files = glob.glob(os.path.join(self.data_path,'validation_set', '*ES*'))

            # Uncomment to train on only ED phase images  
            # self.files = glob.glob(os.path.join(self.data_path,'validation_set', '*ED*'))   


        np.random.shuffle(self.files)
        self.steps = len(self.files)//self.batch_size

    def PopFilePath(self):
        if self.mode == 'train' or 'valid':
            if len(self.files) > 0:
                return self.files.pop()
            else:
                return None
        else:
            print("Got unknown value for mode, supported values are : 'train', 'valid' ", self.mode)
            raise 1

    def ExtractProcessedData(self, path):
        data = h5py.File(path,'r')
        file_name = os.path.basename(path)
        # print (file_name)
        # Preprocessing of Input Image and Label
        patch_img, patch_gt, patch_wmap = PreProcessData(file_name, data, self.mode, self.transformation_params)
        return patch_img, patch_gt, patch_wmap, file_name

    def GetNextBatch(self):
        temp_count = 0
        while True:
            with self.data_access_lock:
                image_batch, self.image_volume = np.split(self.image_volume,[self.batch_size])
                label_batch, self.label_volume = np.split(self.label_volume,[self.batch_size])
                weight_batch, self.weight_volume = np.split(self.weight_volume,[self.batch_size])
                file_batch, self.file_name = self.file_name[:self.batch_size], self.file_name[self.batch_size:] 

                num_imgs_obt = image_batch.shape[0]
                self.n_imgs_in_ram = self.image_volume.shape[0]

            if ((sum(x == True for x in self.iter_over_for_thread.values()) == self.num_threads) and (self.n_imgs_in_ram == 0)):
                self.iter_over = True
                return None, None, None, None, None

            if (num_imgs_obt > 0) or self.iter_over :
                if (num_imgs_obt != self.batch_size) and temp_count <=3 :
                    time.sleep(2)
                    temp_count += 1
                else:
                    break
        batch_class_weights = GetAvgbatchClassWeights(label_batch, scale=1, 
            label_ids=range(self.transformation_params['n_labels']), assign_equal_wt=False)
        # print (image_batch.shape, label_batch.shape)
        return image_batch, label_batch, weight_batch, batch_class_weights, file_batch

    def Reset(self):
        self.image_volume = np.array([])
        self.label_volume = np.array([])
        self.weight_volume = np.array([])
        self.n_imgs_in_ram = self.image_volume.shape[0]
        self.files = []
        self.GetFilePaths()
        self.files_accessed = []
        for key in self.iter_over_for_thread:
            self.iter_over_for_thread[key] = False  
        self.iter_over = False

    def __del__(self):
        print(' Thread exited ')
        self.done_event.set()

# Functions
def ThreadWorker(weak_self):
    self = weak_self()
    name  = threading.current_thread().name

    while not self.done_event.is_set():
        if (self.iter_over == False) and (self.n_imgs_in_ram < self.max_imgs_in_ram):
            with self.file_access_lock:
                input_path = self.PopFilePath()

                if (input_path is not None) and (input_path not in self.files_accessed):
                    self.files_accessed.append(input_path)
                    image, label, wmap, file_name = self.ExtractProcessedData(input_path)

                    with self.data_access_lock:
                        if self.image_volume.size != 0  and self.label_volume.size != 0:
                            try:
                                self.image_volume = np.vstack([self.image_volume, image])
                                self.label_volume = np.vstack([self.label_volume, label])
                                self.weight_volume = np.vstack([self.weight_volume, wmap])
                                self.file_name.append(file_name)
                            except Exception as e:
                                print(str(e))
                                self.image_volume = np.array([])
                                self.label_volume = np.array([])
                                self.weight_volume = np.array([])
                                self.file_name = []
                                print('Image queue shape: ' + str(self.image_volume.shape))
                                print('Image slice shape: ' + str(image.shape))
                                print('Label queue shape: ' + str(self.label_volume.shape))
                                print('Label slice shape: ' + str(label.shape))
                        else:
                            self.image_volume = image
                            self.label_volume = label
                            self.weight_volume = wmap
                            self.file_name.append(file_name)
                        self.n_imgs_in_ram = self.image_volume.shape[0]
                
                elif input_path == None:
                    self.iter_over_for_thread[name] = True


if __name__ == '__main__':
    # Test Routinue to check your threaded dataloader
    # ACDC dataset has 4 labels: LV, RV, MYO, Background
    n_labels = 4
    data_path = '../../processed_acdc_dataset/hdf5_files'
    batch_size = 1
    # Data Augmentation Parameters
    # Set patch extraction parameters
    size1 = (128, 128)
    patch_size = size1
    mm_patch_size = size1
    max_size = size1
    train_transformation_params = {
        'patch_size': patch_size,
        'mm_patch_size': mm_patch_size,
        'add_noise': ['gauss', 'none1', 'none2'],
        'rotation_range': (-5, 5),
        'translation_range_x': (-5, 5),
        'translation_range_y': (-5, 5),
        'zoom_range': (0.8, 1.2),
        'do_flip': (False, False),
        }
    valid_transformation_params = {
        'patch_size': patch_size,
        'mm_patch_size': mm_patch_size}
    transformation_params = { 'train': train_transformation_params,
                              'valid': valid_transformation_params,
                              'n_labels': n_labels,
                              'data_augmentation': False,
                              'full_image': False,
                              'data_deformation': False,
                              'data_crop_pad': max_size}

    train_iterator = DataIterator(data_path, transformation_params, mode = 'train', batch_size = batch_size, num_threads=4, max_imgs_in_ram = 10)
    print (train_iterator.steps)
    for i in range(1000):
        input_batch, target_batch, wmap_batch, batch_class_weights, file_batch = train_iterator.GetNextBatch()
        if type(input_batch) is np.ndarray:
            print (input_batch.shape, target_batch.shape, batch_class_weights)
            print (file_batch)
            imshow(input_batch[0,:,:,0],  target_batch[0,:,:], wmap_batch[0,:,:], input_batch[0,:,:,0],  target_batch[0,:,:], wmap_batch[0,:,:])
            # import seaborn as sns
            # fig = plt.figure(figsize=(12,12))
            # r = sns.heatmap(wmap_batch[0,:,:], cmap='nipy_spectral')
            # plt.axis('off') 
            # plt.savefig('test.png')

            # n=3
            # args = [input_batch[0,:,:,0],  target_batch[0,:,:], wmap_batch[0,:,:]]
            # cmap=['gray', 'gray', 'nipy_spectral']
            # grid_kws = {"height_ratios": (.9, .05), "hspace": .3}
            # plt.figure(figsize=(n*5,10))
            # for i in range(n):
            #     # f, (ax, cbar_ax) = plt.subplots(2, 3, gridspec_kw=grid_kws)
            #     plt.subplot(1,n,i+1,  gridspec_kw=grid_kws)
            #     # if i==n-1:
            #     sns.heatmap(args[i], cmap=cmap[i], cbar_kws={"orientation": "horizontal"})    
            #     plt.imshow(args[i], cmap[i])
            #     plt.axis('off') 
            # # plt.show()
            # plt.savefig('test1.png')



            # fig, (ax1, ax2, ax3) = plt.subplots(figsize=(12, 3), ncols=3)
            # size = fig.get_size_inches()*fig.dpi # get fig size in pixels
            
            # imgf = ax1.imshow(args[0], cmap=cmap[0], interpolation='none')
            # fig.colorbar(imgf, ax=ax1)
            # plt.text(0.5, -0.1, "(a) Image", \
            #       horizontalalignment='center', verticalalignment='center', \
            #       transform=ax1.transAxes) 
            # ax1.axis('off') 

            # gtf = ax2.imshow(args[1], cmap=cmap[1], interpolation='none')
            # fig.colorbar(gtf, ax=ax2)
            # plt.text(0.5, -0.1, "(b) Ground Truth", \
            #       horizontalalignment='center', verticalalignment='center', \
            #       transform=ax2.transAxes) 
            # ax2.axis('off') 

            # wmapf = ax3.imshow(args[2], cmap=cmap[2], interpolation='none')
            # fig.colorbar(wmapf, ax=ax3)
            # plt.text(0.5, -0.1, "(c) Spatial Weight Map", \
            #       horizontalalignment='center', verticalalignment='center', \
            #       transform=ax3.transAxes) 
            # ax3.axis('off') 

           
            # plt.tight_layout()
            # plt.show()
            # plt.savefig("wmap.pdf", dpi=150)
            # plt.savefig("wmap.eps", dpi=150)
            # break            
                # svg backend currently ignores the dpi
            # r.set_title("Heatmap of Flight Density from 1949 to 1961")
            # sns.heatmap(wmap_batch[0,:,:], annot=False,  linewidths=.5)

            # imshow(input_batch[0,:,:,0],  target_batch[0,:,:], wmap_batch[0,:,:], cmap=['gray', 'gray', 'nipy_spectral'], axis_off = True)
            # sns.heatmap(wmap_batch[0,:,:], annot=True,  linewidths=.5)
            # if input_batch.shape[0] == 0:
            #   print input_batch.shape
            #   import pdb; pdb.set_trace();
        if train_iterator.iter_over == True:
            print ('iterator over')
            train_iterator.Reset()
            time.sleep(4)