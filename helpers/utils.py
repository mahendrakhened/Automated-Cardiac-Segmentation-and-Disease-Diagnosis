import pickle
import os, re, sys
import shutil
import nibabel as nib
from scipy.fftpack import fftn, ifftn
import numpy as np
try:
    import matplotlib
    # matplotlib.use("TkAgg")
    import matplotlib.pyplot as plt
    from matplotlib import animation
except:
    print ('matplotlib not imported')


def progress_bar(curr_idx, max_idx, time_step, repeat_elem = "_"):
    max_equals = 55
    step_ms = int(time_step*1000)
    num_equals = int(curr_idx*max_equals/float(max_idx))
    len_reverse =len('Step:%d ms| %d/%d ['%(step_ms, curr_idx, max_idx)) + num_equals
    sys.stdout.write("Step:%d ms|%d/%d [%s]" %(step_ms, curr_idx, max_idx, " " * max_equals,))
    sys.stdout.flush()
    sys.stdout.write("\b" * (max_equals+1))
    sys.stdout.write(repeat_elem * num_equals)
    sys.stdout.write("\b"*len_reverse)
    sys.stdout.flush()
    if curr_idx == max_idx:
        print('\n')

def read_fft_volume(data4D, harmonic=1):
    zslices = data4D.shape[2]
    tframes = data4D.shape[3]
    data3d_fft = np.empty((data4D.shape[:2]+(0,)))
    for slice in range(zslices):
        ff1 = fftn([data4D[:,:,slice, t] for t in range(tframes)])
        fh = np.absolute(ifftn(ff1[harmonic, :, :]))
        fh[fh < 0.1 * np.max(fh)] = 0.0
        image = 1. * fh / np.max(fh)
        # plt.imshow(image, cmap = 'gray')
        # plt.show()
        image = np.expand_dims(image, axis=2)
        data3d_fft = np.append(data3d_fft, image, axis=2)
    return data3d_fft

def save_data(data, filename, out_path):
    out_filename = os.path.join(out_path, filename)
    with open(out_filename, 'wb') as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
    print ('saved to %s' % out_filename)

def load_pkl(path):
    with open(path) as f:
        obj = pickle.load(f)
    return obj

def imshow(*args,**kwargs):
    """ Handy function to show multiple plots in on row, possibly with different cmaps and titles
    Usage:
    imshow(img1, title="myPlot")
    imshow(img1,img2, title=['title1','title2'])
    imshow(img1,img2, cmap='hot')
    imshow(img1,img2,cmap=['gray','Blues']) """
    cmap = kwargs.get('cmap', 'gray')
    title= kwargs.get('title','')
    axis_off = kwargs.get('axis_off','')
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
            if axis_off: 
              plt.axis('off')  
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

def swapaxes_slv(vol):
    return np.swapaxes(np.swapaxes(vol,0,2),0,1)

def reorder_vol(data):
    ED_GT = swapaxes_slv(data[0][1])
    ED_PD = swapaxes_slv(data[0][2])
    ES_GT = swapaxes_slv(data[1][1])
    ES_PD = swapaxes_slv(data[1][2])
    return (ED_GT, ES_GT, ED_PD, ES_PD)
