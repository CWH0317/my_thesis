import tensorflow as tf
from PIL import Image
import numpy as np
import cv2
import os
import random

def compute_iou(y_pred, y_true):
    y_pred = np.round(y_pred)
    y_true = np.round(y_true)
    intersection = np.logical_and(y_pred, y_true).sum()
    union = np.logical_or(y_pred, y_true).sum()
    iou = (intersection + 1e-10) / (union + 1e-10)
    return iou

def cal_iou(img_path, mask_path, model_path):
    # Load your image and mask

    img = np.array(Image.open(img_path).convert('RGB'))
    mask = np.array(Image.open(mask_path).convert('L'))

    # Preprocess your image and mask (resize, normalize, etc.)
    #img = cv2.resize(img, (768, 576))
    img = img / 255.0
    #mask = cv2.resize(mask, (768, 576))
    mask = mask / 255.0

    # Expand dimensions to match model input shape
    img_input = np.expand_dims(img, 0)
    mask_input = np.expand_dims(mask, -1)

    # Predict segmentation mask
    model = tf.keras.models.load_model(model_path, compile=False)
    prediction = model.predict(img_input)[0,:,:,0] > 0.5

    # Compute IoU
    iou = compute_iou(prediction, mask_input[:,:,0])
    return iou
    #print("IoU is: ", iou)
''''''

imgs_file_path = 'H:/UNet_tensorflow/UNet_576x768_20230222/Data3/test_images/test/'
#imgs_file_path = "H:/UNet_tensorflow/UNet_576x768_20230222/Data/Data3_overlaptower_image_ren"
masks_file_path = 'H:/UNet_tensorflow/UNet_576x768_20230222/Data3/test_masks/test/'
#masks_file_path = "H:/UNet_tensorflow/UNet_576x768_20230222/Data/Data3_overlaptower_mask_rev_ren"
imgs_filelist = os.listdir(imgs_file_path)
masks_filelist = os.listdir(masks_file_path)
#print(imgs_filelist)
pick_num = 10
random_list = []
for i in range(pick_num):
    random_list.append(random.randint(0,len(imgs_filelist)-1))
#print(random_list)

#img_path = 'H:/UNet_tensorflow/UNet_576x768_20230222/Data3/test_images/test/IMG_800x600_300_0_0188_olt.png'
#mask_path = 'H:/UNet_tensorflow/UNet_576x768_20230222/Data3/test_masks/test/IMG_800x600_300_0_0188_olt.png'

#model_path = 'UNet3Plus_load_from_disk.hdf5'
#model_path = 'Data3_UNet3Plus_load_from_disk.hdf5'
model_path = 'train_on_cpu/cwh_ML/Data2_batchsize8_30epoches_UNet3Plus_load_from_disk.hdf5'
#model_path = 'train_on_cpu/cwh_ML/Data3_batchsize8_6epoches_UNet3Plus_load_from_disk.hdf5'

iou_list = []
avgiou = 0
for i, ele in enumerate(random_list):
    img_path = os.path.join(imgs_file_path, imgs_filelist[ele])
    mask_path = os.path.join(masks_file_path, imgs_filelist[ele])
    iou = cal_iou(img_path, mask_path, model_path)
    iou_list.append(iou)
    avgiou = avgiou + iou
    print("Image%s's IoU is: "%str(i+1), iou)
avgiou = avgiou / pick_num
print()
print("Average IoU is: ", avgiou)