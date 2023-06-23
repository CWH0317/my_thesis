import os
import random
from shutil import copy

'''
Data/
    train_images/
                train/
                    img1, img2, img3, ......
    
    train_masks/
                train/
                    msk1, msk, msk3, ......
                    
    val_images/
                val/
                    img1, img2, img3, ......                
    val_masks/
                val/
                    msk1, msk, msk3, ......
      
    test_images/
                test/
                    img1, img2, img3, ......    
                    
    test_masks/
                test/
                    msk1, msk, msk3, ......
'''
def split_dataset_random(datasetlist, train_set_Proportion, val_set_Proportion, test_set_Proportion):
    datasetlist = datasetlist

    datasetlist_len = len(datasetlist)
    datasetlist_index = [*range(0, datasetlist_len)]
    datasetlist_index_random = datasetlist_index.copy()
    random.shuffle(datasetlist_index_random)

    train_set_Proportion = train_set_Proportion # 0 ~ 1

    train_set_len = int(round(datasetlist_len * train_set_Proportion))
    val_set_len = int(round(datasetlist_len * val_set_Proportion))
    test_set_len = datasetlist_len - train_set_len - val_set_len

    datasetlist_random = []

    for random_idx in datasetlist_index_random:
        datasetlist_random.append(datasetlist[random_idx])

    train_set_list = datasetlist_random[0:train_set_len]
    val_set_list = datasetlist_random[train_set_len:(train_set_len+val_set_len)]
    test_set_list = datasetlist_random[(train_set_len+val_set_len):datasetlist_len]
    
    return train_set_list, val_set_list, test_set_list

def copy_file(filelist, file_path, save_dir):
    
    for filename in filelist:
        from_path = os.path.join(file_path, filename)
        to_path = os.path.join(save_dir, filename)
        copy(from_path, to_path)

# 取出 images 與 masks 資料夾內相同檔名的檔案列表
image_list_path = r"H:/UNet_tensorflow/UNet_576x768_20230222/Data/Data3_576x768_oriplusolt_image"
mask_list_path = r"H:/UNet_tensorflow/UNet_576x768_20230222/Data/Data3_576x768_oriplusolt_mask_rev"
image_list = os.listdir(image_list_path)
mask_list = os.listdir(mask_list_path)
datasetlist = list(set(image_list)&set(mask_list))

# 依設置之比例配置訓練集及驗證集
train_set_Proportion = 0.8 # 0 ~ 1
val_set_Proportion = 0.1 # 0 ~ 1
test_set_Proportion = 1 - train_set_Proportion - val_set_Proportion # 0 ~ 1

train_set_list, val_set_list, test_set_list = split_dataset_random(datasetlist, train_set_Proportion, val_set_Proportion, test_set_Proportion)

# 建立資料夾 for 訓練集及測試集
train_images_path = r"H:/UNet_tensorflow/UNet_576x768_20230222/Data3/train_images/train"
if not os.path.isdir(train_images_path):
    os.makedirs(train_images_path)
    
train_masks_path = r"H:/UNet_tensorflow/UNet_576x768_20230222/Data3/train_masks/train"
if not os.path.isdir(train_masks_path):
    os.makedirs(train_masks_path)
    
val_images_path = r"H:/UNet_tensorflow/UNet_576x768_20230222/Data3/val_images/val"
if not os.path.isdir(val_images_path):
    os.makedirs(val_images_path)
    
val_masks_path = r"H:/UNet_tensorflow/UNet_576x768_20230222/Data3/val_masks/val"
if not os.path.isdir(val_masks_path):
    os.makedirs(val_masks_path)
    
test_images_path = r"H:/UNet_tensorflow/UNet_576x768_20230222/Data3/test_images/test"
if not os.path.isdir(test_images_path):
    os.makedirs(test_images_path)
    
test_masks_path = r"H:/UNet_tensorflow/UNet_576x768_20230222/Data3/test_masks/test"
if not os.path.isdir(test_masks_path):
    os.makedirs(test_masks_path)
    
# train
copy_file(train_set_list, image_list_path, train_images_path)
copy_file(train_set_list, mask_list_path, train_masks_path)
# val
copy_file(val_set_list, image_list_path, val_images_path)
copy_file(val_set_list, mask_list_path, val_masks_path)
# test
copy_file(test_set_list, image_list_path, test_images_path)
copy_file(test_set_list, mask_list_path, test_masks_path)
