import cv2
import os
from shutil import copy
import tensorflow as tf
import numpy as np
from PIL import Image

def predict_image(modelpath, img):
    # Load the pre-trained model
    model = tf.keras.models.load_model(modelpath, compile=False)

    backimg = Image.new('RGB', (768, 576), color = 'black')
    #img = img.resize((768, 576)) # Example resizing
    img = np.array(img) / 255.0 # Example normalization
    img = np.expand_dims(img, axis=0) # Add batch dimension

    # Make predictions
    mask = model.predict(img)[0]

    # Post-process the segmentation mask
    threshold = 0.6 # Example threshold
    mask = (mask > threshold).astype(np.uint8) # Example thresholding

    # Display the segmentation mask
    prediction = Image.fromarray(mask * 255) # Example visualization
    backimg.paste(prediction, (0, 0), prediction)
    
    return backimg

def crop_img(img, y, x, h, w):
    cropped = img[y:(y + h), x:(x + w)]
    return cropped

def padding_img(cropped_img, py, px, y, x):
    img = Image.new('RGB', (px, py), color = 'black')
    position = (x, y)
    img.paste(cropped_img, position)
    return img
    
    
if __name__ == '__main__':
    # set crop size
    y = int((600 - 576) / 2)
    x = int((800 - 768) / 2)
    h =576
    w = 768
    py = 600
    px = 800

    # set deep learning model path
    modelpath = 'train_on_cpu/cwh_ML/Data2_batchsize8_30epoches_UNet3Plus_load_from_disk.hdf5'

    # set original image path
    ori_image_path = r"H:\CWH_thesis_experimental\PD_V_F_SCBT\ori_image\ex1_300_20_F_V_2"
    ori_image_list = os.listdir(ori_image_path)
    ori_image_list = sorted(ori_image_list)

    # set the folder path to store cropped and predicted image
    cropped_predicted_image_path = ori_image_path + "_Predicted_img"
    if not os.path.isdir(cropped_predicted_image_path):
        os.makedirs(cropped_predicted_image_path)

    # crop and predict the images in the folder you set.
    for img_name in ori_image_list:
        img = cv2.imread(os.path.join(ori_image_path, img_name))
        cropped = crop_img(img, y, x, h, w)
        #cv2.imwrite(os.path.join(cropped_image_path, img_name), cropped)
        pil_img = cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(pil_img)
        # predict image
        prediction = predict_image(modelpath, pil_img)
        padding_prediction = padding_img(prediction, py, px, y, x)
        # save predicted image
        predicted_folder_path = cropped_predicted_image_path
        predicted_file_path = "Predicted_" + img_name
        padding_prediction.save(os.path.join(predicted_folder_path, predicted_file_path))   
        #prediction.show()