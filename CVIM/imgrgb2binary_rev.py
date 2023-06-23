import cv2
import os

def img_rgb2binary(img):
    
    img = cv2.imread(os.path.join(input_filepath, img_name))
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) # 轉換前，都先將圖片轉換成灰階色彩
    #ret, output1 = cv2.threshold(img_gray, 127, 255, cv2.THRESH_BINARY)     # 如果大於 127 就等於 255，反之等於 0。
    ret, output2 = cv2.threshold(img_gray, 127, 255, cv2.THRESH_BINARY_INV) # 如果大於 127 就等於 0，反之等於 255。
    #ret, output3 = cv2.threshold(img_gray, 127, 255, cv2.THRESH_TRUNC)      # 如果大於 127 就等於 127，反之數值不變。
    #ret, output4 = cv2.threshold(img_gray, 127, 255, cv2.THRESH_TOZERO)     # 如果大於 127 數值不變，反之數值等於 0。
    #ret, output5 = cv2.threshold(img_gray, 127, 255, cv2.THRESH_TOZERO_INV) # 如果大於 127 等於 0，反之數值不變。
    return output2
    
input_filepath = "H:/CWH_thesis_experimental/GT_NV/ori_image/ex1_BW_300_20_NV"
output_filepath = "H:/CWH_thesis_experimental/GT_NV/ori_image/ex1_BW_300_20_NV_rev"
imgnamelist = os.listdir(input_filepath)

for img_name in imgnamelist:
    img = cv2.imread(os.path.join(input_filepath, img_name))
    output = img_rgb2binary(img)
    cv2.imwrite(os.path.join(output_filepath, img_name), output)