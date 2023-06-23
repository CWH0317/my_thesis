import os 
import numpy as np
import matplotlib.pyplot as plt
import cv2


file_path = r"H:\blender_render\chessboard_calib"
image_paths = os.listdir(file_path)
#print(image_paths)

# 终止标准
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

w = 12
h = 7

# 准备对象点, 如 (0,0,0), (1,0,0), (2,0,0) ....,(w,5,0)
objp = np.zeros((w*h,3), np.float32)
objp[:,:2] = np.mgrid[0:h,0:w].T.reshape(-1,2)

# 用于存储所有图像对象点与图像点的矩阵
objpoints = [] # 在真实世界中的 3d 点 
imgpoints = [] # 在图像平面中的 2d 点


for fname in image_paths:
    img = cv2.imread(os.path.join(file_path, fname))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 找到棋盘上所有的角点
    ret, corners = cv2.findChessboardCorners(gray, (h,w), None)

    # 如果找到了，便添加对象点和图像点(在细化后)
    if ret == True:
        objpoints.append(objp)

        corners2 = cv2.cornerSubPix(gray,corners, (11,11), (-1,-1), criteria)
        imgpoints.append(corners)

        # 绘制角点
        cv2.drawChessboardCorners(img, (h,w), corners2, ret)
        cv2.imshow('img', img)
        cv2.waitKey(1000)

cv2.destroyAllWindows()

ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

print(mtx)
print(dist[0])
intrinsic_parameters = {'cmat' : mtx, 'dvec' : dist[0]}
np.savez(os.path.join(file_path, "intrinsic_parameters.npz"), **intrinsic_parameters)

img = cv2.imread(os.path.join(file_path, image_paths[0]))
h,  w = img.shape[:2]
newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w,h), 1, (w,h))
# 矫正
dst = cv2.undistort(img, mtx, dist, None, newcameramtx)

# 裁切图像
x, y, w, h = roi
dst = dst[y:y+h, x:x+w]
cv2.imwrite('calibresult.png', dst)

# 矫正
mapx, mapy = cv2.initUndistortRectifyMap(mtx, dist, None, newcameramtx, (w,h), 5)
dst = cv2.remap(img, mapx, mapy, cv2.INTER_LINEAR)

# 裁切图像
x, y, w, h = roi
dst = dst[y:y+h, x:x+w]
cv2.imwrite('calibresult.png', dst)

mean_error = 0
for i in range(len(objpoints)):
    imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
    error = cv2.norm(imgpoints[i], imgpoints2, cv2.NORM_L2)/len(imgpoints2)
    mean_error += error

print( "total error: {}".format(mean_error/len(objpoints)) )