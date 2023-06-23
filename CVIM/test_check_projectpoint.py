import cv2
import pandas as pd
import numpy as np

# read by default 1st sheet of an excel file
dataframe1 = pd.read_excel("H:/CWH_Opensees/OpenSees/turbine_simulate_20230426_novibration/OutputFiles/rotate/real_node_rotate.xlsx")
#print(dataframe1)

x_mm_cc = dataframe1.iloc[:, 0].values
y_mm_cc = dataframe1.iloc[:, 1].values
z_mm_cc = dataframe1.iloc[:, 2].values

#print(y_m.shape)

objectPoints = np.zeros((x_mm_cc.shape[0],3),dtype=float)

for i in range(x_mm_cc.shape[0]):
    objectPoints[i][0] = x_mm_cc[i]/1000
    objectPoints[i][1] = y_mm_cc[i]/1000
    objectPoints[i][2] = z_mm_cc[i]/1000
''''''
#print(objectPoints)

cmat = np.zeros((3, 3), dtype=float)
cmat[0, 0] = 1111.1111
cmat[1, 1] = 1111.1111
cmat[0, 2] = 400.
cmat[1, 2] = 300.
cmat[2, 2] = 1
dvec = np.zeros((5,1), dtype=float)

'''
K
<Matrix 3x3 (5333.3335,    0.0000, 1920.0000)
            (   0.0000, 5333.3335, 1080.0000)
            (   0.0000,    0.0000,    1.0000)>
RT
<Matrix 3x4 ( 0.9397, -0.3420,  0.0000,  -0.0000)
            (-0.0000, -0.0000, -1.0000,  -0.0000)
            ( 0.3420,  0.9397, -0.0000, 300.0001)>
'''
'''
cam_cal_extrinsic_parameter = np.load('D:/Lab/CWH_thesis/code/ex1_300_10_analysis/4K_GT/point_position_pixel_cam_cal_extrinsic_parameter.npz')
r44 = cam_cal_extrinsic_parameter['r44']
'''
r44 = np.zeros((4, 4), dtype=float)
r44[0][0] = 0.9397
r44[0][1] = -0.3420
r44[0][2] = 0.0000
r44[0][3] = -0.0000
r44[1][0] = -0.0000
r44[1][1] = -0.0000
r44[1][2] = -1.0000
r44[1][3] = -0.0000
r44[2][0] = 0.3420
r44[2][1] = 0.9397
r44[2][2] = -0.0000
r44[2][3] = 300.0001
r44[3][0] = 0
r44[3][1] = 0
r44[3][2] = 0
r44[3][3] = 1

print(r44)
rvec, rvecjoc = cv2.Rodrigues(r44[0:3,0:3])
tvec = r44[0:3,3]
print(rvec, tvec)

imagePoints, jacobian = cv2.projectPoints(objectPoints, rvec, tvec, cmat, dvec)
imagePoints = imagePoints.reshape((imagePoints.shape[0], 2))
print(imagePoints)
print(imagePoints.shape)

imgpoints_dict = {'p3':imagePoints}
np.savez(r'H:\CWH_thesis_experimental\OpenSees_NV\ex1_300_20_800x600_NV_proj_point_position.npz', **imgpoints_dict)
r44_dict = {'r44':r44}
np.savez(r'H:\CWH_thesis_experimental\OpenSees_NV\ex1_300_20_800x600_NV_proj_point_position_extrinsic_parameter.npz', **r44_dict)
imgpoints_dict_csv = {'p3':list(imagePoints)}
df_point_position = pd.DataFrame(imgpoints_dict_csv)
print(df_point_position)
df_point_position.to_csv(r"H:\CWH_thesis_experimental\OpenSees_V\ex1_300_20_800x600_V_proj_point_position.csv")