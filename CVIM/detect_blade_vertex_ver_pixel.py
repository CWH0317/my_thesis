import os
import numpy as np
import cv2
from numba import njit
import math
import time
from scipy import signal
import matplotlib.pyplot as plt
import scipy
import pandas as pd
@njit 
def c_to_p_angle(px, py, cx, cy): # 風機中心至任一點之方位角 
    dx = px - cx
    dy = py - cy 
    if (py == cy) and (px >= cx):
        angle = 0.0
    elif (py == cy) and (px < cx):
        angle = 180.0
    elif (py >= cy) and (px == cx):
        angle = 90.0
    elif (py < cy) and (px == cx):
        angle = 270.0
    elif (px > cx) and (py > cy): # first quadrant
        angle = math.atan((dy)/(dx)) * 180 / math.pi
    elif (px < cx) and (py > cy): # second quadrant
        angle = math.atan((dy)/(dx)) * 180 / math.pi + 180
    elif (px < cx) and (py < cy): # third quadrant
        angle = math.atan((dy)/(dx)) * 180 / math.pi + 180
    elif (px > cx) and (py < cy): # fourth quadrant
        angle = math.atan((dy)/(dx)) * 180 / math.pi + 360
    return angle

@njit 
def cal_distance(x1, y1, x2, y2):
    dis = ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5
    return dis

#@njit 
def find_nearest_point(last_p, new_p):
    p_d_1 = (new_p[0][0], new_p[0][1], cal_distance(last_p[0], last_p[1], new_p[0][0], new_p[0][1]))
    p_d_2 = (new_p[1][0], new_p[1][1], cal_distance(last_p[0], last_p[1], new_p[1][0], new_p[1][1]))
    p_d_3 = (new_p[2][0], new_p[2][1], cal_distance(last_p[0], last_p[1], new_p[2][0], new_p[2][1]))
    new_p_d = (p_d_1, p_d_2, p_d_3)
    min_dis = np.asarray(sorted(new_p_d, key=lambda s: s[2]))
    return min_dis[0]

@njit 
def find_windturbine_actual_center(coodr):
    ((x1, y1), (x2, y2), (x3, y3)) = coodr
    actual_cx = (x1 + x2 + x3) / 3
    actual_cy = (y1 + y2 + y3) / 3
    actual_center = (actual_cx, actual_cy)
    return actual_center

@njit 
def find_windturbine_contour_center(img):
    img_h = img.shape[0]
    img_w = img.shape[1]
    contour_pixel = []
    area = 0 
    all_cx = 0
    all_cy = 0
    for i in range(img_h):
        for j in range(img_w):
            if img[i,j].all() > 0:
                area = area + 1
                all_cy = all_cy + i
                all_cx = all_cx + j
                if img[i-1,j-1].all() > 0 and img[i,j-1].all() > 0 and img[i+1,j-1].all() > 0 and img[i-1,j].all() > 0 and img[i+1,j].all() > 0 and img[i-1,j+1].all() > 0 and img[i,j+1].all() > 0 and img[i+1,j+1].all() > 0:
                    pass
                else:
                    contour_pixel.append((j, i))
    cy = all_cy / area
    cx = all_cx / area
    return contour_pixel, cx, cy, area

def find_blade_vertex(img):

    contour_pixel, cx, cy, _  = find_windturbine_contour_center(img)

    # 算出contour每個點到 cx, cy 的距離
    contour_pixel_angle_dis = []
    for p in contour_pixel:
        angle = c_to_p_angle(p[0], p[1], cx, cy)
        dis = cal_distance(p[0], p[1], cx, cy)
        contour_pixel_angle_dis.append((p[0], p[1], angle, dis))
    # 離 cx, cy 較遠的前 25 %
    sort_contour_pixel_angle_dis = sorted(contour_pixel_angle_dis, key=lambda s: s[3], reverse=True)
    percentage_of_contour_pixel_angle_dis = 0.25
    preselection_contour_pixel_angle_dis = int(len(contour_pixel_angle_dis) * percentage_of_contour_pixel_angle_dis)
    possible_vertex_angle_dis= sort_contour_pixel_angle_dis[0:preselection_contour_pixel_angle_dis]

    sortbyangle_possible_vertex_angle_dis = sorted(possible_vertex_angle_dis, key=lambda s: s[2])

    data_x = np.zeros(100 + len(sortbyangle_possible_vertex_angle_dis), dtype=float)
    data_y = np.zeros(100 + len(sortbyangle_possible_vertex_angle_dis), dtype=float)
    for i, ele in enumerate(sortbyangle_possible_vertex_angle_dis):
        data_x[i + 50] = ele[2]
        data_y[i + 50] = ele[3]
    # 照片尺寸大小會影響 order=int(120*percentage_of_contour_pixel_angle_dis，待測試，目前800x600使用120，4K使用800
    #max_index = signal.argrelextrema(data_y, np.greater, order=int(120*percentage_of_contour_pixel_angle_dis))[0]
    max_index = signal.argrelextrema(data_y, np.greater, order=int(len(possible_vertex_angle_dis)/8))[0]
    peak = data_y[max_index]  
    #print(f"peak:{peak}")
    '''
    # Plot all data
    plt.scatter(data_x,data_y,c="b")
    plt.xlabel("Bearing (°)", fontweight = "bold")  
    plt.ylabel("Distance (pixel)", fontweight = "bold")
    plt.title("Scatter of Bearing and Distance (each points on the contour to the center of the contour)", fontsize = 12, fontweight = "bold")
    # Plot peaks
    for index in max_index:
        plt.scatter(data_x[index],data_y[index],c="r")
    plt.show()
    '''
    f_blade_vertex = []
    for ele in sortbyangle_possible_vertex_angle_dis:
        if ele[3] in peak:
            f_blade_vertex.append(ele)
    #print(f_blade_vertex)

    correct_blade_vertex = f_blade_vertex.copy()
    if len(f_blade_vertex) > 3:
        keep = []
        for i in range(len(f_blade_vertex)-1):
            for j in range(i+1,len(f_blade_vertex),1):
                x1 = f_blade_vertex[i][0]
                y1 = f_blade_vertex[i][1]
                x2 = f_blade_vertex[j][0]
                y2 = f_blade_vertex[j][1]
                d = cal_distance(x1, y1, x2, y2)
                #print(d)
                keep.append([i,j,d])
        #print(keep)
        keep = sorted(keep, key=lambda s: s[2])
        #print(keep)
        for ele in keep:
            if ele[2] <= 50:
                if f_blade_vertex[ele[0]][3] >= f_blade_vertex[ele[1]][3]:
                    if f_blade_vertex[ele[1]] in correct_blade_vertex:
                        correct_blade_vertex.remove(f_blade_vertex[ele[1]])
                    else:
                        pass
                else:
                    if f_blade_vertex[ele[0]] in correct_blade_vertex:
                        correct_blade_vertex.remove(f_blade_vertex[ele[0]])
                    else:
                        pass
    return correct_blade_vertex

if __name__ == '__main__':

    point_position = {}
    #wt_rotate_center = {}
    #wt_rotate_center['blade_center'] = []
    #wt_rotate_center['average_blade_center'] = []
    #displacement = {}
    fileDir = r"H:\CWH_thesis_experimental\PD_V_NF_SCBT"
    predicted_img_path = os.path.join(fileDir, "Predicted_img")
    fileExt = r".png"
    filelist = sorted([os.path.join(predicted_img_path, _) for _ in os.listdir(predicted_img_path) if _.endswith(fileExt)])
    filelist_name = []

    checkcorner_image_path = os.path.join(fileDir, "pixel_level", "checkcorner")
    if not os.path.isdir(checkcorner_image_path):
        os.makedirs(checkcorner_image_path)

    for file in filelist:
        _, name = os.path.split(file)
        filelist_name.append(name)
        filelist_name = sorted(filelist_name)

    n = 1
    avg_actual_center_x = 0
    avg_actual_center_y = 0
    for i, filename in enumerate(filelist):
        img = cv2.imread(filename)
        p = find_blade_vertex(img)
        x1 = p[0][0]
        y1 = p[0][1]
        x2 = p[1][0]
        y2 = p[1][1]
        x3 = p[2][0]
        y3 = p[2][1]
        p_for_cal_center = ((x1, y1), (x2, y2), (x3, y3))

        #actual_center = find_windturbine_actual_center(p_for_cal_center)
        #print(actual_center)
        #wt_rotate_center['blade_center'].append(actual_center)
        #avg_actual_center_x = avg_actual_center_x + actual_center[0]
        #avg_actual_center_y = avg_actual_center_y + actual_center[1]
        
        #avg_center_x = avg_actual_center_x / (i+1)
        #avg_center_y = avg_actual_center_y / (i+1)
        #avg_center = (avg_center_x, avg_center_y)
        #print(avg_center)
        #wt_rotate_center['average_blade_center'].append(avg_center)

        print(p)
        print(n)
        print(filename)
        n = n + 1
        
        # if dict is empty
        if not point_position:
            point_position['p1'] = [(p[0][0], p[0][1])]
            point_position['p2'] = [(p[1][0], p[1][1])]
            point_position['p3'] = [(p[2][0], p[2][1])]
            #displacement['p1'] = []
            #displacement['p2'] = []
            #displacement['p3'] = []
        else:
            last_p1 = point_position['p1'][-1]
            np_1 = find_nearest_point(last_p1, p)
            point_position['p1'].append((np_1[0],np_1[1]))
            #displacement['p1'].append(np_1[2])
            
            last_p2 = point_position['p2'][-1]
            np_2 = find_nearest_point(last_p2, p)
            point_position['p2'].append((np_2[0],np_2[1]))
            #displacement['p2'].append(np_2[2])
            
            last_p3 = point_position['p3'][-1]
            np_3 = find_nearest_point(last_p3, p)
            point_position['p3'].append((np_3[0],np_3[1]))
            #displacement['p3'].append(np_3[2])
            
        x1 = int(point_position['p1'][-1][0])
        y1 = int(point_position['p1'][-1][1])
        x2 = int(point_position['p2'][-1][0])
        y2 = int(point_position['p2'][-1][1])
        x3 = int(point_position['p3'][-1][0])
        y3 = int(point_position['p3'][-1][1])
        
        img[y1, x1] = [255, 0, 0]
        cv2.circle(img, (x1, y1), 20, (255, 0, 0), 4)
        img[y2, x2] = [0, 255, 0]
        cv2.circle(img, (x2, y2), 20, (0, 255, 0), 4)
        img[y3, x3] = [0, 0, 255]
        cv2.circle(img, (x3, y3), 20, (0, 0, 255), 4)
        #print(filelist_name[i])
        
        cv2.imwrite(os.path.join(checkcorner_image_path ,filelist_name[i]),img)
    '''
    avg_actual_center_x = avg_actual_center_x / len(filelist)
    avg_actual_center_y = avg_actual_center_y / len(filelist)
    avg_actual_center = (avg_actual_center_x, avg_actual_center_y)
    
    wt_rotate_center['average_blade_center'] = []
    wt_rotate_center['average_blade_center'].append(avg_actual_center)
    '''
    #print()
    #print(avg_center)
    #print(avg_actual_center)
    #print()
    #print(displacement['p1'])
    #print(displacement['p2'])
    #print(displacement['p3'])
    #print()
    #print(max(displacement['p1']), displacement['p1'].index(max(displacement['p1'])))
    #print(max(displacement['p2']), displacement['p2'].index(max(displacement['p2'])))
    #print(max(displacement['p3']), displacement['p3'].index(max(displacement['p3'])))
    #print()
    #print(len(filelist))
    #print(len(displacement['p1']))
    #print(len(displacement['p2']))
    #print(len(displacement['p3']))
    print()
    print(point_position['p1'])
    print(point_position['p2'])
    print(point_position['p3'])
    print()
    print(np.array(point_position['p1'],dtype=float))
    print(np.array(point_position['p1'],dtype=float).shape)  
    
    np.savez(os.path.join(fileDir, "pixel_level", "point_position_pixel.npz"), **point_position)
    
    df_point_position = pd.DataFrame(point_position)
    #df_displacement = pd.DataFrame(displacement)
    #df_wt_rotate_center = pd.DataFrame(wt_rotate_center)

    #print(df_point_position)
    df_point_position.to_csv(os.path.join(fileDir, "pixel_level", "point_position_pixel.csv"))
    #print(df_displacement)
    #print(df_wt_rotate_center)
    #df_wt_rotate_center.to_csv("wt_rotate_center.csv")
