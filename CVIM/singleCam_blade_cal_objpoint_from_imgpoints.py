from math import cos, sin, tan, acos, asin, atan2, sqrt, pi
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
from scipy.optimize import least_squares
import os

def rvecTvecFromR44(r44):
    """
    Returns the rvec and tvec of the camera

    Parameters
    ----------
    r44 : TYPE np.array((4,4),dtype=float)
        The 4-by-4 form of camera extrinsic parameters
    Returns
    -------
    TYPE: tuple ([0]: np.array(3,dtype=float, [1]: np.array(3,dtype=float)))
    Returns the rvec and tvec of the camera 

    """
    rvec, rvecjoc = cv.Rodrigues(r44[0:3,0:3])
    tvec = r44[0:3,3]
    return rvec, tvec

def extrinsicR44ByCamposAndAim(campos, aim):
    """
    Calculates the 4-by-4 matrix form of extrinsic parameters of a camera according to camera position and a point it aims at.
    Considering the world coordinate X-Y-Z where Z is upward, 
    starting from an initial camera orientation (x,y,z) which is (X,-Z,Y), that y is downward (-Z), 
    rotates the camera so that it aims a specified point (aim)
    This function guarantee the camera axis x is always on world plane XY (i.e., x has no Z components)
    Example:
        campos = np.array([ -100, -400, 10],dtype=float)
        aim = np.array([0, -50, 100],dtype=float)
        r44Cam = extrinsicR44ByCamposAndAim(campos,aim)
        # r44Cam would be 
        # np.array([[ 0.961, -0.275,  0.000, -1.374],
        #           [ 0.066,  0.231, -0.971,  108.6],
        #           [ 0.267,  0.933,  0.240,  397.6],
        #           [ 0.000,  0.000,  0.000,  1.000]])
        
    Parameters
    ----------
    campos: TYPE np.array((3,3),dtype=float)
        camera position in the world coordinate 
    aim: TYPE np.array((3,3),dtype=float)
        the aim that the camera is aims at

    Returns
    -------
    TYPE: np.array((4,4),dtype=float)
        the 4-by-4 matrix form of the extrinsic parameters
    """
    # camera vector (extrinsic)
    vz_cam = aim - campos
    vy_cam = np.array([0,0,-1], dtype=np.float64)
    vz_cam = vz_cam / np.linalg.norm(vz_cam)
    vx_cam = np.cross(vy_cam, vz_cam)
    vy_cam = np.cross(vz_cam, vx_cam)
    vx_cam = vx_cam / np.linalg.norm(vx_cam)
    vy_cam = vy_cam / np.linalg.norm(vy_cam)
    vz_cam = vz_cam / np.linalg.norm(vz_cam)
    r44inv = np.eye(4, dtype=np.float64)
    r44inv[0:3, 0] = vx_cam[0:3]
    r44inv[0:3, 1] = vy_cam[0:3]
    r44inv[0:3, 2] = vz_cam[0:3]
    r44inv[0:3, 3] = campos[0:3]
    r44 = np.linalg.inv(r44inv)
    return r44

    
def bladePointByThetaAndDeflection(r_blade, r44_blade, theta, deflection): 
    """
    calculates the 3D coordinates of blade point 
    Example: 
        # the radius of the point on the blade is 55 meters
        r_blade = 55.0; 
        # blade faces towards -Y of global coord. (i.e., blade z is 0,-1,0)
        r44_blade = np.array([[-1,0,0,0],[0,0,-1,0],[0,-1,0,0],[0,0,0,1]],dtype=float)
        # blade rotates to the highest point
        theta = -90; 
        # deflection is 0.3 m along blade z axis
        deflection = 0.3
        # get the coordinate of peak (peakPoint)
        peakPoint = bladePeakByThetaAndDeflection(r_blade, r44_blade, theta, deflection)
    
    Parameters
    ----------
    r_blade : TYPE float
        radius of the blade point
    r44_blade : TYPE numpy.ndarray ((4,4), dtype=np.float)
        the extrinsic parameters of the blade in 4-by-4 matrix form
    theta : TYPE float 
        the angle (in degree) of the blade on the blade axes x and y (note: the blade y axis could be commonly downward. Think carefully about the blade axes according to the r44_blade)
    deflection : TYPE float
        the deflection of the blade along blade axis z 
    
    Returns
    -------
    TYPE (4, dtype=np.float) 
    Returns the 3D coordinate of the point (homogeneous coordinate).
    """
    bladePointLocal = np.array([r_blade * cos(theta / (180./pi)), \
                      r_blade * sin(theta / (180./pi)), deflection, 1], \
                      dtype=np.float64)
    r44inv_blade = np.linalg.inv(r44_blade)  
    bladePointGlobal = np.matmul(r44inv_blade, bladePointLocal.transpose())
    bladePointGlobal /= bladePointGlobal[3]
    return bladePointGlobal


def bladeImgPointByThetaAndDeflection(theta, deflection, r_blade, r44_blade, cmat, dvec, r44_cam):
    """
    calculates the image coordinates of blade point 

    Parameters
    ----------
    theta : TYPE float 
        the angle (in degree) of the blade on the blade axes x and y (note: the blade y axis could be commonly downward. Think carefully about the blade axes according to the r44_blade)
    deflection : TYPE float
        the deflection of the blade along blade axis z 
    r_blade : TYPE float 
        radius of the blade point
    r44_blade : TYPE numpy.ndarray ((4,4), dtype=np.float)
        the extrinsic parameters of the blade in 4-by-4 matrix form
    cmat : TYPE numpy.ndarray((3,3), dtype=np.float)
        camera matrix 
    dvec : TYPE numpy.ndarray(n, dtype=np.float)
        distortion vector
    r44_cam : TYPE numpy.ndarray((4,4),dtype=np.float)
        extrinsic parameters

    Returns
    -------
    TYPE (2, dtype=np.float) 
    Returns the image coordinate of the point .
    """
    bladeWorldPoint = bladePointByThetaAndDeflection(r_blade, r44_blade, theta, deflection)
    rvec, tvec = rvecTvecFromR44(r44_cam)
    bladeImagePoint, jacob = cv.projectPoints(bladeWorldPoint[0:3],                                               
                                              rvec, tvec, cmat, dvec)    
    bladeImagePoint = bladeImagePoint.reshape(2)
    return bladeImagePoint

def funBladeImgPointByThetaAndDeflection(x, r_blade, r44_blade, cmat, dvec, r44_cam, imgPoint):
    theta = x[0]
    deflection = x[1]   
    bladeImagePoint = bladeImgPointByThetaAndDeflection(theta, deflection, r_blade, r44_blade, cmat, dvec, r44_cam)   
    bladeImagePoint -= imgPoint 
    return bladeImagePoint

def bladeThetaAndDeflectionByImgPoint(bladeImagePoint, r_blade, r44_blade, cmat, dvec, r44_cam):
    minCost = 1e30
    bestTheta = -1;
    bestDeflection = -1;
    bestRes =[];
    for theta_i in range(10):
        initTheta = theta_i * 36.0 
        x0 = np.array((initTheta, 0),dtype=float)
        lbound = np.array([  0.0, -r_blade * 0.2])
        ubound = np.array([360.0, +r_blade * 0.2])
        #lbound = np.array([  0.0, -r_blade * 0.1])
        #ubound = np.array([360.0, +r_blade * 0.1])
        bounds = (lbound, ubound)
        res_lsq = least_squares(funBladeImgPointByThetaAndDeflection, x0, \
            bounds= bounds, 
            args=(r_blade, r44_blade, cmat, dvec, r44_cam, bladeImagePoint))
        if (res_lsq.cost < minCost):
            minCost = res_lsq.cost
            bestTheta = res_lsq.x[0]
            bestDeflection = res_lsq.x[1]
            bestRes = res_lsq
#        print(res_lsq)
#        print('-----------------------')
    eigs = np.linalg.eig(bestRes.jac)
    condition = max(abs(eigs[0])) / min(abs(eigs[0]))
    if condition > 10:
        print(bestTheta, bestDeflection, condition) 
    return bestTheta, bestDeflection, condition

def bladeThetaAndDeflectionByImgPoint2(bladeImagePoint, r_blade, r44_blade, cmat, dvec, r44_cam):
    minCost = 1e30
    bestTheta = -1;
    bestDeflection = -1;
    bestRes =[];
    rvec = cv.Rodrigues(r44_cam[0:3,0:3])[0]
    tvec = r44_cam[0:3,3].reshape(3,1)
    # find the angle (bestTheta) which project point is close to 
    # the given image point, assuming the preliminary bestDeflection 
    # is zero
    for theta_i in range(360):
        initTheta = theta_i * 1.0
        x_tip_turbCoord = np.array([
            r_blade * cos(initTheta * np.pi / 180.),
            r_blade * sin(initTheta * np.pi / 180.),
            0.0, 1.], dtype=float).reshape((4,1))
        x_tip_world = np.linalg.inv(r44_blade) @ x_tip_turbCoord
        x_tip_img = cv.projectPoints(x_tip_world[0:3], 
                                     rvec, tvec, cmat, dvec)[0].reshape(-1)
        cost = np.linalg.norm(bladeImagePoint.flatten() - x_tip_img.flatten())
        if cost < minCost:
            minCost = cost
            bestTheta = initTheta
            bestDeflection = 0.0
    # run least square method to find the best theta and deflection, 
    # given the initial guess (bestTheta, 0.0)
    # and the upper/lower bounds:
    #    bestTheta +/- 30 degrees
    #    bestDeflection +/- r_blade * 0.2
    x0 = np.array((bestTheta, bestDeflection),dtype=float)
    lbound = np.array([bestTheta - 20., -r_blade * 0.2])
    ubound = np.array([bestTheta + 20., +r_blade * 0.2])
    bounds = (lbound, ubound)
    res_lsq = least_squares(funBladeImgPointByThetaAndDeflection, x0,
        bounds= bounds, 
        args=(r_blade, r44_blade, cmat, dvec, r44_cam, bladeImagePoint))
    minCost = res_lsq.cost
    bestTheta = res_lsq.x[0]
    bestDeflection = res_lsq.x[1]
    bestRes = res_lsq
    # find the condition number of the system (jacobian matrix)
    eigs = np.linalg.eig(bestRes.jac)
    condition = max(abs(eigs[0])) / min(abs(eigs[0]))
    # return best theta and deflection, and the system condition number 
    return bestTheta, bestDeflection, condition

if __name__ == '__main__':
    work_dir_path = r"H:\CWH_thesis_experimental\PD_V_NF_SCBT\intersection_quadratic_subpixel_level"
    point_position_filepath = r"point_position_quadratic_intersectionsubpixel"
    calib_result_filepath = r"H:\CWH_thesis_experimental\PD_NV_NF_SCBT\intersection_quadratic_subpixel_level\point_position_quadratic_intersectionsubpixel_extrinsic_parameter.npz"

    point_position_dict = np.load(os.path.join(work_dir_path, point_position_filepath + ".npz"))
    cam_cal_extrinsic_parameter = np.load(calib_result_filepath)

    data_item = ['p3']
    for ele in data_item:
        data_len = point_position_dict[ele].shape[0]

        calcTheta = np.zeros(data_len, dtype=float)
        calcDeflection = np.zeros(data_len, dtype=float)
        calcCondition = np.zeros(data_len, dtype=float)
        worldcoord = np.zeros((data_len,4), dtype=float)

        bladeImagePoint = point_position_dict[ele]
        r_blade = 55.0

        '''
        800x600-cmat
        <Matrix 3x3 (1111.1111,    0.0000, 400.0000)
                    (   0.0000, 1111.1111, 300.0000)
                    (   0.0000,    0.0000,   1.0000)>   
        
        4K-cmat
        <Matrix 3x3 (5333.3335,    0.0000, 1920.0000)
                    (   0.0000, 5333.3335, 1080.0000)
                    (   0.0000,    0.0000,    1.0000)>
        '''
        
        cmat = np.zeros((3, 3), dtype=float)
        cmat[0, 0] = 1111.1111
        cmat[1, 1] = 1111.1111
        cmat[0, 2] = 400.
        cmat[1, 2] = 300.
        cmat[2, 2] = 1
        dvec = np.zeros((5,1), dtype=float)

        r44 = cam_cal_extrinsic_parameter['r44']
        
        bladeCenter = np.array([0, 0, 0], dtype=np.float64)
        bladeAim = bladeCenter + np.array([0, -100, 0], dtype=np.float64)
        r44_blade = extrinsicR44ByCamposAndAim(bladeCenter, bladeAim)

        for i in range(data_len):
            calcTheta[i], calcDeflection[i], calcCondition[i] = bladeThetaAndDeflectionByImgPoint2(bladeImagePoint[i], r_blade, r44_blade, cmat, dvec, r44)
            worldcoord[i] = bladePointByThetaAndDeflection(r_blade, r44_blade, calcTheta[i], calcDeflection[i])
            
        print()
        print(worldcoord)
        step = np.arange(0,data_len,1, dtype=float)
        timestep = np.arange(0,data_len/20,0.05, dtype=float)
        y_displacement = np.zeros((data_len, 1), dtype=float)
        for i in range(data_len):
            y_displacement[i][0] = worldcoord[i][1]

        # plot
        plt.plot(timestep,y_displacement,c="b")
        plt.xlabel("Time (sec) ", fontweight = "bold")  
        plt.ylabel("Displacement (meter)", fontweight = "bold")
        plt.title("Y-axis Displacement", fontsize = 12, fontweight = "bold")
        plt.ylim(np.min(y_displacement), np.max(y_displacement))
        plt.savefig(os.path.join(work_dir_path, 'y_axis_Displacement.png'))
        plt.show()

        # plot
        plt.plot(timestep,calcCondition,c="b")
        plt.xlabel("Time (sec) ", fontweight = "bold")  
        plt.ylabel("Condition Number ", fontweight = "bold")
        plt.title("Calculate Condition Number ", fontsize = 12, fontweight = "bold")
        plt.ylim(np.min(calcCondition), np.max(calcCondition))
        plt.savefig(os.path.join(work_dir_path, 'Calculate_Condition_Number_x_time.png'))
        plt.show()
        
        # plot
        plt.plot(step,calcCondition,c="b")
        plt.xlabel("Step ", fontweight = "bold")  
        plt.ylabel("Condition Number ", fontweight = "bold")
        plt.title("Calculate Condition Number ", fontsize = 12, fontweight = "bold")
        plt.ylim(np.min(calcCondition), np.max(calcCondition))
        plt.savefig(os.path.join(work_dir_path, 'Calculate_Condition_Number_x_step.png')) 
        plt.show()
        
        # 存成 csv
        #save_y = y_displacement.reshape(data_len, 1)
        np.savetxt(os.path.join(work_dir_path, point_position_filepath + "_" + ele + '.csv'), y_displacement, delimiter=',')
        