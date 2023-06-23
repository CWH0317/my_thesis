import numpy as np 
import io, re, sys, os
import cv2 as cv
import matplotlib.pyplot as plt
from math import cos, sin, tan, pi
from scipy.optimize import least_squares

from rvecTvecFromPosAim import rvecTvecFromPosAim
from rvecTvecFromR44 import rvecTvecFromR44
from r44FromRvecTvec import r44FromRvecTvec
from r44FromCamposAndAim import r44FromCamposAndAim
from dist_point_to_polyline_2D import minDistPointToPolygonalchain2d, dist_point_to_polyline_2D

from pandas import read_csv

def vertex_csv_dataloader(filename):
    # reading CSV file
    data = read_csv(filename)
    
    # converting column data to list
    input_p1 = data['p1'].tolist()
    input_p2 = data['p2'].tolist()
    input_p3 = data['p3'].tolist()

    p1 = np.zeros((len(input_p1), 2), dtype=float)
    p2 = np.zeros((len(input_p2), 2), dtype=float)
    p3 = np.zeros((len(input_p3), 2), dtype=float)

    for i in range(len(input_p1)):
        p1_a = input_p1[i].strip("()")
        p1_a = p1_a.split(",")
        p1_x = float(p1_a[0])
        p1_y = float(p1_a[1])
        p1[i][0] = p1_x
        p1[i][1] = p1_y

        p2_a = input_p2[i].strip("()")
        p2_a = p2_a.split(",")
        p2_x = float(p2_a[0])
        p2_y = float(p2_a[1])
        p2[i][0] = p2_x
        p2[i][1] = p2_y
        
        p3_a = input_p3[i].strip("()")
        p3_a = p3_a.split(",")
        p3_x = float(p3_a[0])
        p3_y = float(p3_a[1])
        p3[i][0] = p3_x
        p3[i][1] = p3_y
        
    return p1, p2, p3

#
# assuming the ring is on the x-z plane. The ring center is at origin. The 
# camera is close to the -Y axis (negative y axis). The camera x-axis is 
# horizontal, that is, on the x-y plane.  
#
def solvePnP_OnRing(radius, # radius of the ring
                    imagePoints,  # 
                    cameraMatrix, # camera matrix
                    distCoeffs,   # distortion coefficients
                    debug=False,  # print debug information
                    ):
    """
    Brief description
    -----------------
    This function estimates the camera extrinsic parameters (i.e., rvec and 
    tvec) given a set of image points which are on a ring and the radius of 
    the ring. The ring must be on the x-z plane which ring center is at the 
    origin. The camera is close to the -Y axis (netative y axis) and the 
    camera x-axis on the x-y plane.  
    Parameters
    ----------
    radius : TYPE float
        radius of the ring
    imagePoints : TYPE np.ndarray, shape=(N,2), dtype=float, where N is number
                  of image points. It is suggested that N > 10 or much more. 
        image coordinates of the points which are on the ring
    cameraMatrix : TYPE np.ndarray, shape(3,3), dtype=float
        the 3-by-3 camera matrix
    distCoeffs : TYPE np.ndarray, shape(n), dtype=float, where n is number of
                 coefficients
        the distortion coefficients
    Returns
    -------
    TYPE: tuple
        [0]: success (true) or not (false), a value returned from solvePnP
        [1]: rvec of the camera (np.ndarray, shape=(3,1), dtype=float)
        [2]: tvec of the camera (np.ndarray, shape=(3,1), dtype=float)
        [3]: estimated 3D points of the image points
        [4]: prjected image points of the aforementioned 3D points
    """
    #
    nPoints = imagePoints.shape[0]; 
    #
    # plot
    #
    if debug is True:
        plt.figure();
        plotStart = 0; 
        plotEnd = 20;  
        thePlt = plt.scatter(imagePoints[plotStart:plotEnd,0],
                             imagePoints[plotStart:plotEnd,1]); 
        thePlt.axes.invert_yaxis()
        thePlt.axes.axis('equal')
        thePlt.axes.grid('on', linestyle='dashed')
        thePlt.axes.set_title('Blade Image Points')
        thePlt.axes.set_xlabel('Image x (pixel)')
        thePlt.axes.set_ylabel('Image y (pixel)')
        plt.show()       
    #    
    # estimate initial guess of extrinsic parameters
    #   origin is at the center of the blade. z-axis: upward, y-axis: close to 
    #     the z-axis of the camera if camera is right in front of the turbine
    #   estimate image image center and radius of trajectory
    imgCx = sum(imagePoints)[0] / nPoints
    imgCy = sum(imagePoints)[1] / nPoints
    minImgX = np.min(imagePoints[:,0])
    maxImgX = np.max(imagePoints[:,0])
    minImgY = np.min(imagePoints[:,1])
    maxImgY = np.max(imagePoints[:,1])
    avgImgRadiusX = (maxImgX - minImgX) / 2
    avgImgRadiusY = (maxImgY - minImgY) / 2
    #
    #   estimate 5 parameters that can determine the rvec and tvec of the camera
    #   These paramters are cam_x, cam_y, cam_z, aim_x, aim_z
    #   where camera position (cameraPosition) is np.array([cam_x, cam_y, cam_z])
    #   and aiming point (aimPoint) is np.array([aim_x, 0, aim_z])
    #   location of the camera  
    optx = np.zeros((5,), dtype=float)
    optx[0] = 0  # camera position x
    optx[1] = (-1) * radius * cameraMatrix[0,0] / avgImgRadiusX # camera position y
    optx[2] = 0  # camera position z
    optx[3] = 0  # camera aiming point x
    optx[4] = 0  # camera aiming point z
    cameraPosition = np.zeros((3,), dtype=float)
    cameraPosition[0] = optx[0]
    cameraPosition[1] = optx[1]
    cameraPosition[2] = optx[2]
    aimPoint = np.zeros((3,), dtype=float)                                   
    aimPoint[0] = optx[3]
    aimPoint[1] = 0.0
    aimPoint[2] = optx[4] 
    #
    #   convert initial guess of rvec (rvecInit) and tvec (tvecInit)
    #   to 4x4 matrix
    r44Init = r44FromCamposAndAim(cameraPosition, aimPoint)
    rvecInit, tvecInit = rvecTvecFromR44(r44Init)
    #
    # estimate object points 
    # 
    objPoints = np.zeros((nPoints, 3), dtype=float)
    for i in range(nPoints):
        [dx, dy] = imagePoints[i,:] - [imgCx, imgCy]
        dy = -dy
        objPoints[i,0] = radius * dx / avgImgRadiusX 
        objPoints[i,1] = 0
        objPoints[i,2] = radius * dy / avgImgRadiusY 
    # 
    # estimate a set of extrinsic parameters by using solvePnP
    # based on the initial guess
    #
    rvecInit_ = np.copy(rvecInit).reshape(3,1)
    tvecInit_ = np.copy(tvecInit).reshape(3,1)
    retSolvePnP = cv.solvePnP(objPoints, imagePoints, 
                              cameraMatrix, distCoeffs, 
                              rvecInit_, tvecInit_, True)
#    retSolvePnP = cv.solvePnP(objPoints, imagePoints, 
#                              cameraMatrix, distCoeffs)
    solvePnPOkay = retSolvePnP[0]
    rvec = retSolvePnP[1]
    tvec = retSolvePnP[2]
    #
    # calculate project points
    #
    retPrjPoints = cv.projectPoints(objPoints, rvec, tvec, 
                                    cameraMatrix, distCoeffs)
    prjPoints = retPrjPoints[0].reshape((nPoints,2))
    # 
    # generate return data
    #
    listRetSolvePnP = list(retSolvePnP)
    listRetSolvePnP.append(objPoints)
    listRetSolvePnP.append(prjPoints)
    retSolvePnP = tuple(listRetSolvePnP)
    return retSolvePnP


def solvePnP_OnRing2(radius, # radius of the ring
                     imagePoints,  # 
                     cameraMatrix, # camera matrix
                     distCoeffs,   # distortion coefficient 
                     initRvec,     # initial guess of rvec
                     initTvec,     # initial guess of tvec
                     nRingPoints=720, # number of points for the calculated trajectory
                     debug=False,  # print debug information
                     ):
    """
    Brief description
    -----------------
    This function is designed to be called after calling solvePnP_OnRing. 
    This function further optimizes the extrinsic parameters by minimizing
    the distances between image points and the calculated trajectory 
    Parameters
    ----------
    radius : TYPE float
        radius of the ring
    imagePoints : TYPE np.ndarray, shape=(N,2), dtype=float, where N is number
                  of image points. It is suggested that N > 10 or much more. 
        image coordinates of the points which are on the ring
    cameraMatrix : TYPE np.ndarray, shape(3,3), dtype=float
        the 3-by-3 camera matrix
    distCoeffs : TYPE np.ndarray, shape(n), dtype=float, where n is number of
                 coefficients
        the distortion coefficients
    initRvec : TYPE np.ndarray, shape=(3), dtype=float
        initial guess of rvec, can be element [1] of returned value of 
        solvePnP_OnRing()
    initTvec : Type np.ndarray, shape=(3), dtype=float
        initial guess of tvec
    nRingPoints : TYPE int
        number of points that regularly distributed on the ring (default 720)
    Returns
    -------
    TYPE: tuple
        [0]: success (true) or not (false), a value returned from solvePnP
        [1]: rvec of the camera
        [2]: tvec of the camera
        [3]: estimated 3D points of the image points
        [4]: prjected image points of the aforementioned 3D points
    """
    # 
    # generate ringObjPoints
    # ringObjPoints are 3d object points that regularly distribute on the ring
    #
    ringObjPoints = np.zeros((nRingPoints,3),dtype=float)
    for i in range(nRingPoints):
        theta = (i * 2. * np.pi) / nRingPoints
        ringObjPoints[i,0] = radius * np.cos(theta)
        ringObjPoints[i,1] = 0.0
        ringObjPoints[i,2] = radius * np.sin(theta)
        
    #
    # get initial guess of 5-coefficient extrinsic parameters:
    # camposx, camposy, camposz, aimpx, aimpz
    #
    initR44 = r44FromRvecTvec(initRvec, initTvec)
    initR44inv = np.linalg.inv(initR44)
    camposx =  initR44inv[0,3]
    camposy =  initR44inv[1,3]
    camposz =  initR44inv[2,3]    
    fac = -camposy / initR44inv[1,2];
    aimx = camposx + fac * initR44inv[0,2]
    aimz = camposz + fac * initR44inv[2,2]
    initOptx = np.array([camposx, camposy, camposz, aimx, aimz])
    
    def projErrCostVec(optx, ringObjPoints, cameraMatrix, distCoeffs, 
                       imagePoints):
        #
        # calculate projected points of regular ring points given 5-coef
        # parameters
        #
        campos = optx[0:3]
        aimpnt = np.array([optx[3], 0., optx[4]])
        tryR44 = r44FromCamposAndAim(campos, aimpnt)
        tryRvec, tryTvec = rvecTvecFromR44(tryR44)
        retPrjPoints = cv.projectPoints(ringObjPoints, 
                                       tryRvec, tryTvec, 
                                       cameraMatrix, distCoeffs)
        prjPoints = retPrjPoints[0].reshape((nRingPoints,2))
        #
        # generate the vector of cost function (vecy)
        # vecy is composed of [(x0 - x0m) (y0 - y0m) (x1 - x1m) (y1 - y1m) ...] 
        #    where [xi,yi] is imagePoints[i,:], and [xim,yim] is the point on the 
        #    ring (projected in image) which has the minimum distance to [xi,yi]
        # We will then try to minimize vecy by least squares
        #
        nImgPoints = imagePoints.shape[0]
        vecy = np.zeros((2 * nImgPoints), dtype=float)
        for i in range(nImgPoints):
            thePoint = imagePoints[i]
            retMinDist = minDistPointToPolygonalchain2d(thePoint, prjPoints)
            # retMinDist = dist_point_to_polyline_2D(thePoint, prjPoints)
            minDist = retMinDist[0]
            minDistPoint = retMinDist[1]
#            minSegment = retMinDist[2]
#            minAlpha = retMinDist[3]
            vecy[0 + i * 2] = imagePoints[i, 0] - minDistPoint[0]
            vecy[1 + i * 2] = imagePoints[i, 1] - minDistPoint[1]
        return vecy
    #
    # start least squares
    #
    retLS = least_squares(projErrCostVec, initOptx, 
                          args=(ringObjPoints, cameraMatrix, 
                                distCoeffs, imagePoints))
    #
    # get rvec and tvec from the least squares result
    #
    r44LS = r44FromCamposAndAim(retLS.x[0:3], 
            np.array([retLS.x[3], 0., retLS.x[4]]))
    rvecLS, tvecLS = rvecTvecFromR44(r44LS)
    # 
    retPrjRingPoints = cv.projectPoints(ringObjPoints, 
                                   rvecLS, tvecLS, 
                                   cameraMatrix, distCoeffs)
    prjRingPoints = retPrjRingPoints[0].reshape((nRingPoints,2))
    # find the best object points on the ring (actually a multi-point polyline)
    # that can project to the imagePoints (given by the user)
    objPoints = np.zeros((imagePoints.shape[0], 3), dtype=float)
    for i in range(imagePoints.shape[0]):
        p = imagePoints[i]
        # dist, point, seg, alpha = dist_point_to_polyline_2D(p, prjRingPoints)
        dist, point, seg, alpha = minDistPointToPolygonalchain2d(p, prjRingPoints)
        objPoints[i] = (1. - alpha) * ringObjPoints[seg] + alpha * ringObjPoints[seg + 1] 
    # Find the project points of objPoints (most likely 3D points of given image points)
    retPrjPoints = cv.projectPoints(objPoints, 
                                    rvecLS, tvecLS, 
                                    cameraMatrix, distCoeffs)
    prjPoints = retPrjPoints[0].reshape((-1, 2))
    #
    #  plot
    #
    if debug is True:
        fig = plt.figure()
        thePlt = plt.scatter(imagePoints[:,0],
                             imagePoints[:,1], s=10,
                             label='Image points'); 
        thePlt = plt.scatter(prjPoints[:,0],
                             prjPoints[:,1], s=10,
                             label='Best projected points'); 
#        thePlt = plt.scatter(prjRingPoints[:,0],
#                             prjRingPoints[:,1], s=2, 
#                             label='Dense-ring points'); 
        plt.plot(prjRingPoints[:,0], prjRingPoints[:,1], 
                 color='green', linewidth=1,
                 markersize=12,
                 label='Ring (by dense points)'); 
        thePlt.axes.invert_yaxis()
        thePlt.axes.axis('equal')
        thePlt.axes.grid('on', linestyle='dashed')
        thePlt.axes.set_title('Projected Image Points')
        thePlt.axes.set_xlabel('Image x (pixel)')
        thePlt.axes.set_ylabel('Image y (pixel)')
        thePlt.axes.legend()
    # 
    # generate return data
    #   
    retSolvePnP = [True]
    retSolvePnP.append(np.array(rvecLS).reshape(3, 1))
    retSolvePnP.append(np.array(tvecLS).reshape(3, 1))
    listRetSolvePnP = list(retSolvePnP)
    listRetSolvePnP.append(np.array(objPoints).reshape(-1, 3))
    listRetSolvePnP.append(np.array(prjPoints).reshape(-1, 2))
    retSolvePnP = tuple(listRetSolvePnP)

    return retSolvePnP
         

#
# assuming the ring is on the x-z plane. The ring center is at origin. User needs 
# to guess a rough camera position. The camera x-axis is 
# horizontal, that is, on the x-y plane.  
#
def solvePnP_OnRing_CamPosGuess(radius, # radius of the ring
                    imagePoints,  # 
                    cameraMatrix, # camera matrix
                    distCoeffs,   # distortion coefficients
                    initCamPos,   # initial guess of the camera position
                    debug=False,  # print debug information
                    ):
    """
    Brief description
    -----------------
    This function estimates the camera extrinsic parameters (i.e., rvec and 
    tvec) given a set of image points which are on a ring and the radius of 
    the ring. The ring must be on the x-z plane which ring center is at the 
    origin. The camera x-axis on the x-y plane.  
    Parameters
    ----------
    radius : TYPE float
        radius of the ring
    imagePoints : TYPE np.ndarray, shape=(N,2), dtype=float, where N is number
                  of image points. It is suggested that N > 10 or much more. 
        image coordinates of the points which are on the ring
    cameraMatrix : TYPE np.ndarray, shape(3,3), dtype=float
        the 3-by-3 camera matrix
    distCoeffs : TYPE np.ndarray, shape(n), dtype=float, where n is number of
                 coefficients
        the distortion coefficients
    initCamPos : TYPE np.ndarray, shape(3), dtype=float
        initial guess of the camera position 
    Returns
    -------
    TYPE: tuple
        [0]: success (true) or not (false), a value returned from solvePnP
        [1]: rvec of the camera (np.ndarray, shape=(3,1), dtype=float)
        [2]: tvec of the camera (np.ndarray, shape=(3,1), dtype=float)
        [3]: estimated 3D points of the image points
        [4]: prjected image points of the aforementioned 3D points
    """
    #
    nPoints = imagePoints.shape[0]; 
    #
    # plot
    #
    if debug is True:
        plt.figure();
        plotStart = 0; 
        plotEnd = 20;  
        thePlt = plt.scatter(imagePoints[plotStart:plotEnd,0],
                             imagePoints[plotStart:plotEnd,1]); 
        thePlt.axes.invert_yaxis()
        thePlt.axes.axis('equal')
        thePlt.axes.grid('on', linestyle='dashed')
        thePlt.axes.set_title('Blade Image Points')
        thePlt.axes.set_xlabel('Image x (pixel)')
        thePlt.axes.set_ylabel('Image y (pixel)')
        plt.show()       
    #    
    # estimate initial guess of extrinsic parameters
    #   origin is at the center of the blade. z-axis: upward, y-axis: close to 
    #     the z-axis of the camera if camera is right in front of the turbine
    #   estimate image image center and radius of trajectory
    imgCx = sum(imagePoints)[0] / nPoints
    imgCy = sum(imagePoints)[1] / nPoints
    minImgX = np.min(imagePoints[:,0])
    maxImgX = np.max(imagePoints[:,0])
    minImgY = np.min(imagePoints[:,1])
    maxImgY = np.max(imagePoints[:,1])
    avgImgRadiusX = (maxImgX - minImgX) / 2
    avgImgRadiusY = (maxImgY - minImgY) / 2
    #
    #   estimate 5 parameters that can determine the rvec and tvec of the camera
    #   These paramters are cam_x, cam_y, cam_z, aim_x, aim_z
    #   where camera position (cameraPosition) is np.array([cam_x, cam_y, cam_z])
    #   and aiming point (aimPoint) is np.array([aim_x, 0, aim_z])
    #   location of the camera  
    optx = np.zeros((5,), dtype=float)
#    optx[0] = 0  # camera position x
#    optx[1] = (-1) * radius * cameraMatrix[0,0] / avgImgRadiusX # camera position y
#    optx[2] = 0  # camera position z
    optx[0] = initCamPos[0]
    optx[1] = initCamPos[1]
    optx[2] = initCamPos[2]
    optx[3] = 0  # camera aiming point x
    optx[4] = 0  # camera aiming point z
    cameraPosition = np.zeros((3,), dtype=float)
    cameraPosition[0] = optx[0]
    cameraPosition[1] = optx[1]
    cameraPosition[2] = optx[2]
    aimPoint = np.zeros((3,), dtype=float)                                   
    aimPoint[0] = optx[3]
    aimPoint[1] = 0.0
    aimPoint[2] = optx[4] 
    #
    #   convert initial guess of rvec (rvecInit) and tvec (tvecInit)
    #   to 4x4 matrix
    r44Init = r44FromCamposAndAim(cameraPosition, aimPoint)
    rvecInit, tvecInit = rvecTvecFromR44(r44Init)
    #
    # estimate object points 
    # 
    objPoints = np.zeros((nPoints, 3), dtype=float)
    for i in range(nPoints):
        [dx, dy] = imagePoints[i,:] - [imgCx, imgCy]
        dy = -dy
        objPoints[i,0] = radius * dx / avgImgRadiusX 
        objPoints[i,1] = 0
        objPoints[i,2] = radius * dy / avgImgRadiusY 
    # 
    # estimate a set of extrinsic parameters by using solvePnP
    # based on the initial guess
    #
    rvecInit_ = np.copy(rvecInit).reshape(3,1)
    tvecInit_ = np.copy(tvecInit).reshape(3,1)
    retSolvePnP = cv.solvePnP(objPoints, imagePoints, 
                              cameraMatrix, distCoeffs, 
                              rvecInit_, tvecInit_, True)
#    retSolvePnP = cv.solvePnP(objPoints, imagePoints, 
#                              cameraMatrix, distCoeffs)
    solvePnPOkay = retSolvePnP[0]
    rvec = retSolvePnP[1]
    tvec = retSolvePnP[2]
    #
    # calculate project points
    #
    retPrjPoints = cv.projectPoints(objPoints, rvec, tvec, 
                                    cameraMatrix, distCoeffs)
    prjPoints = retPrjPoints[0].reshape((nPoints,2))
    # 
    # generate return data
    #
    listRetSolvePnP = list(retSolvePnP)
    listRetSolvePnP.append(objPoints)
    listRetSolvePnP.append(prjPoints)
    retSolvePnP = tuple(listRetSolvePnP)
    return retSolvePnP
    

def calibBladeTip(radius,       # radius of the ring
                  imagePoints,  # image points
                  cameraMatrix, # camera matrix
                  distCoeffs,   # distortion coefficients
                  debug=False,  # print debug information
                  ):
    """
    Brief description
    -----------------
    This function estimates the camera extrinsic parameters (i.e., rvec and 
    tvec) given a set of image points which are on a ring and the radius of 
    the ring. The ring must be on the x-z plane which ring center is at the 
    origin. The camera is close to the -Y axis (netative y axis) and the 
    camera x-axis on the x-y plane.
    This function calls solvePnP_OnRing() and solvePnP_OnRing2().
    Parameters
    ----------
    radius : TYPE float
        radius of the ring
    imagePoints : TYPE np.ndarray, shape=(N,2), dtype=float, where N is number
                  of image points. It is suggested that N > 10 or much more. 
        image coordinates of the points which are on the ring
    cameraMatrix : TYPE np.ndarray, shape(3,3), dtype=float
        the 3-by-3 camera matrix
    distCoeffs : TYPE np.ndarray, shape(n), dtype=float, where n is number of
                 coefficients
        the distortion coefficients
    Returns
    -------
    TYPE: tuple
        [0]: success (true) or not (false), a value returned from solvePnP
        [1]: rvec of the camera (np.ndarray, shape=(3,1), dtype=float)
        [2]: tvec of the camera (np.ndarray, shape=(3,1), dtype=float)
        [3]: estimated 3D points of the image points
        [4]: prjected image points of the aforementioned 3D points
        [5]: projection errors, i.e., image points - projected points
    """
    # phase 1 calibration: estimates object points roughly and carries out 
    #                      solvePnP()
    cmat = cameraMatrix.reshape(3, 3)
    dvec = distCoeffs.reshape(-1, 1)
    retSolvePnP_OnRing = solvePnP_OnRing(
            radius, 
            imagePoints, 
            cmat, 
            dvec)
    retSolvePnP_OnRing_Okay = retSolvePnP_OnRing[0]
    initRvec = retSolvePnP_OnRing[1]
    initTvec = retSolvePnP_OnRing[2]
#    objPoints = retSolvePnP_OnRing[3]
    prjPoints = retSolvePnP_OnRing[4]
    # check
    if debug is True:
        initR44 = np.eye(4, dtype=float)
        initR44[0:3, 0:3] = cv.Rodrigues(initRvec)[0]
        initR44[0:3, 3] = initTvec[0:3,0]
        initR44inv = np.linalg.inv(initR44)
        print("# calibBladeTip():")
        print("#   The first-phase solvePnP result:\n")
        print("#   Rvec: \n", initRvec)
        print("#   Tvec: \n", initRvec)
        print("#   R44: \n", initR44)
        print("#   R44 inverse:\n", initR44inv)
        print("#   The positive maximum projection error: ", 
              np.max(prjPoints.flatten() - imagePoints.flatten()))
    # phase 2: given a good initial guess of rvec and tvec, employes 
    #          least-square process to find the optimized rvec and tvec.
    #          assuming the camera x-axis has no z component in the 
    #          world coordinate
    retSolvePnP_OnRing2 = solvePnP_OnRing2(
        radius, 
        imagePoints, 
        cmat, 
        dvec, 
        initRvec, 
        initTvec,
        nRingPoints=720,
        debug=debug
        )
    retSolvePnP_OnRing2_Okay = retSolvePnP_OnRing2[0]
    rvec = retSolvePnP_OnRing2[1].reshape(3, 1)
    tvec = retSolvePnP_OnRing2[2].reshape(3, 1)
    objPoints = retSolvePnP_OnRing2[3]
    prjPoints = retSolvePnP_OnRing2[4]
    prjErrors = prjPoints - imagePoints
    # check
    if debug is True:
        r44 = np.eye(4, dtype=float)
        r44[0:3, 0:3] = cv.Rodrigues(rvec)[0]
        r44[0:3, 3] = tvec[0:3,0]
        r44inv = np.linalg.inv(r44)
        # check
        print("# The second-phase solvePnP (retSolvePnP_OnRing2) result:\n")
        print("Rvec: \n", rvec)
        print("Tvec: \n", tvec)
        print("R44: \n", r44)
        print("R44 inverse:\n", r44inv)
        print("The positive maximum projection error: ", 
              np.max(prjPoints.flatten() - imagePoints.flatten()))
    # return
    return retSolvePnP_OnRing2_Okay, rvec, tvec, objPoints, prjPoints, prjErrors


def calibBladeTipCamPosGuess(
                   radius,       # radius of the ring
                   imagePoints,  # image points
                   cameraMatrix, # camera matrix
                   distCoeffs,   # distortion coefficients
                   initCamPos,   # initial guess of the camera position. 
                   debug=False,  # print debug information
                  ):
    """
    Brief description
    -----------------
    This function estimates the camera extrinsic parameters (i.e., rvec and 
    tvec) given a set of image points which are on a ring and the radius of 
    the ring. The ring must be on the x-z plane which ring center is at the 
    origin. The camera x-axis on the x-y plane. The camera does not need to be
    close to -Y axis. It can be around +Y. 
    This function calls solvePnP_OnRing() and solvePnP_OnRing2().
    Parameters
    ----------
    radius : TYPE float
        radius of the ring
    imagePoints : TYPE np.ndarray, shape=(N,2), dtype=float, where N is number
                  of image points. It is suggested that N > 10 or much more. 
        image coordinates of the points which are on the ring
    cameraMatrix : TYPE np.ndarray, shape(3,3), dtype=float
        the 3-by-3 camera matrix
    distCoeffs : TYPE np.ndarray, shape(n), dtype=float, where n is number of
                 coefficients
        the distortion coefficients
    initCamPos : TYPE np.ndarray, shape(3), dtype=float
        the initial guess of the camera position.\
    Returns
    -------
    TYPE: tuple
        [0]: success (true) or not (false), a value returned from solvePnP
        [1]: rvec of the camera (np.ndarray, shape=(3,1), dtype=float)
        [2]: tvec of the camera (np.ndarray, shape=(3,1), dtype=float)
        [3]: estimated 3D points of the image points
        [4]: prjected image points of the aforementioned 3D points
        [5]: projection errors, i.e., image points - projected points
    """
    # phase 1 calibration: estimates object points roughly and carries out 
    #                      solvePnP()
    cmat = cameraMatrix.reshape(3, 3)
    dvec = distCoeffs.reshape(-1, 1)
    retSolvePnP_OnRing = solvePnP_OnRing_CamPosGuess(
            radius, 
            imagePoints, 
            cmat, 
            dvec, 
            initCamPos)
    retSolvePnP_OnRing_Okay = retSolvePnP_OnRing[0]
    initRvec = retSolvePnP_OnRing[1]
    initTvec = retSolvePnP_OnRing[2]
#    objPoints = retSolvePnP_OnRing[3]
    prjPoints = retSolvePnP_OnRing[4]
    # check
    if debug is True:
        initR44 = np.eye(4, dtype=float)
        initR44[0:3, 0:3] = cv.Rodrigues(initRvec)[0]
        initR44[0:3, 3] = initTvec[0:3,0]
        initR44inv = np.linalg.inv(initR44)
        print("# calibBladeTip():")
        print("#   The first-phase solvePnP result:\n")
        print("#   Rvec: \n", initRvec)
        print("#   Tvec: \n", initRvec)
        print("#   R44: \n", initR44)
        print("#   R44 inverse:\n", initR44inv)
        print("#   The positive maximum projection error: ", 
              np.max(prjPoints.flatten() - imagePoints.flatten()))
    # phase 2: given a good initial guess of rvec and tvec, employes 
    #          least-square process to find the optimized rvec and tvec.
    #          assuming the camera x-axis has no z component in the 
    #          world coordinate
    retSolvePnP_OnRing2 = solvePnP_OnRing2(
        radius, 
        imagePoints, 
        cmat, 
        dvec, 
        initRvec, 
        initTvec,
        nRingPoints=720,
        debug=debug
        )
    retSolvePnP_OnRing2_Okay = retSolvePnP_OnRing2[0]
    rvec = retSolvePnP_OnRing2[1].reshape(3, 1)
    tvec = retSolvePnP_OnRing2[2].reshape(3, 1)
    objPoints = retSolvePnP_OnRing2[3]
    prjPoints = retSolvePnP_OnRing2[4]
    prjErrors = prjPoints - imagePoints
    # check
    if debug is True:
        r44 = np.eye(4, dtype=float)
        r44[0:3, 0:3] = cv.Rodrigues(rvec)[0]
        r44[0:3, 3] = tvec[0:3,0]
        r44inv = np.linalg.inv(r44)
        # check
        print("# The second-phase solvePnP (retSolvePnP_OnRing2) result:\n")
        print("Rvec: \n", rvec)
        print("Tvec: \n", tvec)
        print("R44: \n", r44)
        print("R44 inverse:\n", r44inv)
        print("The positive maximum projection error: ", 
              np.max(prjPoints.flatten() - imagePoints.flatten()))
        cam_cal_extrinsic_parameter = {'rvec' : rvec, 'tvec': tvec, 'r44': r44, 'r44_inv': r44inv }
        max_proj_error = np.max(prjPoints.flatten() - imagePoints.flatten())
    # return
    return retSolvePnP_OnRing2_Okay, rvec, tvec, objPoints, prjPoints, prjErrors, cam_cal_extrinsic_parameter, max_proj_error


def test_calibBladeTip():
    radius = 55; # known radius of the blade (precisely speaking, 
                 # the distance between the marker on the blade and the 
                 # center of rotation)
    #    directory = "D:\\yuansen\\ImPro\\impropy\\tests\\bladeRotatingForCalib3\\";
    directory = 'examples/2023windturbine/'
    fileImgPoints = "point_position.txt"
    fullfile = os.path.join(directory, fileImgPoints)
    print('# Image points file: %s\n' % (fullfile))
    # generate numpy array from file (delimiters: spaces, tabs, comma)
    imagePoints = np.loadtxt(fullfile)
#    imagePoints = imagePoints[0:200]
    nPoints = imagePoints.shape[0]; 
    #
    if (imagePoints.shape[0] > 1 and 
        imagePoints.shape[1] == 2): 
        print("# I got %d image points\n" % (nPoints))
    else:
        print("# Error. Cannot recognize data from file\n")
        print("# Shape of data: (%d,%d)\n" % 
              (imagePoints.shape[0], imagePoints.shape[1]))
    # get intrinsic (cmat, dvec)
    #directory = "D:\\yuansen\\ImPro\\impropy\\misc\\tests\\bladeRotatingForCalib3\\";
    fileCmat = "cmat.txt"
    fullfile = os.path.join(directory, fileCmat)
    print('# Intrinsic data file: %s\n' % (fullfile))
    cmat = np.loadtxt(fullfile).reshape(3,3)
    dvec = np.zeros((5,1), dtype=float)
    print("# Camera matrix: \n", cmat)
    print("# Distortion coefficients: \n", dvec)
    ret_calibBladeTip = calibBladeTip(
            radius, 
            imagePoints, 
            cmat, 
            dvec, 
            True)
    ret_calibBladeTip_OK = ret_calibBladeTip[0]
    rvec = ret_calibBladeTip[1].reshape(3, 1)
    tvec = ret_calibBladeTip[2].reshape(3, 1)
    objPoints = ret_calibBladeTip[3]
    prjPoints = ret_calibBladeTip[4]
    prjErrors = ret_calibBladeTip[5]
    #
    # plot
    #       
    if (True):
        fig = plt.figure();
        plotStart = 0; 
        plotEnd = nPoints;  
        thePlt = plt.scatter(imagePoints[plotStart:plotEnd,0],
                             imagePoints[plotStart:plotEnd,1], s=1); 
        thePlt = plt.scatter(prjPoints[plotStart:plotEnd,0],
                             prjPoints[plotStart:plotEnd,1], s=1); 
        thePlt.axes.invert_yaxis()
        thePlt.axes.axis('equal')
        thePlt.axes.grid('on', linestyle='dashed')
        thePlt.axes.set_title('Projected Image Points')
        thePlt.axes.set_xlabel('Image x (pixel)')
        thePlt.axes.set_ylabel('Image y (pixel)')
        plt.show()
    if (True):
        fig = plt.figure();
        ax = fig.add_subplot(projection='3d')
        plotStart = 0; 
        plotEnd = nPoints;  
        thePlt = ax.plot3D(objPoints[plotStart:plotEnd,0], 
                           objPoints[plotStart:plotEnd,1],
                           objPoints[plotStart:plotEnd,2], 
                           'gray', 
                           lw=0, marker='.', ms = 2)
        ax.set_box_aspect([1,1,1])
        thePlt[0].axes.grid('on', linestyle='dashed')
        thePlt[0].axes.set_title('Projected Image Points')
        retSetXLabel = ax.set_xlabel('World coord. X (mm)')
        retSetYLabel = ax.set_ylabel('World coord. Y (mm)')
        retSetZLabel = ax.set_zlabel('World coord. Z (mm)')
        plt.show()


def test_calibBladeTip_camPosGuess():
    radius = 55; # known radius of the blade (precisely speaking, 
                 # the distance between the marker on the blade and the 
                 # center of rotation)
    camPosGuess = np.array([-234., -89., 0.])
    #    directory = "D:\\yuansen\\ImPro\\impropy\\tests\\bladeRotatingForCalib3\\";
    directory = 'examples/2023windturbine/'
    fileImgPoints = "point_position.txt"
    fullfile = os.path.join(directory, fileImgPoints)
    print('# Image points file: %s\n' % (fullfile))
    # generate numpy array from file (delimiters: spaces, tabs, comma)
    imagePoints = np.loadtxt(fullfile)
#    imagePoints = imagePoints[0:200]
    nPoints = imagePoints.shape[0]; 
    #
    if (imagePoints.shape[0] > 1 and 
        imagePoints.shape[1] == 2): 
        print("# I got %d image points\n" % (nPoints))
    else:
        print("# Error. Cannot recognize data from file\n")
        print("# Shape of data: (%d,%d)\n" % 
              (imagePoints.shape[0], imagePoints.shape[1]))
    # get intrinsic (cmat, dvec)
    #directory = "D:\\yuansen\\ImPro\\impropy\\misc\\tests\\bladeRotatingForCalib3\\";
    fileCmat = "cmat.txt"
    fullfile = os.path.join(directory, fileCmat)
    print('# Intrinsic data file: %s\n' % (fullfile))
    cmat = np.loadtxt(fullfile).reshape(3,3)
    dvec = np.zeros((5,1), dtype=float)
    print("# Camera matrix: \n", cmat)
    print("# Distortion coefficients: \n", dvec)
    ret_calibBladeTip = calibBladeTipCamPosGuess(
            radius, 
            imagePoints, 
            cmat, 
            dvec, 
            camPosGuess, 
            True)
    ret_calibBladeTip_OK = ret_calibBladeTip[0]
    rvec = ret_calibBladeTip[1].reshape(3, 1)
    tvec = ret_calibBladeTip[2].reshape(3, 1)
    objPoints = ret_calibBladeTip[3]
    prjPoints = ret_calibBladeTip[4]
    prjErrors = ret_calibBladeTip[5]
    #
    # plot
    #       
    if (True):
        fig = plt.figure();
        plotStart = 0; 
        plotEnd = nPoints;  
        thePlt = plt.scatter(imagePoints[plotStart:plotEnd,0],
                             imagePoints[plotStart:plotEnd,1], s=1); 
        thePlt = plt.scatter(prjPoints[plotStart:plotEnd,0],
                             prjPoints[plotStart:plotEnd,1], s=1); 
        thePlt.axes.invert_yaxis()
        thePlt.axes.axis('equal')
        thePlt.axes.grid('on', linestyle='dashed')
        thePlt.axes.set_title('Projected Image Points')
        thePlt.axes.set_xlabel('Image x (pixel)')
        thePlt.axes.set_ylabel('Image y (pixel)')
        plt.show()
    if (True):
        fig = plt.figure();
        ax = fig.add_subplot(projection='3d')
        plotStart = 0; 
        plotEnd = nPoints;  
        thePlt = ax.plot3D(objPoints[plotStart:plotEnd,0], 
                           objPoints[plotStart:plotEnd,1],
                           objPoints[plotStart:plotEnd,2], 
                           'gray', 
                           lw=0, marker='.', ms = 2)
        ax.set_box_aspect([1,1,1])
        thePlt[0].axes.grid('on', linestyle='dashed')
        thePlt[0].axes.set_title('Projected Image Points')
        retSetXLabel = ax.set_xlabel('World coord. X (mm)')
        retSetYLabel = ax.set_ylabel('World coord. Y (mm)')
        retSetZLabel = ax.set_zlabel('World coord. Z (mm)')
        plt.show()



#test_calibBladeTip()
#test_calibBladeTip_camPosGuess()

def cwh_test_solvePnP_OnRing():
    radius = 55; # known radius of the blade (precisely speaking, 
                 # the distance between the marker on the blade and the 
                 # center of rotation)
    camPosGuess = np.array([-50., -280., 0.])
    #filename = "ex1_300_10_analysis/800x600_GT/point_position_intersectionsubpixel_quadratic.csv"
    #p1, p2, p3 = vertex_csv_dataloader(filename)
    #imagePoints = np.concatenate((p1, p2, p3))
    point_position_dict = np.load("D:/Lab/CWH_thesis/code/ex1_300_10_Z500mm_analysis_20230427/novibration/ex1_BW_300_10_Z500mm_novibration_point_position_pixel.npz")
    #point_position_dict = np.load("D:/Lab/CWH_thesis/code/ex1_300_10_analysis/800x600_GT/point_position_intersectionsubpixel_quadratic.npz")
    p1 = point_position_dict['p1']
    #p2 = point_position_dict['p2']
    #p3 = point_position_dict['p3']
    '''
    imagePoints = np.array(p3[0],p3[50],p3[100],p3[150],
                           p3[200],p3[250],p3[300],p3[350],p3[400],p3[450],p3[500],
                           p3[550],p3[600],p3[650],p3[700],p3[750],p3[800],p3[850],
                           p3[900],p3[950],p3[1000],p3[1050],p3[1100],p3[1150])
    '''
    imagePoints = p1
#    imagePoints = imagePoints[0:200]
    nPoints = imagePoints.shape[0]; 
    #
    if (imagePoints.shape[0] > 1 and 
        imagePoints.shape[1] == 2): 
        print("# I got %d image points\n" % (nPoints))
    else:
        print("# Error. Cannot recognize data from file\n")
        print("# Shape of data: (%d,%d)\n" % 
              (imagePoints.shape[0], imagePoints.shape[1]))
    # get intrinsic (cmat, dvec)
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
    print("# Camera matrix: \n", cmat)
    print("# Distortion coefficients: \n", dvec)
    ret_calibBladeTip = calibBladeTipCamPosGuess(
            radius, 
            imagePoints, 
            cmat, 
            dvec, 
            camPosGuess, 
            True)
    ret_calibBladeTip_OK = ret_calibBladeTip[0]
    rvec = ret_calibBladeTip[1].reshape(3, 1)
    tvec = ret_calibBladeTip[2].reshape(3, 1)
    objPoints = ret_calibBladeTip[3]
    prjPoints = ret_calibBladeTip[4]
    prjErrors = ret_calibBladeTip[5]
    cam_cal_extrinsic_parameter = ret_calibBladeTip[6]
    np.savez('D:/Lab/CWH_thesis/code/ex1_300_10_Z500mm_analysis_20230427/novibration/ex1_BW_300_10_Z500mm_novibration_point_position_pixel_cam_cal_extrinsic_parameter.npz', **cam_cal_extrinsic_parameter)
    #
    # plot
    #       
    if (True):
        fig = plt.figure();
        plotStart = 0; 
        plotEnd = nPoints;  
        thePlt = plt.scatter(imagePoints[plotStart:plotEnd,0],
                             imagePoints[plotStart:plotEnd,1], s=1); 
        thePlt = plt.scatter(prjPoints[plotStart:plotEnd,0],
                             prjPoints[plotStart:plotEnd,1], s=1); 
        thePlt.axes.invert_yaxis()
        thePlt.axes.axis('equal')
        thePlt.axes.grid('on', linestyle='dashed')
        thePlt.axes.set_title('Projected Image Points')
        thePlt.axes.set_xlabel('Image x (pixel)')
        thePlt.axes.set_ylabel('Image y (pixel)')
        plt.show()
    if (True):
        fig = plt.figure();
        ax = fig.add_subplot(projection='3d')
        plotStart = 0; 
        plotEnd = nPoints;  
        thePlt = ax.plot3D(objPoints[plotStart:plotEnd,0], 
                           objPoints[plotStart:plotEnd,1],
                           objPoints[plotStart:plotEnd,2], 
                           'gray', 
                           lw=0, marker='.', ms = 2)
        ax.set_box_aspect([1,1,1])
        thePlt[0].axes.grid('on', linestyle='dashed')
        thePlt[0].axes.set_title('Projected Image Points')
        retSetXLabel = ax.set_xlabel('World coord. X (mm)')
        retSetYLabel = ax.set_ylabel('World coord. Y (mm)')
        retSetZLabel = ax.set_zlabel('World coord. Z (mm)')
        plt.show()
    
#cwh_test_solvePnP_OnRing()

def test2_calibBladeTip_camPosGuess(work_dir_path, point_position_filepath, save_calib_result_filepath):
    radius = 55; # known radius of the blade (precisely speaking, 
                 # the distance between the marker on the blade and the 
                 # center of rotation)
    camPosGuess = np.array([-250., -250., 0.])
    point_position_dict = np.load(point_position_filepath)
    #point_position_dict = np.load("ex1_300_10_800x600_novibration_test_proj_point_position_pixel.npz")
    #p1 = point_position_dict['p1']
    #p2 = point_position_dict['p2']
    p3 = point_position_dict['p3']
    imagePoints = p3
#    imagePoints = imagePoints[0:200]
    nPoints = imagePoints.shape[0]; 
    #
    if (imagePoints.shape[0] > 1 and 
        imagePoints.shape[1] == 2): 
        print("# I got %d image points\n" % (nPoints))
    else:
        print("# Error. Cannot recognize data from file\n")
        print("# Shape of data: (%d,%d)\n" % 
              (imagePoints.shape[0], imagePoints.shape[1]))
    # get intrinsic (cmat, dvec)
    cmat = np.zeros((3, 3), dtype=float)
    cmat[0, 0] = 1111.1111
    cmat[1, 1] = 1111.1111
    cmat[0, 2] = 400.
    cmat[1, 2] = 300.
    cmat[2, 2] = 1
    dvec = np.zeros((5,1), dtype=float)
    ret_calibBladeTip = calibBladeTipCamPosGuess(
            radius, 
            imagePoints, 
            cmat, 
            dvec, 
            camPosGuess, 
            True)
    ret_calibBladeTip_OK = ret_calibBladeTip[0]
    rvec = ret_calibBladeTip[1].reshape(3, 1)
    tvec = ret_calibBladeTip[2].reshape(3, 1)
    objPoints = ret_calibBladeTip[3]
    prjPoints = ret_calibBladeTip[4]
    prjErrors = ret_calibBladeTip[5]
    cam_cal_extrinsic_parameter = ret_calibBladeTip[6]
    max_proj_error = ret_calibBladeTip[7]
    np.savez(save_calib_result_filepath, **cam_cal_extrinsic_parameter)
    max_proj_error_filepath = os.path.join(work_dir_path, "max_projection_error.txt")
    mpef = open(max_proj_error_filepath, 'w')
    print("The positive maximum projection error: ", max_proj_error, file=mpef)
    mpef.close()
    #
    # plot
    #       
    if (True):
        fig = plt.figure();
        plotStart = 0; 
        plotEnd = nPoints;  
        thePlt = plt.scatter(imagePoints[plotStart:plotEnd,0],
                             imagePoints[plotStart:plotEnd,1], s=1); 
        thePlt = plt.scatter(prjPoints[plotStart:plotEnd,0],
                             prjPoints[plotStart:plotEnd,1], s=1); 
        thePlt.axes.invert_yaxis()
        thePlt.axes.axis('equal')
        thePlt.axes.grid('on', linestyle='dashed')
        thePlt.axes.set_title('Projected Image Points')
        thePlt.axes.set_xlabel('Image x (pixel)')
        thePlt.axes.set_ylabel('Image y (pixel)')
        plt.show()
    if (True):
        fig = plt.figure();
        ax = fig.add_subplot(projection='3d')
        plotStart = 0; 
        plotEnd = nPoints;  
        thePlt = ax.plot3D(objPoints[plotStart:plotEnd,0], 
                           objPoints[plotStart:plotEnd,1],
                           objPoints[plotStart:plotEnd,2], 
                           'gray', 
                           lw=0, marker='.', ms = 2)
        ax.set_box_aspect([1,1,1])
        thePlt[0].axes.grid('on', linestyle='dashed')
        thePlt[0].axes.set_title('Projected Image Points')
        retSetXLabel = ax.set_xlabel('World coord. X (mm)')
        retSetYLabel = ax.set_ylabel('World coord. Y (mm)')
        retSetZLabel = ax.set_zlabel('World coord. Z (mm)')
        plt.show()
        
if __name__ == '__main__':
    work_dir_path = r"H:\CWH_thesis_experimental\PD_NV_NF_DCBT\intersection_quadratic_subpixel_level"
    point_position_filepath = r"point_position_quadratic_intersectionsubpixel"
    save_calib_result_filepath = r"_extrinsic_parameter"
    test2_calibBladeTip_camPosGuess(work_dir_path, os.path.join(work_dir_path, point_position_filepath + ".npz"), os.path.join(work_dir_path, point_position_filepath + save_calib_result_filepath + ".npz"))
    