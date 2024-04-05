
import pyrealsense2 as rs2
import argparse
import cv2
import math

import numpy as np
from numpy import genfromtxt


class RealsenseSubscriber:
    
    '''Method to initialize the realsense with the bounding box, intrinsics, and image'''
    def __init__(self,in_params,in_model,in_coeff,depArr,depImg):
        self.intrinsics = None
        self.extrinsics = None
        self.depthScale =  0.001
        self.get_intrinsic_parameters(in_params,in_model,in_coeff)
        self.depthCallback(depArr,depImg)

    ''' Method to get intrinsic camera parameters required for projection and deprojection tasks'''
    def get_intrinsic_parameters(self,in_params,in_model,in_coeff):
        self.intrinsics = rs2.intrinsics()

        # reading params from file
        temp_params = genfromtxt(in_params, delimiter='\n')
        self.intrinsics.width = int(temp_params[0])
        self.intrinsics.height = int(temp_params[1])
        self.intrinsics.ppx = int(temp_params[2])
        self.intrinsics.ppy =int(temp_params[3])
        self.intrinsics.fx = int(temp_params[4])
        self.intrinsics.fy = int(temp_params[5])
        f= open(in_model,"r")
        temp_model = f.readline().split("\n")[0]
        if temp_model == 'plumb_bob':
            self.intrinsics.model = rs2.distortion.brown_conrady
        elif temp_model == 'equidistant':
            self.intrinsics.model = rs2.distortion.kannala_brandt4
        temp_coeffs = genfromtxt(in_coeff, delimiter='\n')
        self.intrinsics.coeffs = temp_coeffs
        print("Params Loaded!")

    '''Method to save the depth data '''
    def depthCallback(self,depArr,depImg):
        self.depthArray = genfromtxt(depArr, delimiter=',')
        self.depthImage = cv2.imread(depImg)
       

    '''Method to deproject a pixel based on the intrinsic camera parameters'''
    def deproject_pixel_to_point(self,img_mask):

        img_h, img_w = img_mask.shape
        print(img_h,img_w)

        self.index_pairs = []
        self.resultSemanticSegmentation = []
        for i in range(img_h):
            for j in range(img_w):
                if img_mask[i, j] != False:
                    self.index_pairs.append(np.array([i,j]))
                    depth = self.depthArray[i, j]
                    coord = rs2.rs2_deproject_pixel_to_point(self.intrinsics, [j,i], depth)
                    self.resultSemanticSegmentation.append(coord)
        if(self.resultSemanticSegmentation == []):
            return []
        self.MostRecentDeprojectedPoint = self.resultSemanticSegmentation[0]
        return self.resultSemanticSegmentation


    '''Method to transform the point from the pov of the first camera location to the second camera location'''
    def transform_point(self,pos1_extrinsics,pos2_extrinsics,point):

        # Point wrt first camera pose 
        c1_P = np.reshape(point,[3,1])

        # Transform matrix computation
        c2_T_c1 = np.matmul(np.linalg.inv(pos2_extrinsics),pos1_extrinsics)

        # Extracting the rotation matrix and translation vector from transform matrix
        c2_R_c1 = c2_T_c1[0:3,0:3]
        c2_P_c1 = np.reshape(c2_T_c1[0:3,3],[3,1])

        # Computing the point based on the second camera pose using the Transform rotation matrix  and translation vector
        self.MostRecentTransformedPoint = np.add(np.matmul(c2_R_c1,c1_P),c2_P_c1)

        return self.MostRecentTransformedPoint

    '''Method to project the newly obtained point to pixel'''
    def project_point_to_pixel(self,point,in_params2,in_model2,in_coeff2):
        intrinsics = rs2.intrinsics()

        # reading params from file
        temp_params = genfromtxt(in_params2, delimiter='\n')
        intrinsics.width = int(temp_params[0])
        intrinsics.height = int(temp_params[1])
        intrinsics.ppx = int(temp_params[2])
        intrinsics.ppy =int(temp_params[3])
        intrinsics.fx = int(temp_params[4])
        intrinsics.fy = int(temp_params[5])
        f= open(in_model2,"r")
        temp_model = f.readline().split("\n")[0]
        if temp_model == 'plumb_bob':
            intrinsics.model = rs2.distortion.brown_conrady
        elif temp_model == 'equidistant':
            intrinsics.model = rs2.distortion.kannala_brandt4
        temp_coeffs = genfromtxt(in_coeff2, delimiter='\n')
        intrinsics.coeffs = temp_coeffs

        self.MostRecentProjectedPoint = rs2.rs2_project_point_to_pixel(intrinsics,point)
        return self.MostRecentProjectedPoint
    