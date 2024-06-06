# Python script to generate end effector poses in SE(3) for automated data collection using 
# an 'eye-in-hand' configuration
# By: Aditya Patankar

# Open3D for point cloud processing and visualization
import open3d as o3d

import numpy as np
from numpy import linalg as la
import csv
import math
import random

# Matplotlib libraries for plotting and visualization in Python:
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from matplotlib import cm

class samplePose(object):

    def __init__(self):

        # Base reference frame:
        self.R_base = None
        self.p_base = None

        # Initial camera reference frame:
        self.initial_camera_pose_base = None
        self.R_initial_camera_pose_base = None
        self.p_initial_camera_pose_base = None

        # Final camera reference frame:
        self.final_camera_pose_base = None
        self.final_camera_pose_base = None
        self.final_camera_pose_base = None

        self.final_camera_pose_updated = None
        self.R_final_camera_pose_updated = None
        self.p_final_camera_pose_updated = None

        # Pose of the camera reference frame with respect to the end-effector reference frame:
        self.g_camera_hand = None

        # Attributes associated with transforming the initial camera reference frame into the final reference frame:
        self.initial_camera_pose_transformed = None
        self.R_initial_camera_pose_transformed = None
        self.p_initial_camera_pose_transformed = None

        # Attributes associated with uniformly sampling points on the surface of a sphere centered at the camera reference frame:
        self.radius = None
        self.num_points = None
        self.dim = None
        self.mu = None
        self.sigma = None
        self.x = None
        self.y = None
        self.z = None
        self.points = None
        self.points_updated = None
        self.points_selected = None
        self.transformed_points = None

        # Attributes associated with the orientation information at each of the points sampled on the surface of the sphere:
        self.x_EE = None
        self.y_EE = None
        self.z_EE = None
        self.R_EE = None
        self.p_EE = None

        # Attributes associated with the computing the camera and end-effector poses:
        self.camera_pose =None
        self.poses_EE = None
        self.transformed_poses = None
        self.transformed_end_effector_poses = None
        self.constant = None
        self.distance_threshold = None
        self.nearest_poses = None

        # Attributes associated with visualizing the point cloud:
        self.pcd = None
        self.cloud_points = None

    '''Function to sample the camera positions from a Normal distribution:'''
    def samplePositions(self):
        # self.x = np.reshape(np.random.normal(self.mu, self.sigma, self.num_points), self.num_points)
        # self.y = np.reshape(np.random.normal(self.mu, self.sigma, self.num_points), self.num_points)
        # self.z = np.reshape(np.random.normal(self.mu, self.sigma, self.num_points), self.num_points)
        # self.points = np.zeros([self.num_points, self.dim])

        # for i in range(self.num_points):
        #     self.points[i, :] = np.asarray([self.x[i], self.y[i], self.z[i]])
        
        
        # Number of points
        num_points = self.num_points

        # Radius of the hemisphere
        r = 0.75 * self.radius 

        def fibonacci_sphere(samples=1000, radius=1, randomize=True):
            rnd = 1.
            if randomize:
                rnd = np.random.random() * samples

            points = []
            offset = 2./samples
            increment = np.pi * (3. - np.sqrt(5.))

            for i in range(samples):
                y = ((i * offset) - 1) + (offset / 2)
                r = np.sqrt(1 - y*y) * radius  # Scale the radius
                phi = ((i + rnd) % samples) * increment
                x = np.cos(phi) * r
                z = np.sin(phi) * r

                # Adjust to only keep points on the upper hemisphere
                points.append((x, y * radius, z))  # Apply radius scaling to y

            return np.array(points)


        points = fibonacci_sphere(num_points, r, randomize=False)

        # Convert points list to a numpy array for better handling
        points_array = np.array(points)


        self.points = points_array
      
        
        # Normalizing the points. This process ensures that the sampled points are on the surface of a unit sphere:
        # for i in range(self.num_points):
        #     point = self.points[i, :]
        #     self.points[i, :] = np.divide(point, la.norm(point))

        # Multiplying the points with the computed radius:
        # self.points_updated = self.radius*self.points

        # Now selecting the points from a specific region/quadrants:
        self.points_selected = []
        for point in self.points:
            x = point[0]
            # y = point[1]
            z = point[2]
            # if x < 0 and y < 0 and z > 0:
            # if (x < 0 and z > 0.3) or (x > 0 and z > 0.3):
            if x < 0 and z > 0 or x > 0 and z > 0:
                self.points_selected.append(point)

        self.points_selected = np.asarray(self.points_selected)
    
    '''Function to compute the orientation of the camera reference frame corresponding to the positons'''
    def samplePoses(self):
        self.poses_EE = []
        for p in self.points_selected:
            self.z_EE = np.reshape(-1*p, [3])
            self.z_EE /= la.norm(self.z_EE)
            # x_EE = x_axis
            self.x_EE = np.random.randn(3)
            self.x_EE -= self.x_EE.dot(self.z_EE)*self.z_EE
            self.x_EE /= la.norm(self.x_EE)
            self.y_EE = np.cross(self.z_EE, self.x_EE)
            self.R_EE = np.zeros([3,3])
            self.R_EE[:, 0] = self.x_EE
            self.R_EE[:, 1] = self.y_EE
            self.R_EE[:, 2] = self.z_EE
            self.gripper_pose = np.zeros([4,4])
            self.gripper_pose[0:3, 0:3] = self.R_EE
            self.gripper_pose[0:3, 3] = np.reshape(p, [3])
            self.gripper_pose[3,3] = 1
            self.poses_EE.append(self.gripper_pose)

        self.poses_EE = np.asarray(self.poses_EE)
    
    '''Function to transform the computed camera poses to the base reference frame of the robot:'''
    def transformToBase(self):
        # Transforming the selected locations back to the robot base reference frame:
        self.R_final_camera_pose_base = self.final_camera_pose_base[0:3, 0:3]
        self.p_final_camera_pose_base = self.final_camera_pose_base[0:3, 3]
        # transformed_points = np.zeros([vec_updated.shape[0], vec_updated.shape[1]])
        self.transformed_points = np.zeros([self.points_selected.shape[0], self.points_selected.shape[1]])
        self.transformed_poses = np.zeros([self.poses_EE.shape[0], self.poses_EE.shape[1], self.poses_EE.shape[2]])

        for i in range(len(self.points_selected)):
            # Transforming the position vector: 
            prod = np.dot(self.R_final_camera_pose_base, self.points_selected[i, :])
            result = np.subtract(self.p_final_camera_pose_base, prod)
            self.transformed_points[i, :] = np.reshape(result, [1,3])
            # Transforming the sampled camera poses to the base reference frame:
            pose_EE_base = np.matmul(self.final_camera_pose_updated, self.poses_EE[i, :, :])
            self.transformed_poses[i, :, :] = pose_EE_base

    '''Function to compute the end-effector poses corresponding to the computed camera poses:'''
    def computeEEPoses(self):
        self.transformed_end_effector_poses = []
        for pose in self.transformed_poses:
            ee_pose = np.matmul(pose, la.inv(self.g_camera_hand))
            self.transformed_end_effector_poses.append(ee_pose)
    
    '''Function to get the end-effector poses nearest to the robot base reference frame:'''
    def getNearestPoses(self):
        # Computing the 
        self.distance_threshold = self.radius + self.constant*self.radius
        self.nearest_poses = []
        for pose in self.transformed_end_effector_poses:
            position_vector = pose[0:3, 3]
            if la.norm(position_vector) < self.distance_threshold:
                self.nearest_poses.append(pose)