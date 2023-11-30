#!/usr/bin/env python3
# Python script to generate poses in SE(3) 
# By: Aditya Patankar

# Open3D for point cloud processing and visualization
import numpy as np
from numpy import linalg as la
#import open3d as o3d

import csv
import math
import random
import actionlib
import rospy
import rosbag
from std_msgs.msg import Int32, String
import rospy
import numpy as np
import argparse
import time
import cv2
import os
import sys
import torch
from PIL import Image
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError



#
#
# Matplotlib libraries for plotting and visualization in Python:
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from matplotlib import cm
from geometry_msgs.msg import Pose
from scipy.spatial.transform import Rotation as Rot
from automated_image_capture.msg import MotionExecutionAction, MotionExecutionGoal, MotionExecutionActionFeedback, MotionExecutionActionResult
from sensor_msgs.msg import JointState, CameraInfo

import pyrealsense2 as rs
import numpy as np
import cv2
import heapq


bridge = CvBridge()

pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

class samplePoses(object):

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

        # Attributes associated with the computed end-effector poses:
        self.camera_pose =None
        self.poses_EE = None
        self.transformed_poses = None

        # Attributes associated with visualizing the point cloud:
        self.pcd = None
        self.cloud_points = None

    def samplePositions(self):
        self.x = np.reshape(np.random.normal(self.mu, self.sigma, self.num_points), self.num_points)
        self.y = np.reshape(np.random.normal(self.mu, self.sigma, self.num_points), self.num_points)
        self.z = np.reshape(np.random.normal(self.mu, self.sigma, self.num_points), self.num_points)
        self.points = np.zeros([self.num_points, self.dim])

        for i in range(self.num_points):
            self.points[i, :] = np.asarray([self.x[i], self.y[i], self.z[i]])

        # Normalizing the points. This process ensures that the sampled points are on the surface of a unit sphere:
        for i in range(self.num_points):
            point = self.points[i, :]
            self.points[i, :] = np.divide(point, la.norm(point))

        # Multiplying the points with the computed radius:
        self.points_updated = self.radius*self.points

        # Now selecting the points from a specific region/quadrants:
        self.points_selected = []
        for point in self.points_updated:
            x = point[0]
            # y = point[1]
            z = point[2]
            # if x < 0 and y < 0 and z > 0:
            if x < 0 and z > 0 or x > 0 and z > 0:
                self.points_selected.append(point)

        self.points_selected = np.asarray(self.points_selected)
    
    def samplePoses(self):
        # Computing the desired orientation at all the sampled positions:
        # x_axis = np.zeros([3])
        # x_axis[:] = np.reshape(self.initial_camera_pose_base[0:3, 1], [3])
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
            
    def computeEEPoses(self):
        self.transformed_end_effector_poses = []
        for pose in self.transformed_poses:
            ee_pose = np.matmul(pose, la.inv(self.g_camera_hand))
            self.transformed_end_effector_poses.append(ee_pose)
    def getNearestPoses(self):
        self.distance_threshold = self.radius + self.constant*self.radius
        self.nearest_poses = []
        for pose in self.transformed_end_effector_poses:
            position_vector = pose[0:3, 3]
            if la.norm(position_vector) < self.distance_threshold:
                self.nearest_poses.append(pose)


    
'''Function to read a CSV file:'''
def readCSV(filename):
    columns = []
    datapoints = []
    with open(filename) as file:
        csvreader = csv.reader(file)
        for col in csvreader:
            columns.append(col)

    for col in columns:
        datapoint = np.asarray(col, dtype = np.float64)
        datapoints.append(datapoint)
    return datapoints
    
def write_image(bag, bagname):
  try: 
    pipeline.start(config)
    frames = pipeline.wait_for_frames()
    color_frame = frames.get_color_frame()  
    if color_frame:
        color_image = np.asanyarray(color_frame.get_data())
        cv2.imwrite("captured_image" + bagname + ".jpg", color_image)
        image_msg = bridge.cv2_to_imgmsg(color_image, encoding="bgr8")
        bag.write("/camera/color_image" , image_msg)
        cv2.waitKey(1)
    depth_frame = frames.get_depth_frame()
    if depth_frame:
          depth_image = np.asanyarray(depth_frame.get_data())
          cv2.imwrite("captured_image" + bagname + ".jpg", depth_image)
          bag.write("/camera/depth_image" , depth_image)
          cv2.waitKey(1)
    pipeline.stop()
  finally:
    cv2.destroyAllWindows()
          
'''Function to plot a reference frame:'''
def plotReferenceFrame(R, p, scale_value, length_value, ax):
    ax.quiver(p[0], p[1], p[2], scale_value*R[0, 0], scale_value*R[1, 0], scale_value*R[2, 0], color = "r", arrow_length_ratio = length_value)
    ax.quiver(p[0], p[1], p[2], scale_value*R[0, 1], scale_value*R[1, 1], scale_value*R[2, 1], color = "g", arrow_length_ratio = length_value)
    ax.quiver(p[0], p[1], p[2], scale_value*R[0, 2], scale_value*R[1, 2], scale_value*R[2, 2], color = "b", arrow_length_ratio = length_value)
    
    return ax

def get_poses():
    sampling_object = samplePoses()
    sampling_object.g_camera_hand = np.asarray([[0.011474372443544,	-0.999880765552861,	0.008461769855726,	0.048111793315264],
                                                [0.999821675185035,	0.011356425256825,	-0.013850879699048,	-0.034177805441354],
                                                [0.013753003714653,	0.008624186680371,	0.999870533486379,	0.068706800907662],
                                                [0, 0,	0,	1]])

    # Loading the point cloud just for visualization purpose:
    #sampling_object.pcd = o3d.io.read_point_cloud("dominosugar_trial1_Processed_Transformed_segmented.ply")
    #sampling_object.cloud_points = np.asarray(sampling_object.pcd.points)

    # Defining the robot base reference frame:
    sampling_object.R_base = np.identity(3)
    sampling_object.p_base = np.zeros([3,1])

    # Reading the initial camera pose expressed with respect to the base reference frame:
    sampling_object.initial_camera_pose_base = np.asarray(readCSV('camera_pose_trial1.csv'))

    # Reading the camera pose obtained after moving the end-effector closer to the objects expressed with respect to the base reference frame:
    sampling_object.final_camera_pose_base = np.asarray(readCSV('camera_pose_trial2.csv'))

    # Updating the rotation matrix of the final camera pose with identity
    sampling_object.final_camera_pose_updated = np.identity(4)
    sampling_object.final_camera_pose_updated[0:3, 0:3] =  np.identity(3)
    sampling_object.final_camera_pose_updated[0:3, 3] = sampling_object.final_camera_pose_base[0:3, 3]

    # Transforming the initial camera pose from the base reference frame to the newly updated final camera pose:
    sampling_object.initial_camera_pose_transformed = np.matmul(la.inv(sampling_object.final_camera_pose_updated), sampling_object.initial_camera_pose_base)

    # Computing the radius after the initial camera pose has been transformed:
    sampling_object.p_initial_camera_pose_transformed = np.reshape(sampling_object.initial_camera_pose_transformed[0:3, 3], [3,1])
    # radius_vector = np.subtract()
    sampling_object.radius = la.norm(sampling_object.p_initial_camera_pose_transformed)

    # Generating points on the surface of the sphere centered at final_pose_updated:
    sampling_object.num_points = 1000
    sampling_object.dim = 3

    # Sampling random points from a Gaussian distribution
    sampling_object.mu, sampling_object.sigma = 0, 0.1
    
    sampling_object.samplePositions()
    sampling_object.samplePoses()
    sampling_object.transformToBase()

    sampling_object.computeEEPoses()
    sampling_object.constant = 0.2
    sampling_object.getNearestPoses()  
    
    
   
    '''
    distance_threshold = sampling_object.radius + constant*sampling_object.radius
    nearest_poses = []
    for pose in sampling_object.transformed_end_effector_poses:
        position_vector = pose[0:3, 3]
        if la.norm(position_vector) < distance_threshold:
            nearest_poses.append(pose)

   '''
    
    pose_coll=[]
    #pose_coll.append(sampling_object.initial_camera_pose_base )
    for pose in sampling_object.nearest_poses:
      rot_quat = Rot.from_matrix(pose[0:3, 0:3])
      rot_quat = rot_quat.as_quat()
      grasp_pose = Pose()
      grasp_pose.position.x = pose[0,3]
      grasp_pose.position.y = pose[1,3]
      grasp_pose.position.z = pose[2,3]

      grasp_pose.orientation.x = rot_quat[0]
      grasp_pose.orientation.y = rot_quat[1]
      grasp_pose.orientation.z = rot_quat[2]
      grasp_pose.orientation.w = rot_quat[3]
      pose_coll.append(grasp_pose)
      
    return pose_coll

'''
    counter = 10
    dist_map=[]
    import heapq
    
    
    motion_execution_client = actionlib.SimpleActionClient('PandaMotionExecutionActionServer',MotionExecutionAction)
    motion_execution_client.wait_for_server()
      
    
    i = 1
    print("Starting execution")
    while counter > 0 : 
      if not dist_map:
        pose = random.choices(pose_coll)
        pose = pose[0]
        dist = 0
        dist_map.append((dist,pose))
      heapq.heapify(dist_map)
      #print("AGAIN")
      pose = dist_map[0][1]
      #print(pose)
      #print(type(pose))
      goal = MotionExecutionGoal()
      goal.ee_trajectory = [pose]
      goal.gripper_state = [False]
      motion_execution_client.send_goal(goal)  
      motion_execution_client.wait_for_result()
      exe_result = motion_execution_client.get_result()
      print("At iteration: ", counter)
      if(exe_result.result == exe_result.SUCCESS):
        print("Next pose found")
        bag_name = f"bag_{i}.bag"  # Construct the bag name
        print(bag_name)
        i = i+1
        # Open the bag file for writing
        with rosbag.Bag(bag_name, 'w') as bag:
          joint_state_sub = rospy.wait_for_message("/joint_states", JointState)
          bag.write("/camera/joint_states", joint_state_sub)
          print("Obtained joint angles")
          camera_pose = rospy.wait_for_message("/panda_camera_pose", Pose)
          bag.write("/camera/camera_pose", camera_pose)
          print("Obtained camera pose")
          write_image(bag, bag_name)
          print("Obtained images")
          print("Checking for next pose")
          for j in range(1, len(pose_coll)):
            if pose_coll[j] == pose :
              continue
            distance = np.sqrt((pose_coll[j].position.x - pose.position.x)**2 + (pose_coll[j].position.y - pose.position.y)**2  + (pose_coll[j].position.z - pose.position.z)**2)
            #print(type(pose_coll[j]))
      heapq.heapify(dist_map)
      heapq.heappop(dist_map) 
      counter = counter -1
        
    return motion_execution_client.get_result()'''
   
def captureImagesAlongTrajectory(pose_coll, no_of_iterations):
    counter = no_of_iterations
    dist_map=[]
    
    motion_execution_client = actionlib.SimpleActionClient('PandaMotionExecutionActionServer',MotionExecutionAction)
    motion_execution_client.wait_for_server()

    i = 1  # for creating indexed bag values 
    print("Starting execution")
    while counter > 0 : 
      if not dist_map:
        pose = random.choices(pose_coll)
        pose = pose[0]
        dist = 0
        dist_map.append((dist,pose))
      heapq.heapify(dist_map)
      pose = dist_map[0][1]
      goal = MotionExecutionGoal()
      goal.ee_trajectory = [pose]
      goal.gripper_state = [False]
      motion_execution_client.send_goal(goal)  
      motion_execution_client.wait_for_result()
      exe_result = motion_execution_client.get_result()
      print("At iteration: ", counter)
      if(exe_result.result == exe_result.SUCCESS):
        print("Next pose found")
        bag_name = f"bag_{i}.bag"  # Construct the bag name
        print(bag_name)
        i = i+1
        # Open the bag file for writing
        with rosbag.Bag(bag_name, 'w') as bag:
          joint_state_sub = rospy.wait_for_message("/joint_states", JointState)
          bag.write("/camera/joint_states", joint_state_sub)
          print("Obtained joint angles")
          camera_pose = rospy.wait_for_message("/panda_camera_pose", Pose)
          bag.write("/camera/camera_pose", camera_pose)
          print("Obtained camera pose")
          write_image(bag, bag_name)
          print("Obtained images")
          print("Checking for next pose")
          for j in range(1, len(pose_coll)):
            if pose_coll[j] == pose :
              continue
            distance = np.sqrt((pose_coll[j].position.x - pose.position.x)**2 + (pose_coll[j].position.y - pose.position.y)**2  + (pose_coll[j].position.z - pose.position.z)**2)
      heapq.heapify(dist_map)
      heapq.heappop(dist_map) 
      counter = counter -1
        
    return motion_execution_client.get_result()
      

if __name__ == '__main__':
    rospy.init_node('main')
    poses = get_poses()
    #can also pass a custom pose list
    no_of_iterations = int(input("Enter the number of iterations: "))
    captureImagesAlongTrajectory(poses, no_of_iterations)
    
    
