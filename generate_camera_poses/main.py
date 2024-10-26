# Python script to generate poses in SE(3) 
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

from sample_camera_poses import samplePose
import sys
sys.path.append('Task_Oriented_Grasping_from_Point_Cloud_Representation')



from Task_Oriented_Grasping_from_Point_Cloud_Representation.main_pivoting import main_pivoting_main


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

'''Function to plot a reference frame:'''
def plotReferenceFrame(R, p, scale_value, length_value, ax):
    ax.quiver(p[0], p[1], p[2], scale_value*R[0, 0], scale_value*R[1, 0], scale_value*R[2, 0], color = "r", arrow_length_ratio = length_value)
    ax.quiver(p[0], p[1], p[2], scale_value*R[0, 1], scale_value*R[1, 1], scale_value*R[2, 1], color = "g", arrow_length_ratio = length_value)
    ax.quiver(p[0], p[1], p[2], scale_value*R[0, 2], scale_value*R[1, 2], scale_value*R[2, 2], color = "b", arrow_length_ratio = length_value)
    
    return ax

if __name__ == '__main__':

    # Initializing an object of the samplePoses class:
    sampling_object = samplePose()

    # Pose of the camera reference frame with respect to the end-effector reference frame:
    # NOTE: The following 4x4 matrix can be readily obtained by subscribing the corresponding topic.
    sampling_object.g_camera_hand = np.asarray([[0.011474372443544,	-0.999880765552861,	0.008461769855726,	0.048111793315264],
                                                [0.999821675185035,	0.011356425256825,	-0.013850879699048,	-0.034177805441354],
                                                [0.013753003714653,	0.008624186680371,	0.999870533486379,	0.068706800907662],
                                                [0, 0,	0,	1]])

    # Loading the point cloud just for visualization purpose:
    sampling_object.pcd = o3d.io.read_point_cloud("dominosugar_trial1_Processed_Transformed_segmented.ply")
    sampling_object.cloud_points = np.asarray(sampling_object.pcd.points)

    # Defining the robot base reference frame:
    sampling_object.R_base = np.identity(3)
    sampling_object.p_base = np.zeros([3,1])

    # Reading the initial camera pose expressed with respect to the base reference frame:
    sampling_object.initial_camera_pose_base = np.asarray(readCSV('camera_pose_trial1.csv'))

    # Reading the camera pose obtained after moving the end-effector closer to the objects expressed with respect to the base reference frame:
    #sampling_object.final_camera_pose_base = main_pivoting_main("object_processed_base_frame_12.ply")
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
    sampling_object.num_points = 2000
    sampling_object.dim = 3

    # Sampling random points from a Gaussian distribution
    sampling_object.mu, sampling_object.sigma = 0, 0.1
    
    sampling_object.samplePositions()
    sampling_object.samplePoses()
    sampling_object.transformToBase()

    # Computing the end-effector poses corresponding to the computed camera poses:
    sampling_object.computeEEPoses()

    # Getting the poses nearest to the robot base reference frame. Please note that we only use the 
    # position vector corresponding to a particular pose while computing nearest poses. 
    sampling_object.constant = 0.2

    sampling_object.getNearestPoses()
    '''distance_threshold = sampling_object.radius + constant*sampling_object.radius
    nearest_poses = []
    for pose in sampling_object.transformed_end_effector_poses:
        position_vector = pose[0:3, 3]
        if la.norm(position_vector) < distance_threshold:
            nearest_poses.append(pose)
'''
    ###################### DATA PROCESSING FOR VISUALIZATION ######################

    x_points = np.reshape(sampling_object.cloud_points[:, 0], [sampling_object.cloud_points.shape[0],1])
    y_points = np.reshape(sampling_object.cloud_points[:, 1], [sampling_object.cloud_points.shape[0],1])
    z_points = np.reshape(sampling_object.cloud_points[:, 2], [sampling_object.cloud_points.shape[0],1])

    x_points_vec = np.reshape(sampling_object.points[:, 0], [sampling_object.points.shape[0],1])
    y_points_vec = np.reshape(sampling_object.points[:, 1], [sampling_object.points.shape[0],1])
    z_points_vec = np.reshape(sampling_object.points[:, 2], [sampling_object.points.shape[0],1])

    # x_points_vec_updated = np.reshape(sampling_object.points_updated[:, 0], [sampling_object.points_updated.shape[0],1])
    # y_points_vec_updated = np.reshape(sampling_object.points_updated[:, 1], [sampling_object.points_updated.shape[0],1])
    # z_points_vec_updated = np.reshape(sampling_object.points_updated[:, 2], [sampling_object.points_updated.shape[0],1])

    x_points_vec_selected = np.reshape(sampling_object.points_selected[:, 0], [sampling_object.points_selected.shape[0],1])
    y_points_vec_selected = np.reshape(sampling_object.points_selected[:, 1], [sampling_object.points_selected.shape[0],1])
    z_points_vec_selected = np.reshape(sampling_object.points_selected[:, 2], [sampling_object.points_selected.shape[0],1])

    x_transformed_points = np.reshape(sampling_object.transformed_points[:, 0], [sampling_object.transformed_points.shape[0],1])
    y_transformed_points = np.reshape(sampling_object.transformed_points[:, 1], [sampling_object.transformed_points.shape[0],1])
    z_transformed_points = np.reshape(sampling_object.transformed_points[:, 2], [sampling_object.transformed_points.shape[0],1])

    ###################### VISUALIZATION ###################### 


    ## Plot 7: 
    fig7 = plt.figure()
    ax7 = fig7.add_subplot(projection='3d')

    # Base reference Frame:
    ax7 = plotReferenceFrame(sampling_object.R_base, sampling_object.p_base, 0.25, 0.15, ax7)

    # Initial camera reference frame expressed with respect to the base reference frame:
    sampling_object.R_initial_camera_pose_base = sampling_object.initial_camera_pose_base[0:3, 0:3]
    sampling_object.p_initial_camera_pose_base = sampling_object.initial_camera_pose_base[0:3, 3]
    ax7 = plotReferenceFrame(sampling_object.R_initial_camera_pose_base, sampling_object.p_initial_camera_pose_base, 0.15, 0.15, ax7)

    # Plotting the entire point cloud:
    ax7.scatter(x_points, y_points, z_points, s = 0.2)

    # Plotting the transformed new camera positions:
    ax7.scatter(x_transformed_points, y_transformed_points, z_transformed_points, s = 0.2)

    # ax7.scatter(x_transformed_points[100:120], y_transformed_points[100:120], z_transformed_points[100:120], color='blue', s=50)

    # ax7.scatter(x_transformed_points[360:380], y_transformed_points[360:380], z_transformed_points[360:380], color='red', s=50)

    # ax7.scatter(x_transformed_points[570:590], y_transformed_points[570:590], z_transformed_points[570:590], color='green', s=50)

    ax7.set_xlabel('X')
    ax7.set_ylabel('Y')
    ax7.set_zlabel('Z')
    ax7.set_xlim(-0.8, 0.8)
    ax7.set_ylim(-0.8, 0.8)
    ax7.set_zlim(-0.8, 0.8)

    ## Plot 8: 
    fig8 = plt.figure()
    ax8 = fig8.add_subplot(projection='3d')
    # Base reference Frame:
    ax8 = plotReferenceFrame(sampling_object.R_base, sampling_object.p_base, 0.25, 0.15, ax8)
    # Initial camera reference frame expressed with respect to the base reference frame:
    ax8 = plotReferenceFrame(sampling_object.R_initial_camera_pose_base, sampling_object.p_initial_camera_pose_base, 0.15, 0.15, ax8)
    # Plotting the points of the point cloud:
    ax8.scatter(x_points, y_points, z_points, s = 0.2)
    # Sampled camera reference poses:
    for i in range(0,500,6):
        pose = sampling_object.transformed_poses[i, :, :]
        R = pose[0:3, 0:3]
        p = pose[0:3, 3]
        ax8 = plotReferenceFrame(R, p, 0.15, 0.15, ax8)

    ax8.set_xlabel('X')
    ax8.set_ylabel('Y')
    ax8.set_zlabel('Z')
    ax8.set_xlim(-0.8, 1)
    ax8.set_ylim(-0.8, 1)
    ax8.set_zlim(-0.8, 1)

    ## Plot 9: 
    fig9 = plt.figure()
    ax9 = fig9.add_subplot(projection='3d')
    # Base reference Frame:
    ax9 = plotReferenceFrame(sampling_object.R_base, sampling_object.p_base, 0.25, 0.15, ax9)
    # Initial camera reference frame expressed with respect to the base reference frame:
    ax9 = plotReferenceFrame(sampling_object.R_initial_camera_pose_base, sampling_object.p_initial_camera_pose_base, 0.15, 0.15, ax9)
    # Plotting the points of the point cloud:
    ax9.scatter(x_points, y_points, z_points, s = 0.2)
    # Sampled end-effector poses:
    for i in range(0,500,6):
        pose = sampling_object.transformed_end_effector_poses[i]
        R = pose[0:3, 0:3]
        p = pose[0:3, 3]
        ax9 = plotReferenceFrame(R, p, 0.15, 0.15, ax9)

    ax9.set_xlabel('X')
    ax9.set_ylabel('Y')
    ax9.set_zlabel('Z')
    ax9.set_xlim(-0.8, 1)
    ax9.set_ylim(-0.8, 1)
    ax9.set_zlim(-0.8, 1)

    ## Plot 10:
    # Plot showing the selected points nearest to the base reference frame:
    fig10 = plt.figure()
    ax10 = fig10.add_subplot(projection='3d')
    # Base reference Frame:
    ax10 = plotReferenceFrame(sampling_object.R_base, sampling_object.p_base, 0.25, 0.15, ax10)
    # Initial camera reference frame expressed with respect to the base reference frame:
    ax10 = plotReferenceFrame(sampling_object.R_initial_camera_pose_base, sampling_object.p_initial_camera_pose_base, 0.15, 0.15, ax10)
    # Plotting the points of the point cloud:
    ax10.scatter(x_points, y_points, z_points, s = 0.2)
    # Sampled end-effector poses:
    for i in range(len(sampling_object.nearest_poses)):
        pose = sampling_object.nearest_poses[i]
        R = pose[0:3, 0:3]
        p = pose[0:3, 3]
        ax10 = plotReferenceFrame(R, p, 0.15, 0.15, ax10)

    print(len(sampling_object.nearest_poses))

    ax10.set_xlabel('X')
    ax10.set_ylabel('Y')
    ax10.set_zlabel('Z')
    ax10.set_xlim(-0.8, 1)
    ax10.set_ylim(-0.8, 1)
    ax10.set_zlim(-0.8, 1)

    plt.show()