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

'''Function to sample positions an corresponding orientation on a sphere: '''
def samplePoses():
    return None

if __name__ == '__main__':

    # Loading the point cloud just for visualization purpose:
    pcd = o3d.io.read_point_cloud("dominosugar_trial1_Processed_Transformed_segmented.ply")
    cloud_points = np.asarray(pcd.points)

    # Defining the robot base reference frame:
    R_base = np.identity(3)
    p_base = np.zeros([3,1])

    # Reading the initial camera pose expressed with respect to the base reference frame:
    initial_camera_pose_base = np.asarray(readCSV('camera_pose_trial1.csv'))

    # Reading the camera pose obtained after moving the end-effector closer to the objects expressed with respect to the base reference frame:
    final_camera_pose_base = np.asarray(readCSV('camera_pose_trial2.csv'))

    # Updating the rotation matrix of the final camera pose with identity
    final_camera_pose_updated = np.identity(4)
    final_camera_pose_updated[0:3, 0:3] =  np.identity(3)
    final_camera_pose_updated[0:3, 3] = final_camera_pose_base[0:3, 3]

    # Transforming the initial camera pose from the base reference frame to the newly updated final camera pose:
    initial_camera_pose_updated = np.matmul(la.inv(final_camera_pose_updated), initial_camera_pose_base)

    # Computing the radius after the initial camera pose has been transformed:
    p_initial_updated = np.reshape(initial_camera_pose_updated[0:3, 3], [3,1])
    # radius_vector = np.subtract()
    radius = la.norm(p_initial_updated)

    print("radius: ", radius)

    # Generating points on the surface of the sphere centered at final_pose_updated:
    num_points = 1000
    dim = 3

    # Sampling random points from a Gaussian distribution
    mu, sigma = 0, 0.1
    x = np.reshape(np.random.normal(mu, sigma, num_points), num_points)
    y = np.reshape(np.random.normal(mu, sigma, num_points), num_points)
    z = np.reshape(np.random.normal(mu, sigma, num_points), num_points)
    vec = np.zeros([num_points, dim])

    for i in range(num_points):
        vec[i, :] = np.asarray([x[i], y[i], z[i]])

    # Normalizing the points. This process ensures that the sampled points are on the surface of a unit sphere:
    for i in range(num_points):
        point = vec[i, :]
        vec[i, :] = np.divide(point, la.norm(point))

    # Multiplying the points with the computed radius:
    vec_updated = radius*vec

    # Now selecting the points from a specific region/quadrants:
    vec_selected = []
    for point in vec_updated:
        x = point[0]
        # y = point[1]
        z = point[2]
        # if x < 0 and y < 0 and z > 0:
        if x < 0 and z > 0 or x > 0 and z > 0:
            vec_selected.append(point)

    vec_selected = np.asarray(vec_selected)

    # Computing the desired orientation at all the sampled positions:
    x_axis = np.zeros([3])
    x_axis[:] = np.reshape(initial_camera_pose_base[0:3, 1], [3])
    poses_EE = []
    for p in vec_selected:
        r = la.norm(p)
        z_EE = np.reshape(-1*p, [3])
        z_EE /= la.norm(z_EE)
        # x_EE = x_axis
        x_EE = np.random.randn(3)
        x_EE -= x_EE.dot(z_EE)*z_EE
        x_EE /= la.norm(x_EE)
        y_EE = np.cross(z_EE, x_EE)
        R_EE = np.zeros([3,3])
        R_EE[:, 0] = x_EE
        R_EE[:, 1] = y_EE
        R_EE[:, 2] = z_EE
        gripper_pose = np.zeros([4,4])
        gripper_pose[0:3, 0:3] = R_EE
        gripper_pose[0:3, 3] = np.reshape(p, [3])
        gripper_pose[3,3] = 1
        poses_EE.append(gripper_pose)

        # Checking if the unit vectors are normalized: 
        print('x norm: ', la.norm(x_EE))
        print('y norm: ', la.norm(y_EE))
        print('z norm: ', la.norm(z_EE))

    poses_EE = np.asarray(poses_EE)
    print("poses_EE shape: ", poses_EE.shape)
    print("vec_selected shape: ", vec_selected.shape)


    # Transforming the selected locations back to the robot base reference frame:
    R_final_camera_pose_base = final_camera_pose_base[0:3, 0:3]
    p_final_camera_pose_base = final_camera_pose_base[0:3, 3]
    # transformed_points = np.zeros([vec_updated.shape[0], vec_updated.shape[1]])
    transformed_points = np.zeros([vec_selected.shape[0], vec_selected.shape[1]])
    transformed_poses = np.zeros([poses_EE.shape[0], poses_EE.shape[1], poses_EE.shape[2]])

    for i in range(len(vec_selected)):
        # Transforming the position vector 
        prod = np.dot(R_final_camera_pose_base, vec_selected[i, :])
        result = np.subtract(p_final_camera_pose_base, prod)
        transformed_points[i, :] = np.reshape(result, [1,3])
        pose_EE_base = np.matmul(final_camera_pose_updated, poses_EE[i, :, :])
        transformed_poses[i, :, :] = pose_EE_base
    
    ###################### DATA PROCESSING FOR VISUALIZATION ######################

    x_points = np.reshape(cloud_points[:, 0], [cloud_points.shape[0],1])
    y_points = np.reshape(cloud_points[:, 1], [cloud_points.shape[0],1])
    z_points = np.reshape(cloud_points[:, 2], [cloud_points.shape[0],1])

    x_points_vec = np.reshape(vec[:, 0], [vec.shape[0],1])
    y_points_vec = np.reshape(vec[:, 1], [vec.shape[0],1])
    z_points_vec = np.reshape(vec[:, 2], [vec.shape[0],1])

    x_points_vec_updated = np.reshape(vec_updated[:, 0], [vec_updated.shape[0],1])
    y_points_vec_updated = np.reshape(vec_updated[:, 1], [vec_updated.shape[0],1])
    z_points_vec_updated = np.reshape(vec_updated[:, 2], [vec_updated.shape[0],1])

    x_points_vec_selected = np.reshape(vec_selected[:, 0], [vec_selected.shape[0],1])
    y_points_vec_selected = np.reshape(vec_selected[:, 1], [vec_selected.shape[0],1])
    z_points_vec_selected = np.reshape(vec_selected[:, 2], [vec_selected.shape[0],1])

    x_transformed_points = np.reshape(transformed_points[:, 0], [transformed_points.shape[0],1])
    y_transformed_points = np.reshape(transformed_points[:, 1], [transformed_points.shape[0],1])
    z_transformed_points = np.reshape(transformed_points[:, 2], [transformed_points.shape[0],1])

    ###################### VISUALIZATION ###################### 

    ## Plot 1: 
    fig1 = plt.figure()
    ax1 = fig1.add_subplot(projection='3d')

    # Base reference Frame:
    ax1 = plotReferenceFrame(R_base, p_base, 0.25, 0.15, ax1)
    
    # Initial camera reference frame expressed with respect to the base reference frame:
    R_initial_camera_pose_base = initial_camera_pose_base[0:3, 0:3]
    p_initial_camera_pose_base = initial_camera_pose_base[0:3, 3]
    ax1 = plotReferenceFrame(R_initial_camera_pose_base, p_initial_camera_pose_base, 0.15, 0.15, ax1)

    # Final camera reference frame expressed with respect to the base reference frame:
    R_final_camera_pose_base = final_camera_pose_base[0:3, 0:3]
    p_final_camera_pose_base = final_camera_pose_base[0:3, 3]
    ax1 = plotReferenceFrame(R_final_camera_pose_base, p_final_camera_pose_base, 0.15, 0.15, ax1)

    # Plotting the points of the point cloud:
    ax1.scatter(x_points, y_points, z_points, s = 0.2)

    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('Z')
    ax1.set_xlim(-0.8, 0.8)
    ax1.set_ylim(-0.8, 0.8)
    ax1.set_zlim(-0.8, 0.8)

    ## Plot 2: 
    fig2 = plt.figure()
    ax2 = fig2.add_subplot(projection='3d')

    # Base reference Frame:
    ax2 = plotReferenceFrame(R_base, p_base, 0.25, 0.15, ax2)
    
    # Initial camera reference frame expressed with respect to the base reference frame:
    R_initial_camera_pose_base = initial_camera_pose_base[0:3, 0:3]
    p_initial_camera_pose_base = initial_camera_pose_base[0:3, 3]
    ax2 = plotReferenceFrame(R_initial_camera_pose_base, p_initial_camera_pose_base, 0.15, 0.15, ax2)

    # Final camera reference frame expressed with respect to the base reference frame:
    R_final_camera_pose_updated = final_camera_pose_updated[0:3, 0:3]
    p_final_camera_pose_updated = final_camera_pose_updated[0:3, 3]
    ax2 = plotReferenceFrame(R_final_camera_pose_updated, p_final_camera_pose_updated, 0.15, 0.15, ax2)

    # Plotting the points of the point cloud:
    ax2.scatter(x_points, y_points, z_points, s = 0.2)

    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    ax2.set_zlabel('Z')
    ax2.set_xlim(-0.8, 0.8)
    ax2.set_ylim(-0.8, 0.8)
    ax2.set_zlim(-0.8, 0.8)

    ## Plot 3: 
    fig3 = plt.figure()
    ax3 = fig3.add_subplot(projection='3d')

    # Base reference Frame:
    ax3 = plotReferenceFrame(R_base, p_base, 0.25, 0.15, ax3)
    
    # Initial camera reference frame expressed with respect to the base reference frame:
    R_initial_camera_pose_updated = initial_camera_pose_updated[0:3, 0:3]
    p_initial_camera_pose_updated = initial_camera_pose_updated[0:3, 3]
    ax3 = plotReferenceFrame(R_initial_camera_pose_updated, p_initial_camera_pose_updated, 0.15, 0.15, ax3)

    ax3.set_xlabel('X')
    ax3.set_ylabel('Y')
    ax3.set_zlabel('Z')
    ax3.set_xlim(-0.8, 0.8)
    ax3.set_ylim(-0.8, 0.8)
    ax3.set_zlim(-0.8, 0.8)

    ## Plot 4:
    fig4 = plt.figure()
    ax4 = fig4.add_subplot(projection='3d')

    # Base reference Frame:
    ax4 = plotReferenceFrame(R_base, p_base, 0.25, 0.15, ax4)
    
    # Initial camera reference frame expressed with respect to the base reference frame:
    R_initial_camera_pose_updated = initial_camera_pose_updated[0:3, 0:3]
    p_initial_camera_pose_updated = initial_camera_pose_updated[0:3, 3]
    ax4 = plotReferenceFrame(R_initial_camera_pose_updated, p_initial_camera_pose_updated, 0.15, 0.15, ax4)

    # Plotting the initially sampled points on the surface of the sphere:
    ax4.scatter(x_points_vec, y_points_vec, z_points_vec, s = 0.2)

    ax4.set_xlabel('X')
    ax4.set_ylabel('Y')
    ax4.set_zlabel('Z')
    ax4.set_xlim(-0.8, 0.8)
    ax4.set_ylim(-0.8, 0.8)
    ax4.set_zlim(-0.8, 0.8)


    ## Plot 5:
    fig5 = plt.figure()
    ax5 = fig5.add_subplot(projection='3d')

    # Base reference Frame:
    ax5 = plotReferenceFrame(R_base, p_base, 0.25, 0.15, ax5)
    
    # Initial camera reference frame expressed with respect to the base reference frame:
    R_initial_camera_pose_updated = initial_camera_pose_updated[0:3, 0:3]
    p_initial_camera_pose_updated = initial_camera_pose_updated[0:3, 3]
    ax5 = plotReferenceFrame(R_initial_camera_pose_updated, p_initial_camera_pose_updated, 0.15, 0.15, ax5)

    # Plotting the initially sampled points on the surface of the sphere:
    ax5.scatter(x_points_vec_updated, y_points_vec_updated, z_points_vec_updated, s = 0.2)

    ax5.set_xlabel('X')
    ax5.set_ylabel('Y')
    ax5.set_zlabel('Z')
    ax5.set_xlim(-0.8, 0.8)
    ax5.set_ylim(-0.8, 0.8)
    ax5.set_zlim(-0.8, 0.8)

    ## Plot 6:
    fig6 = plt.figure()
    ax6 = fig6.add_subplot(projection='3d')

    # Base reference Frame:
    ax6 = plotReferenceFrame(R_base, p_base, 0.25, 0.15, ax6)
    
    # Initial camera reference frame expressed with respect to the base reference frame:
    R_initial_camera_pose_updated = initial_camera_pose_updated[0:3, 0:3]
    p_initial_camera_pose_updated = initial_camera_pose_updated[0:3, 3]
    ax6 = plotReferenceFrame(R_initial_camera_pose_updated, p_initial_camera_pose_updated, 0.15, 0.15, ax6)

    # Plotting the initially sampled points on the surface of the sphere:
    ax6.scatter(x_points_vec_selected, y_points_vec_selected, z_points_vec_selected, s = 0.2)

    # Sampled camera reference poses:
    for i in range(100,150):
        pose = poses_EE[i, :, :]
        R = pose[0:3, 0:3]
        p = pose[0:3, 3]
        ax6 = plotReferenceFrame(R, p, 0.15, 0.15, ax6)

    ax6.set_xlabel('X')
    ax6.set_ylabel('Y')
    ax6.set_zlabel('Z')
    ax6.set_xlim(-0.8, 0.8)
    ax6.set_ylim(-0.8, 0.8)
    ax6.set_zlim(-0.8, 0.8)


    ## Plot 7: 
    fig7 = plt.figure()
    ax7 = fig7.add_subplot(projection='3d')

    # Base reference Frame:
    ax7 = plotReferenceFrame(R_base, p_base, 0.25, 0.15, ax7)
    
    # Initial camera reference frame expressed with respect to the base reference frame:
    R_initial_camera_pose_base = initial_camera_pose_base[0:3, 0:3]
    p_initial_camera_pose_base = initial_camera_pose_base[0:3, 3]
    ax7 = plotReferenceFrame(R_initial_camera_pose_base, p_initial_camera_pose_base, 0.15, 0.15, ax7)

    # Plotting the points of the point cloud:
    ax7.scatter(x_points, y_points, z_points, s = 0.2)

    # Plotting the transformed new camera positions:
    ax7.scatter(x_transformed_points, y_transformed_points, z_transformed_points, s = 0.2)

    ax7.set_xlabel('X')
    ax7.set_ylabel('Y')
    ax7.set_zlabel('Z')
    ax7.set_xlim(-0.8, 0.8)
    ax7.set_ylim(-0.8, 0.8)
    ax7.set_zlim(-0.8, 0.8)

    ## Plot 7: 
    fig8 = plt.figure()
    ax8 = fig8.add_subplot(projection='3d')

    # Base reference Frame:
    ax8 = plotReferenceFrame(R_base, p_base, 0.25, 0.15, ax8)
    
    # Initial camera reference frame expressed with respect to the base reference frame:
    R_initial_camera_pose_base = initial_camera_pose_base[0:3, 0:3]
    p_initial_camera_pose_base = initial_camera_pose_base[0:3, 3]
    ax8 = plotReferenceFrame(R_initial_camera_pose_base, p_initial_camera_pose_base, 0.15, 0.15, ax8)

    # Plotting the points of the point cloud:
    ax8.scatter(x_points, y_points, z_points, s = 0.2)

    # Sampled camera reference poses:
    for i in range(100,150):
        pose = transformed_poses[i, :, :]
        R = pose[0:3, 0:3]
        p = pose[0:3, 3]
        ax8 = plotReferenceFrame(R, p, 0.15, 0.15, ax8)

    ax8.set_xlabel('X')
    ax8.set_ylabel('Y')
    ax8.set_zlabel('Z')
    ax8.set_xlim(-0.8, 1)
    ax8.set_ylim(-0.8, 1)
    ax8.set_zlim(-0.8, 1)



    plt.show()