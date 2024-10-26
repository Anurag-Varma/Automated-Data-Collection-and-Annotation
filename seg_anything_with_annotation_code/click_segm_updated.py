import argparse
import torch
import cv2
import os
import numpy as np
from PIL import Image
from  matplotlib import pyplot as plt
from torchvision.transforms import Compose, Resize, ToTensor, Normalize
from torchvision.transforms import InterpolationMode
BICUBIC = InterpolationMode.BICUBIC

from segment_anything import sam_model_registry, SamPredictor
from numpy import savetxt
from numpy import genfromtxt
from numpy import linalg as la
from MaskGeneratorClass import MaskGenerator 
import tkinter as tk
from tkinter import PhotoImage
from tkinter import ttk, messagebox, filedialog
from PIL import Image, ImageTk
from pathlib import Path

# Open3D for point cloud processing and visualization
import open3d as o3d

from detection import RealsenseSubscriber
from process_point_cloud_baseline1 import pointCloud
import math

# cv2_to_imageTK read image and return ImageTk.PhotoImage
def cv2_to_imageTK(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGBA)
    imagePIL = Image.fromarray(image)
    imgtk = ImageTk.PhotoImage(image= imagePIL)
    return imgtk
        
# show_mask shows the mask on the image and returns the image with mask
def show_mask(mask, img, random_color=False, opacity=0.6):
    if random_color:
        color = np.concatenate([np.random.random(3)*255], axis=0)
    else:
        color = np.array([30, 144, 255])
    h, w = mask.shape[-2:]
    color_seg = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    fg_mask = mask != False
    
    img[fg_mask] = color_seg[fg_mask] * opacity
    
    return img

# show_mask_box gives the box around the image based on minX, minY, maxX, maxY
def show_mask_box(img, box):
    x0, y0, x1, y1 = box
    cv2.rectangle(img, (int(x0), int(y0)), (int(x1), int(y1)), (0, 255, 0), 2)
    
    return img

# find_centroid_from_coordinates returns the centroid of the downsampled mask from sam output
def find_centroid_from_coordinates(coords):
    # Convert the coordinates to numpy array
    coords_array = np.array(coords)

    # Calculate the centroid coordinates
    centroid_x = int(np.mean(coords_array[:, 0]))
    centroid_y = int(np.mean(coords_array[:, 1]))

    return centroid_x, centroid_y

# clear_stacks clears previous undo_mask_stack, redo_mask_stack, and keeps 'zeroMask.setCoordinates(0,0)' in undo_mask_stack, redo_mask_stack
def clear_stacks(undo_mask_stack, redo_mask_stack,predictor,img,label):
    print("clear stacks")
    undo_mask_stack.clear()
    redo_mask_stack.clear()
    zeroMask  = MaskGenerator(predictor)
    zeroMask.setCoordinates(0,0)
    undo_mask_stack.append(zeroMask)
    redo_mask_stack.append(zeroMask)
    imgtk = cv2_to_imageTK(img)
    label.imgtk = imgtk
    label.configure(image = imgtk)
   

   


# Save the union of all the mask to a file and also display transformed masks output
def save_all_masks_to_file_and_transform(undo_mask_stack,data_path,predictor, ui_pil_img):

    file_name = filedialog.asksaveasfilename(defaultextension=".csv",
                                                filetypes=[("csv file", ".csv")],
       
                                             )
    # print(undo_mask_stack)
    # print("no of masks",len(undo_mask_stack))
    # for point in undo_mask_stack:
    #     print(point.x,point.y)
    # masks = undo_mask_stack[-1].mask
    # print(masks)
    # print("no of masks",len(masks))
    undo_mask_stack = undo_mask_stack[1:]
    # Iterate from 2nd folder to 8th folder from Spring_24_Data
    previous_masks = []
    for obj in undo_mask_stack:
        previous_masks.append(obj.mask)

    for trans_cnt in range(2,9):

        # Get the following files and assume they are related to 1st image information
        extrinsics1 = data_path+"pose_"+str(trans_cnt-1)+"/"+"camera_pose.csv"
        in_params = data_path+"/"+"intrinsic_params.csv"
        in_model = data_path+"/"+"distortion_model.csv"
        in_coeff = data_path+"/"+"intrinsic_coeffs.csv"
        depImg = data_path+"pose_"+str(trans_cnt-1)+"/"+"depth_image_pixel_transform.png"
        depArr = data_path+"pose_"+str(trans_cnt-1)+"/"+"depth_array.csv"
        image_path = cv2.imread(data_path+"pose_"+str(trans_cnt-1)+"/item_image.png")
        image2 = cv2.imread(data_path+"pose_"+str(trans_cnt)+"/item_image.png")
        transformed_masks = []
        boxes = []
        centroids = []

        # Iterate through each mask in the stack
        for mask_index, mask in enumerate(previous_masks):

            print("save most recent mask to file")

            rsObj = RealsenseSubscriber(in_params,in_model,in_coeff,depArr,depImg)

            savetxt(file_name, mask, delimiter=',')

            # Call deproject project and transform code 
            extrinsinc_pos1 = genfromtxt(extrinsics1, delimiter=',')

            img_mask =  genfromtxt(file_name,delimiter=",")
            result_arr = rsObj.deproject_pixel_to_point(img_mask)
            print(len(result_arr))
            if(len(result_arr) <= 10):
                print("less points 0")
                continue
            ############### DEPROJECTED POINTS ########################
            # Initializing object to class pointCloud() for visualization purposes:
            cloud_object_deprojected_points = pointCloud()

            '''Rotation matrix and position vector for the robot base or world reference frame: '''
            cloud_object_deprojected_points.R_base = np.identity(3)
            cloud_object_deprojected_points.p_base = np.zeros([3,1])

            cloud_object_deprojected_points.g_base_cam = extrinsinc_pos1

            # Extracting the rotation matrix and position vector: 
            R_pose_1 = extrinsinc_pos1[0:3, 0:3]
            p_pose_1 = np.reshape(extrinsinc_pos1[0:3, 3], [3,1])

            cloud_object_deprojected_points.R_base_cam = R_pose_1
            cloud_object_deprojected_points.p_base_cam = p_pose_1

            num_points = len(result_arr)
            result_arr = np.reshape(np.asarray(result_arr), [num_points, 3])

            '''Creating a Open3d PointCloud Object for the cloud corresponding to just the bounding box'''
            objectCloud = o3d.geometry.PointCloud()
            objectCloud.points = o3d.utility.Vector3dVector(result_arr.astype(np.float64))
            objectCloud.paint_uniform_color([0, 0, 1])

            '''Visualizing just the CheezIt point cloud using open3D:'''
            #o3d.visualization.draw_geometries([objectCloud])

            cloud_object_deprojected_points.cloud = objectCloud

            '''Transforming the point cloud in the Panda base reference frame: '''
            cloud_object_deprojected_points.transformToBase()

            '''Visualizing the downsampled point cloud. '''
            print('Cloud transformed to base')
            #o3d.visualization.draw_geometries([cloud_object_deprojected_points.cloud])

            '''# Downsample it and inspect the normals'''
            cloud_object_deprojected_points.cloud = cloud_object_deprojected_points.cloud.voxel_down_sample(voxel_size=0.009)
            #cloud_object_deprojected_points.cloud = cloud_object_deprojected_points.cloud.uniform_down_sample(every_k_points=100)

            if len(cloud_object_deprojected_points.cloud.points) < 40:
                print("less number of points")
                continue

            '''This needs to commented out when dealing with objects like the spatula and screw driver'''
            cloud_object_deprojected_points.removePlaneSurface()

            '''# Visualizing the downsampled point cloud. '''
            print('Plane surface removed!')
            #o3d.visualization.draw_geometries([cloud_object_deprojected_points.cloud])

            '''Specifying parameters for DBSCAN Clustering:
            Just like the parameters for downsampling even the parameters for DBSCAN Clustering are dependent on the 
            units used computing and extracting the point cloud data.'''
            cloud_object_deprojected_points.eps = 0.02
            cloud_object_deprojected_points.min_points = 10
            cloud_object_deprojected_points.getObjectPointCloud()




            # Get the following files and assume they are related to 2nd image information
            print("New transofrmed Image "+str(trans_cnt))
            new_cloud_object_deprojected_points = cloud_object_deprojected_points
            extrinsics2 = data_path+"pose_"+str(trans_cnt)+"/camera_pose.csv"
            in_params2 = data_path+"/intrinsic_params.csv"
            in_model2 = data_path+"/distortion_model.csv"
            in_coeff2 = data_path+"/intrinsic_coeffs.csv"

            transformed_coords = ""
            cnt = 0


            # Call deproject project and transform code 
            extrinsinc_pos2 = genfromtxt(extrinsics2, delimiter=',')

            ############### TRANSFORMING THE POINTS (UPDATED) ###############

            # Extracting the deprojected points which have been transformed in the base reference frame: 
            new_cloud_object_deprojected_points.points = np.asarray(new_cloud_object_deprojected_points.processed_cloud.points)


            print('Shape of the deprojected points: ', new_cloud_object_deprojected_points.points.shape)

            # Transforming these deprojected points from the base reference frame to the second camera pose:
            # Initializing object to class pointCloud() for visualization purposes:
            cloud_object_transformed_points = pointCloud()

            '''Rotation matrix and position vector for the robot base or world reference frame: '''
            cloud_object_transformed_points.R_base = np.identity(3)
            cloud_object_transformed_points.p_base = np.zeros([3,1])

            cloud_object_transformed_points.g_base_cam = extrinsinc_pos2

            # Extracting the rotation matrix and position vector: 
            R_pose_2 = extrinsinc_pos2[0:3, 0:3]
            R_pose_2_inv = la.inv(R_pose_2)
            p_pose_2 = np.reshape(extrinsinc_pos2[0:3, 3], [3,1])

            cloud_object_transformed_points.R_base_cam = R_pose_2
            cloud_object_transformed_points.p_base_cam = p_pose_2

            # transformed_points_updated = []
            transformed_points_updated = np.zeros([new_cloud_object_deprojected_points.points.shape[0], new_cloud_object_deprojected_points.points.shape[1]])

            # Implementation with homogeneous coordinates: 
            point_h = np.ones([4,1])
            for i in range(new_cloud_object_deprojected_points.points.shape[0]):
                point_h[0,:] = new_cloud_object_deprojected_points.points[i, 0]
                point_h[1,:] = new_cloud_object_deprojected_points.points[i, 1]
                point_h[2,:] = new_cloud_object_deprojected_points.points[i, 2]
                extrinsinc_pos2_inv = la.inv(extrinsinc_pos2)
                result = np.matmul(extrinsinc_pos2_inv, point_h)
                transformed_points_updated[i,:] = np.reshape(result[0:3, :], [1,3])
                transformed_pixel = rsObj.project_point_to_pixel(result[0:3,:],in_params2,in_model2,in_coeff2)

                if not math.isnan(transformed_pixel[0])  and not math.isnan(transformed_pixel[1]):
                    transformed_coords += str(round(transformed_pixel[0])) + " " + str(round(transformed_pixel[1]))+ "\n"
                cnt+=1

            f2 = open("transformed_points.txt","w")
            f2.write(transformed_coords)
            f2.close()

            # Convert transformed_coords string to a list of tuples
            coords = [tuple(map(int, line.split())) for line in transformed_coords.strip().split('\n') if line]
            
            # Get the centroid coordinates for the downsampled and then transformed mask of previous image
            xcent, ycent = find_centroid_from_coordinates(coords)
            input_point = np.array([[xcent, ycent]])

            image = image2.copy()
            f = open("transformed_points.txt","r")

            # Get minX, minY, maxX, maxY to use the box method from SAM to which a box is sent as input in predict method
            minx = 2000
            miny= 2000
            maxx = -1
            maxy = -1
            for line in f:
                line = line.strip("\n")
                x, y = line.split(" ")
                cv2.circle(image, (int(x), int(y)), 3, (255, 0, 0), 3) 

                minx = min(int(x),minx)
                miny= min(int(y),miny)
                maxx = max(int(x),maxx)
                maxy = max(int(y),maxy)

            # To see the image with transformed points which are downsampled from the mask of the previous image sam output
            #cv2.imwrite(data_path+"Sampled from "+str(trans_cnt-1)+" img and Transformed to "+str(trans_cnt)+" img"+".png",image)
            image = image2.copy()
            image_path = data_path+"pose_"+str(trans_cnt)+"/item_image.png"
            pil_img = Image.open(image_path)

            predictor.set_image(np.array(pil_img))

            box = np.array([minx,miny,maxx,maxy])

            input_label = np.array([1])

            # Give centroid points and also box as input
            masks, scores, _ = predictor.predict(   
                    point_coords=input_point,
                    point_labels=input_label,
                    box = box[None, :],
                    multimask_output=True,
                    )   
            transformed_masks.append(masks[np.argmax(scores)])
            # img_with_mask = show_mask(masks[np.argmax(scores)], image, False, 0.6)

            
            
            # masks=masks[np.argmax(scores)]

            boxes.append(box)
            centroids.append([xcent, ycent])
            # Draw box
            # x0, y0, x1, y1 = box
            # cv2.rectangle(img_with_mask, (int(x0), int(y0)), (int(x1), int(y1)), (0, 255, 0), 2)
            # cv2.circle(img_with_mask, (int(xcent), int(ycent)), 3, (255, 255, 255), -1)
        previous_masks = transformed_masks
        masks = transformed_masks[0]

        for i in range(1,len(transformed_masks)):
            masks += transformed_masks[i]
        img_with_mask = show_mask(masks, image2.copy(), False, 0.6)

            # Draw box
            # x0, y0, x1, y1 = box
        for box in boxes:
            x0, y0, x1, y1 = box
            cv2.rectangle(img_with_mask, (int(x0), int(y0)), (int(x1), int(y1)), (0, 255, 0), 2)
        for centroid in centroids:
            xcent, ycent = centroid
            cv2.circle(img_with_mask, (int(xcent), int(ycent)), 3, (255, 255, 255), -1)

        # for elem in range(1,len(undo_mask_stack)):
        #     if undo_mask_stack[elem].coord_available: 
        #         cv2.circle(img_with_mask, (undo_mask_stack[elem].x, undo_mask_stack[elem].y), 3, (255, 0, 0), 3)

        # Save the new image which has predicted output of sam along with the bounding box of previous mask and centroid point of previous mask, which are transformed to new image
        cv2.imwrite(data_path+" SAM output of multiple masks img "+str(trans_cnt)+".png",img_with_mask)

    predictor.set_image(np.array(ui_pil_img))

    print("Done")

# Save the recent mask to a file and transform the mask
def save_recent_mask_to_file_and_transform(undo_mask_stack,data_path,predictor, ui_pil_img):

    file_name = filedialog.asksaveasfilename(defaultextension=".csv",
                                                filetypes=[("csv file", ".csv")],
                                                )
    masks = undo_mask_stack[-1].mask

    # Iterate from 2nd folder to 8th folder from Spring_24_Data
    for trans_cnt in range(2,8):

        # Get the following files and assume they are related to 1st image information
        extrinsics1 = data_path+"pose_"+str(trans_cnt-1)+"/"+"camera_pose.csv"
        in_params = data_path+"/"+"intrinsic_params.csv"
        in_model = data_path+"/"+"distortion_model.csv"
        in_coeff = data_path+"/"+"intrinsic_coeffs.csv"
        depImg = data_path+"pose_"+str(trans_cnt-1)+"/"+"depth_image_pixel_transform.png"
        depArr = data_path+"pose_"+str(trans_cnt-1)+"/"+"depth_array.csv"
        image_path = cv2.imread(data_path+"pose_"+str(trans_cnt-1)+"/item_image.png")


        print("save most recent mask to file")

        rsObj = RealsenseSubscriber(in_params,in_model,in_coeff,depArr,depImg)

        savetxt(file_name, masks, delimiter=',')

        # Call deproject project and transform code 
        extrinsinc_pos1 = genfromtxt(extrinsics1, delimiter=',')

        img_mask =  genfromtxt(file_name,delimiter=",")
        result_arr = rsObj.deproject_pixel_to_point(img_mask)


        ############### DEPROJECTED POINTS ########################
        # Initializing object to class pointCloud() for visualization purposes:
        cloud_object_deprojected_points = pointCloud()

        '''Rotation matrix and position vector for the robot base or world reference frame: '''
        cloud_object_deprojected_points.R_base = np.identity(3)
        cloud_object_deprojected_points.p_base = np.zeros([3,1])

        cloud_object_deprojected_points.g_base_cam = extrinsinc_pos1

        # Extracting the rotation matrix and position vector: 
        R_pose_1 = extrinsinc_pos1[0:3, 0:3]
        p_pose_1 = np.reshape(extrinsinc_pos1[0:3, 3], [3,1])

        cloud_object_deprojected_points.R_base_cam = R_pose_1
        cloud_object_deprojected_points.p_base_cam = p_pose_1

        num_points = len(result_arr)
        result_arr = np.reshape(np.asarray(result_arr), [num_points, 3])

        '''Creating a Open3d PointCloud Object for the cloud corresponding to just the bounding box'''
        objectCloud = o3d.geometry.PointCloud()
        objectCloud.points = o3d.utility.Vector3dVector(result_arr.astype(np.float64))
        objectCloud.paint_uniform_color([0, 0, 1])

        '''Visualizing just the CheezIt point cloud using open3D:'''
        #o3d.visualization.draw_geometries([objectCloud])

        cloud_object_deprojected_points.cloud = objectCloud

        '''Transforming the point cloud in the Panda base reference frame: '''
        cloud_object_deprojected_points.transformToBase()

        '''Visualizing the downsampled point cloud. '''
        print('Cloud transformed to base')
        #o3d.visualization.draw_geometries([cloud_object_deprojected_points.cloud])

        '''# Downsample it and inspect the normals'''
        cloud_object_deprojected_points.cloud = cloud_object_deprojected_points.cloud.voxel_down_sample(voxel_size=0.009)
        #cloud_object_deprojected_points.cloud = cloud_object_deprojected_points.cloud.uniform_down_sample(every_k_points=100)


        '''This needs to commented out when dealing with objects like the spatula and screw driver'''
        cloud_object_deprojected_points.removePlaneSurface()

        '''# Visualizing the downsampled point cloud. '''
        print('Plane surface removed!')
        #o3d.visualization.draw_geometries([cloud_object_deprojected_points.cloud])

        '''Specifying parameters for DBSCAN Clustering:
        Just like the parameters for downsampling even the parameters for DBSCAN Clustering are dependent on the 
        units used computing and extracting the point cloud data.'''
        cloud_object_deprojected_points.eps = 0.02
        cloud_object_deprojected_points.min_points = 10
        cloud_object_deprojected_points.getObjectPointCloud()




        # Get the following files and assume they are related to 2nd image information
        print("New transofrmed Image "+str(trans_cnt))
        new_cloud_object_deprojected_points = cloud_object_deprojected_points
        extrinsics2 = data_path+"pose_"+str(trans_cnt)+"/camera_pose.csv"
        in_params2 = data_path+"/intrinsic_params.csv"
        in_model2 = data_path+"/distortion_model.csv"
        in_coeff2 = data_path+"/intrinsic_coeffs.csv"
        image2 = cv2.imread(data_path+"pose_"+str(trans_cnt)+"/item_image.png")

        transformed_coords = ""
        cnt = 0


        # Call deproject project and transform code 
        extrinsinc_pos2 = genfromtxt(extrinsics2, delimiter=',')

        ############### TRANSFORMING THE POINTS (UPDATED) ###############

        # Extracting the deprojected points which have been transformed in the base reference frame: 
        new_cloud_object_deprojected_points.points = np.asarray(new_cloud_object_deprojected_points.processed_cloud.points)


        print('Shape of the deprojected points: ', new_cloud_object_deprojected_points.points.shape)

        # Transforming these deprojected points from the base reference frame to the second camera pose:
        # Initializing object to class pointCloud() for visualization purposes:
        cloud_object_transformed_points = pointCloud()

        '''Rotation matrix and position vector for the robot base or world reference frame: '''
        cloud_object_transformed_points.R_base = np.identity(3)
        cloud_object_transformed_points.p_base = np.zeros([3,1])

        cloud_object_transformed_points.g_base_cam = extrinsinc_pos2

        # Extracting the rotation matrix and position vector: 
        R_pose_2 = extrinsinc_pos2[0:3, 0:3]
        R_pose_2_inv = la.inv(R_pose_2)
        p_pose_2 = np.reshape(extrinsinc_pos2[0:3, 3], [3,1])

        cloud_object_transformed_points.R_base_cam = R_pose_2
        cloud_object_transformed_points.p_base_cam = p_pose_2

        # transformed_points_updated = []
        transformed_points_updated = np.zeros([new_cloud_object_deprojected_points.points.shape[0], new_cloud_object_deprojected_points.points.shape[1]])

        # Implementation with homogeneous coordinates: 
        point_h = np.ones([4,1])
        for i in range(new_cloud_object_deprojected_points.points.shape[0]):
            point_h[0,:] = new_cloud_object_deprojected_points.points[i, 0]
            point_h[1,:] = new_cloud_object_deprojected_points.points[i, 1]
            point_h[2,:] = new_cloud_object_deprojected_points.points[i, 2]
            extrinsinc_pos2_inv = la.inv(extrinsinc_pos2)
            result = np.matmul(extrinsinc_pos2_inv, point_h)
            transformed_points_updated[i,:] = np.reshape(result[0:3, :], [1,3])
            transformed_pixel = rsObj.project_point_to_pixel(result[0:3,:],in_params2,in_model2,in_coeff2)

            if not math.isnan(transformed_pixel[0])  and not math.isnan(transformed_pixel[1]):
                transformed_coords += str(round(transformed_pixel[0])) + " " + str(round(transformed_pixel[1]))+ "\n"
            cnt+=1

        f2 = open("transformed_points.txt","w")
        f2.write(transformed_coords)
        f2.close()

        # Convert transformed_coords string to a list of tuples
        coords = [tuple(map(int, line.split())) for line in transformed_coords.strip().split('\n') if line]
        
        # Get the centroid coordinates for the downsampled and then transformed mask of previous image
        xcent, ycent = find_centroid_from_coordinates(coords)
        input_point = np.array([[xcent, ycent]])

        image = image2.copy()
        f = open("transformed_points.txt","r")

        # Get minX, minY, maxX, maxY to use the box method from SAM to which a box is sent as input in predict method
        minx = 2000
        miny= 2000
        maxx = -1
        maxy = -1
        for line in f:
            line = line.strip("\n")
            x, y = line.split(" ")
            cv2.circle(image, (int(x), int(y)), 3, (255, 0, 0), 3) 

            minx = min(int(x),minx)
            miny= min(int(y),miny)
            maxx = max(int(x),maxx)
            maxy = max(int(y),maxy)

        # To see the image with transformed points which are downsampled from the mask of the previous image sam output
        #cv2.imwrite(data_path+"Sampled from "+str(trans_cnt-1)+" img and Transformed to "+str(trans_cnt)+" img"+".png",image)

        image = image2.copy()
        image_path = data_path+"pose_"+str(trans_cnt)+"/item_image.png"
        pil_img = Image.open(image_path)

        predictor.set_image(np.array(pil_img))

        box = np.array([minx,miny,maxx,maxy])

        input_label = np.array([1])

        # Give centroid points and also box as input
        masks, scores, _ = predictor.predict(   
                point_coords=input_point,
                point_labels=input_label,
                box = box[None, :],
                multimask_output=True,
                )   
        
        img_with_mask = show_mask(masks[np.argmax(scores)], image, False, 0.6)
        
        
        masks=masks[np.argmax(scores)]

        # Draw box
        x0, y0, x1, y1 = box
        cv2.rectangle(img_with_mask, (int(x0), int(y0)), (int(x1), int(y1)), (0, 255, 0), 2)
        cv2.circle(img_with_mask, (int(xcent), int(ycent)), 3, (255, 255, 255), -1)

        # Save the new image which has predicted output of sam along with the bounding box of previous mask and centroid point of previous mask, which are transformed to new image
        cv2.imwrite(data_path+" SAM output img "+str(trans_cnt)+".png",img_with_mask)

    predictor.set_image(np.array(ui_pil_img))

    print("Done")
    
# subtract_event does subtraction of top 2 masks in undo_mask_stack
def subtract_event(undo_mask_stack, img, predictor,label):
    print("subtract")

    if len(undo_mask_stack) < 3:
        return
    
    masks1 = undo_mask_stack[-1].mask
    masks2 = undo_mask_stack[-2].mask

    sum_mask_1 = np.sum(masks1)
    sum_mask_2 = np.sum(masks2)

    if(sum_mask_1>sum_mask_2):
         subtracted_mask = np.absolute(masks1 - masks2)
    else:
         subtracted_mask = np.absolute(masks2 - masks1)
   
    temp = MaskGenerator(predictor)
    temp.setMask(subtracted_mask)
    temp.printMask(img,label)

    undo_mask_stack.pop()
    undo_mask_stack.pop()
    undo_mask_stack.append(temp)

# union_event does union of all the masks in undo_mask_stack
def union_event(undo_mask_stack, img,label):
    print("print all masks")

    masks = undo_mask_stack[1].mask

    for i in range(2,len(undo_mask_stack)):
        masks += undo_mask_stack[i].mask
    img_with_mask = show_mask(masks, img.copy(), False, 0.6)

    for elem in range(1,len(undo_mask_stack)):
        if undo_mask_stack[elem].coord_available: 
            cv2.circle(img_with_mask, (undo_mask_stack[elem].x, undo_mask_stack[elem].y), 3, (255, 0, 0), 3)
    imgtk = cv2_to_imageTK(img_with_mask)
    label.imgtk = imgtk
    label.configure(image = imgtk)

# Save the x and y pixel coordinates in a text file
def save_pts(undo_mask_stack,img,label):

    points = ""
    for i in range(1,len(undo_mask_stack)-1):
        points += str(undo_mask_stack[i].x) + " " + str(undo_mask_stack[i].y) + "\n"
    points += str(undo_mask_stack[-1].x) + " " + str(undo_mask_stack[-1].y)

    file_name = filedialog.asksaveasfilename(defaultextension=".txt",
                                            filetypes=[("text file", ".txt")],
    )
    file = open(file_name,"w")
    file.write(points)
    file.close

# Visualize masks from csv and points from text file
def visualize_masks(img,label,predictor,undo_mask_stack):
    file_name = filedialog.askopenfilename()
    print(file_name)
    extension = Path(file_name).suffix
    print(extension)

    if extension == ".csv":
        mask = genfromtxt(file_name, delimiter=',')
        img_with_mask = show_mask(mask, img.copy(), False, 0.6)
        imgtk = cv2_to_imageTK(img_with_mask)
        label.imgtk = imgtk
        label.configure(image = imgtk)
    else:

        image = img.copy()
        f = open(file_name, "r")
        for line in f:
            line = line.strip("\n")
            x, y = line.split(" ")
            cv2.circle(image, (int(x), int(y)), 3, (255, 0, 0), 3) 
            # temp =  MaskGenerator(predictor)
            # temp.setCoordinates(int(x),int(y))
            # undo_mask_stack.append(temp)

        imgtk = cv2_to_imageTK(image)
        label.imgtk = imgtk
        label.configure(image = imgtk)


# redo_event redo the previous event and change redo_mask_stack, undo_mask_stack respectively
def redo_event(undo_mask_stack, redo_mask_stack, img, label):
    print("redo")
    # check if the redo_mask_stack is empty, 2 due to 'zeroMask.setCoordinates(0,0)' in main
    if len(redo_mask_stack) < 2:
        return
    
    # remove state from redo stack and add to undo stack
    undo_mask_stack.append(redo_mask_stack.pop())

    if len(undo_mask_stack) < 2:
        imgtk = cv2_to_imageTK(img)
        label.imgtk = imgtk
        label.configure(image = imgtk)
        return

    # based on the topmost value of the undo stack display the mask
    undo_mask_stack[-1].printMask(img,label)
    

# undo_event undo the previous event and change redo_mask_stack, undo_mask_stack respectively
def undo_event(undo_mask_stack,redo_mask_stack,img,label):
    print("undo")

    # check if the undo_mask_stack is empty, 2 due to 'zeroMask.setCoordinates(0,0)' in main
    if len(undo_mask_stack) < 2:
        return
    
    # remove state from undo stack and add to redo stack
    redo_mask_stack.append(undo_mask_stack.pop())

    if len(undo_mask_stack) < 2:
        imgtk = cv2_to_imageTK(img)
        label.imgtk = imgtk
        label.configure(image = imgtk)
        return

    #based on the topmost value of the undo stack display the mask
    undo_mask_stack[-1].printMask(img,label)
   
# click_event on click get the points and get the mask for that point on image
def click_event(eventorigin, undo_mask_stack, predictor, img, label):
      global x,y

      x = eventorigin.x
      y = eventorigin.y
      print(x,y)

      temp =  MaskGenerator(predictor)
      temp.setCoordinates(x,y)

      # Add new mask to undo_mask_stack
      undo_mask_stack.append(temp)
      temp.printMask(img,label)
     

def main(args):

    # Hard coded values as per the given dataset in ./Spring_24_Data/
    data_path="fall_24_Data/"

    ui_extrinsics1 = data_path+"pose_1/"+"camera_pose.csv"
    ui_in_params = data_path+"intrinsic_params.csv"
    ui_in_model = data_path+"distortion_model.csv"
    ui_in_coeff = data_path+"intrinsic_coeffs.csv"
    ui_depImg = data_path+"pose_1/"+"depth_image_pixel_transform.png"
    ui_depArr = data_path+"pose_1/"+"depth_array.csv"
    ui_image_path = data_path+"pose_1/"+"item_image.png"

    ui_rsObj = RealsenseSubscriber(ui_in_params,ui_in_model,ui_in_coeff,ui_depArr,ui_depImg)

    device = "cpu"

    #1. Prepare Image For Inference 
    print("Preparing Model + Images")
    ui_pil_img = Image.open(ui_image_path) 
    
    #2. Prepare Model For Inference
    ui_sam_checkpoint = "sam_vit_h_4b8939.pth"
    ui_model_type = "vit_h"
    ui_sam = sam_model_registry[ui_model_type](checkpoint=ui_sam_checkpoint)
    ui_sam.to(device=device)
    ui_predictor = SamPredictor(ui_sam)
    ui_predictor.set_image(np.array(ui_pil_img))
    print("Model + Image Ready")

    #3. Inference
    zeroMask  = MaskGenerator(ui_predictor)
    zeroMask.setCoordinates(0,0)
    undo_mask_stack = [zeroMask]
    redo_mask_stack = [zeroMask] 

    # Main application Window
    window = tk.Tk()

    window.title("Image Segmentation Interface")

    # Load the image using the PhotoImage class
    ui_temp_image = PhotoImage(file=ui_image_path)

    # Create a label to display the image
    ui_label = tk.Label(window, image=ui_temp_image)
    ui_img = cv2.imread(ui_image_path)

    ui_label.bind('<Button-1>', lambda e: click_event(e, undo_mask_stack, ui_predictor,ui_img,ui_label))
    ui_label.grid(row=1,column=0,columnspan=10,ipadx=0,padx=0)


    ########  tkinter code buttons  ######
    undo_button_image = ImageTk.PhotoImage(Image.open('button_imgs/icons8-undo-50.png'))
    # Undo button
    undo_button = tk.Button(
        window,
        text='Undo',
        font= ('Helvetica 15 bold'),
        image=undo_button_image,
        compound= "left",
        command=lambda: undo_event(undo_mask_stack,redo_mask_stack,ui_img,ui_label)
    ).grid(row=0,column=0,sticky='nesw')
    
    redo_button_image = ImageTk.PhotoImage(Image.open('button_imgs/icons8-redo-50.png'))
    # Redo button
    redo_button = tk.Button(
        window,
        text='Redo',
        font= ('Helvetica 15 bold'),
        image=redo_button_image,
        compound= "left",
        command=lambda: redo_event(undo_mask_stack,redo_mask_stack,ui_img,ui_label) 
    ).grid(row=0,column=1,sticky='nesw')
    
    clear_button_image = ImageTk.PhotoImage(Image.open('button_imgs/icons8-trash-50.png'))
    # Clear button
    clear_button = tk.Button(
        window,
        text='Clear',
        font= ('Helvetica 15 bold'),
        image=clear_button_image,
        compound= "left",
        command=lambda: clear_stacks(undo_mask_stack,redo_mask_stack,ui_predictor,ui_img,ui_label)
    ).grid(row=0,column=2,sticky='nesw')
    
    subtract_button_image = ImageTk.PhotoImage(Image.open('button_imgs/icons8-query-inner-join-50.png'))
    # Subtract button
    subtract_button = tk.Button(
        window,
        text='Subtract',
        font= ('Helvetica 15 bold'),
        image=subtract_button_image,
        compound= "left",
        command=lambda: subtract_event(undo_mask_stack,ui_img, ui_predictor,ui_label)
    ).grid(row=0,column=3,sticky='nesw')
    
    union_button_image = ImageTk.PhotoImage(Image.open('button_imgs/icons8-query-outer-join-50.png'))
    #Union button
    union_button = tk.Button(
        window,
        text='Union',
        font= ('Helvetica 15 bold'),
        image=union_button_image,
        compound= "left",
        command=lambda: union_event(undo_mask_stack,ui_img,ui_label)
    ).grid(row=0,column=4,sticky='nesw')

    save_button_image = ImageTk.PhotoImage(Image.open('button_imgs/icons8-save-all-50.png'))
    #Save to file button
    save_all_button = tk.Button(
        window,
        text='Save All & Transform',
        font= ('Helvetica 15 bold'),
        image=save_button_image,
        compound= "left",
        command=lambda:  save_all_masks_to_file_and_transform(undo_mask_stack,data_path,ui_predictor, ui_pil_img)
    ).grid(row=0,column=5,sticky='nesw')
    
    save_recent_button_image = ImageTk.PhotoImage(Image.open('button_imgs/icons8-save-50.png'))
    #Save recent to button
    save_button = tk.Button(
        window,
        text='Save & Transform',
        font= ('Helvetica 15 bold'),
        image=save_recent_button_image,
        compound= "left",
        command=lambda:  save_recent_mask_to_file_and_transform(undo_mask_stack,data_path,ui_predictor, ui_pil_img)
    ).grid(row=0,column=6,sticky='nesw')
    
    exit_button_image = ImageTk.PhotoImage(Image.open('button_imgs/icons8-close-window-50.png'))
    # Exit button
    exit_button = tk.Button(
        window,
        text='Exit',
        font= ('Helvetica 15 bold'),
        image=exit_button_image,
        compound= "left",
        command=lambda: window.quit()
    ).grid(row=0,column=7,sticky='nesw')
    
    save_pts_button  = ImageTk.PhotoImage(Image.open('button_imgs/icons8-save-as-50.png'))
    # Save points button
    savepts_button = tk.Button(
        window,
        text='Save points',
        font= ('Helvetica 15 bold'),
        image = save_pts_button,
        compound= "left",
        command=lambda: save_pts(undo_mask_stack,ui_img,ui_label)
    ).grid(row=0,column=8,sticky='nesw')
    
    visualize_button_image = ImageTk.PhotoImage(Image.open('button_imgs/icons8-save-to-grid-50.png'))
    # Visualize button
    visualize_button = tk.Button(
        window,
        text='Visualize',
        font= ('Helvetica 15 bold'),
        image = visualize_button_image,
        compound= "left",
        command=lambda: visualize_masks(ui_img,ui_label,ui_predictor,undo_mask_stack)
    ).grid(row=0,column=9,sticky='nesw')
    
    # Run the tkinter event loop
    window.mainloop()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    ########### Currently using hardcoded values and not the arguments  ##########
    """    
    parser.add_argument("-g", "--gpu", type=str, default='0', help="GPU to use")
    parser.add_argument("--img", type=str, default="demo_imgs/image_1_pixel_transform.png", help="image")
    parser.add_argument("--i2", type=str, default="demo_imgs/image_2_pixel_transform.png", help="image")
    parser.add_argument("--e1", type=str, default="Data/trial_1/screwdriver/camera_pose.csv", help="camera pose 1")
    parser.add_argument("--e2", type=str, default="Data/trial_2/screwdriver/camera_pose.csv", help="camera pose 2")
    parser.add_argument("--inparams", type=str, default="Data/trial_1/screwdriver/intrinsic_params_1.csv", help="intrinsic params for pose 1")
    parser.add_argument("--incoeff", type=str, default="Data/trial_1/screwdriver/intrinsic_coeffs_1.csv", help="intrinsic coeff for pose 1")
    parser.add_argument("--inmodel", type=str, default="Data/trial_1/screwdriver/distortion_model_1.csv", help="intrinsic model for pose 1")
    parser.add_argument("--inparams2", type=str, default="Data/trial_2/screwdriver/intrinsic_params_2.csv", help="intrinsic params for pose 2")
    parser.add_argument("--incoeff2", type=str, default="Data/trial_2/screwdriver/intrinsic_coeffs_2.csv", help="intrinsic coeff for pose 2")
    parser.add_argument("--inmodel2", type=str, default="Data/trial_2/screwdriver/distortion_model_2.csv", help="intrinsic model for pose 2")
    parser.add_argument("--dimg", type=str, default="Data/trial_1/screwdriver/depth_image_1_pixel_transform.png", help="depth image")
    parser.add_argument("--darr", type=str, default="Data/trial_1/screwdriver/depth_array_1.csv", help="depth array")
    """

    args = parser.parse_args()
    main(args)
