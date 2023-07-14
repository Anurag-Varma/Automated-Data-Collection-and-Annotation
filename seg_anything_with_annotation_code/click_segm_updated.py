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

def cv2_to_imageTK(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGBA)
    imagePIL = Image.fromarray(image)
    imgtk = ImageTk.PhotoImage(image= imagePIL)
    return imgtk
        

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


def clear_stacks(undo_stack, redo_stack,predictor,img,label):
    print("clear stacks")
    undo_stack.clear()
    redo_stack.clear()
    zeroMask  = MaskGenerator(predictor)
    zeroMask.setCoordinates(0,0)
    undo_stack.append(zeroMask)
    redo_stack.append(zeroMask)
    imgtk = cv2_to_imageTK(img)
    label.imgtk = imgtk
    label.configure(image = imgtk)
   
 

def print_all_masks(undo_stack, img,label):
    print("print all masks")

    masks = undo_stack[1].mask

    for i in range(2,len(undo_stack)):
        masks += undo_stack[i].mask
    img_with_mask = show_mask(masks, img.copy(), False, 0.6)

    for elem in range(1,len(undo_stack)):
        if undo_stack[elem].coord_available: 
            cv2.circle(img_with_mask, (undo_stack[elem].x, undo_stack[elem].y), 3, (255, 0, 0), 3)
    imgtk = cv2_to_imageTK(img_with_mask)
    label.imgtk = imgtk
    label.configure(image = imgtk)
   
# Save the union of all the mask to a file and also display transformed masks output
def save_to_file(undo_stack,extrinsics1,extrinsics2,image2,label,rsObj,in_params2,in_model2,in_coeff2):
    print("save to file")

    masks = undo_stack[1].mask

    for i in range(2,len(undo_stack)):
        masks += undo_stack[i].mask

    file_name = filedialog.asksaveasfilename(defaultextension=".csv",
                                            filetypes=[("csv file", ".csv")],
                                            )
    savetxt(file_name, masks, delimiter=',')
   
    # Call deproject project and transform code 
    extrinsinc_pos1 = genfromtxt(extrinsics1, delimiter=',')
    extrinsinc_pos2 = genfromtxt(extrinsics2, delimiter=',')

    img_mask =  genfromtxt(file_name,delimiter=",")
    result_arr = rsObj.deproject_pixel_to_point(img_mask)
    transformed_coords = ""
    cnt = 0

    for pixel_coord in result_arr:
        point = rsObj.transform_point(extrinsinc_pos1,extrinsinc_pos2,pixel_coord)
        transformed_pixel = rsObj.project_point_to_pixel(point,in_params2,in_model2,in_coeff2)
    
        if not math.isnan(transformed_pixel[0])  and not math.isnan(transformed_pixel[1]):
            transformed_coords += str(round(transformed_pixel[0])) + " " + str(round(transformed_pixel[1]))+ "\n"
        cnt+=1
        
    f2 = open("transformed_points.txt","w")
    f2.write(transformed_coords)
    f2.close()

    image = image2.copy()
    f = open("transformed_points.txt","r")
    for line in f:
        line = line.strip("\n")
        x, y = line.split(" ")
        # print(x,y)
        cv2.circle(image, (int(x), int(y)), 3, (255, 0, 0), 3) 
    cv2.imshow('Transformed Points', image)
    f.close()
    cv2.waitKey(0)
    cv2.destroyAllWindows()

  

# Save the recent mask to a file and also display transformed masks output
def save_recent_mask_to_file(undo_stack,extrinsics1,extrinsics2,image2,label,rsObj,in_params2,in_model2,in_coeff2):
    print("save most recent mask to file")

    masks = undo_stack[-1].mask
    file_name = filedialog.asksaveasfilename(defaultextension=".csv",
                                            filetypes=[("csv file", ".csv")],
                                            )
    savetxt(file_name, masks, delimiter=',')

    # Call deproject project and transform code 
    extrinsinc_pos1 = genfromtxt(extrinsics1, delimiter=',')
    extrinsinc_pos2 = genfromtxt(extrinsics2, delimiter=',')

    img_mask =  genfromtxt(file_name,delimiter=",")
    result_arr = rsObj.deproject_pixel_to_point(img_mask)
    transformed_coords = ""
    cnt = 0

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
    # o3d.visualization.draw_geometries([objectCloud])

    cloud_object_deprojected_points.cloud = objectCloud

    '''Transforming the point cloud in the Panda base reference frame: '''
    cloud_object_deprojected_points.transformToBase()

    '''Visualizing the downsampled point cloud. '''
    print('Cloud transformed to base')
    # o3d.visualization.draw_geometries([cloud_object.cloud])

    '''# Downsample it and inspect the normals'''
    # cloud_object_deprojected_points.cloud = cloud_object_deprojected_points.cloud.voxel_down_sample(voxel_size=0.009)
    
    '''This needs to commented out when dealing with objects like the spatula and screw driver'''
    # cloud_object.removePlaneSurface()

    '''# Visualizing the downsampled point cloud. '''
    print('Plane surface removed!')
    # o3d.visualization.draw_geometries([cloud_object.cloud])

    '''Specifying parameters for DBSCAN Clustering:
    Just like the parameters for downsampling even the parameters for DBSCAN Clustering are dependent on the 
    units used computing and extracting the point cloud data.'''
    cloud_object_deprojected_points.eps = 0.02
    cloud_object_deprojected_points.min_points = 10
    cloud_object_deprojected_points.getObjectPointCloud()


    ############### TRANSFORMING THE POINTS (UPDATED) ###############

    # Extracting the deprojected points which have been transformed in the base reference frame: 
    cloud_object_deprojected_points.points = np.asarray(cloud_object_deprojected_points.processed_cloud.points)

    print('Shape of the deprojected points: ', cloud_object_deprojected_points.points.shape)

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
    transformed_points_updated = np.zeros([cloud_object_deprojected_points.points.shape[0], cloud_object_deprojected_points.points.shape[1]])
    
    # Nonhomogeneous coordinates implementation: 
    ''' for i in range(cloud_object_deprojected_points.points.shape[0]):
        point = np.reshape(cloud_object_deprojected_points.points[i, :], [3,1])
        prod = np.dot(R_pose_2_inv, point)
        result = np.add(p_pose_2, prod)
        print('result: ', result)
        # Appending the points in a single list:
        # transformed_points_updated.append(result)
        transformed_points_updated[i,:] = np.reshape(result, [1,3])'''

    # Implementation with homogeneous coordinates: 
    point_h = np.ones([4,1])
    for i in range(cloud_object_deprojected_points.points.shape[0]):
        point_h[0,:] = cloud_object_deprojected_points.points[i, 0]
        point_h[1,:] = cloud_object_deprojected_points.points[i, 1]
        point_h[2,:] = cloud_object_deprojected_points.points[i, 2]
        extrinsinc_pos2_inv = la.inv(extrinsinc_pos2)
        result = np.matmul(extrinsinc_pos2_inv, point_h)
        transformed_points_updated[i,:] = np.reshape(result[0:3, :], [1,3])
        transformed_pixel = rsObj.project_point_to_pixel(result[0:3,:],in_params2,in_model2,in_coeff2)

        if not math.isnan(transformed_pixel[0])  and not math.isnan(transformed_pixel[1]):
            transformed_coords += str(round(transformed_pixel[0])) + " " + str(round(transformed_pixel[1]))+ "\n"
        cnt+=1

    # Initializing object to class pointCloud() for visualization purposes:
    cloud_object_transformed_points = pointCloud()

    print('Shape of the transformed points: ', transformed_points_updated.shape)

    '''Rotation matrix and position vector for the robot base or world reference frame: '''
    cloud_object_transformed_points.R_base = np.identity(3)
    cloud_object_transformed_points.p_base = np.zeros([3,1])

    cloud_object_transformed_points.g_base_cam = extrinsinc_pos2

    # Extracting the rotation matrix and position vector: 
    R_pose_2 = extrinsinc_pos2[0:3, 0:3]
    p_pose_2 = np.reshape(extrinsinc_pos2[0:3, 3], [3,1])

    cloud_object_transformed_points.R_base_cam = R_pose_2
    cloud_object_transformed_points.p_base_cam = p_pose_2
    
    '''Creating a Open3d PointCloud Object for the cloud corresponding to just the bounding box'''
    objectCloud = o3d.geometry.PointCloud()
    objectCloud.points = o3d.utility.Vector3dVector(transformed_points_updated.astype(np.float64))
    objectCloud.paint_uniform_color([0, 0, 1])

    '''Visualizing just the CheezIt point cloud using open3D:'''
    o3d.visualization.draw_geometries([objectCloud])

    cloud_object_transformed_points.cloud = objectCloud

    '''Transforming the point cloud in the Panda base reference frame: '''
    # cloud_object_transformed_points.transformToBase()

    '''Visualizing the downsampled point cloud. '''
    print('Cloud transformed to base')
    # o3d.visualization.draw_geometries([cloud_object.cloud])

    '''# Downsample it and inspect the normals'''
    # cloud_object_transformed_points.cloud = cloud_object_transformed_points.cloud.voxel_down_sample(voxel_size=0.009)
    
    '''This needs to commented out when dealing with objects like the spatula and screw driver'''
    # cloud_object.removePlaneSurface()

    '''# Visualizing the downsampled point cloud. '''
    print('Plane surface removed!')
    # o3d.visualization.draw_geometries([cloud_object.cloud])

    '''Specifying parameters for DBSCAN Clustering:
    Just like the parameters for downsampling even the parameters for DBSCAN Clustering are dependent on the 
    units used computing and extracting the point cloud data.'''
    cloud_object_transformed_points.eps = 0.02
    cloud_object_transformed_points.min_points = 10
    cloud_object_transformed_points.getObjectPointCloud()


    ##########################################################
    '''PLOTTING AND VISUALIZATION:'''




    ##########################################################

    f2 = open("transformed_points.txt","w")
    f2.write(transformed_coords)
    f2.close()

    image = image2.copy()
    f = open("transformed_points.txt","r")
    for line in f:
        line = line.strip("\n")
        x, y = line.split(" ")
        # print(x,y)
        cv2.circle(image, (int(x), int(y)), 3, (255, 0, 0), 3) 
    cv2.imshow('Transformed Points', image)
    f.close()
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def subtract_event(undo_stack, img, predictor,label):
    print("subtract")

    if len(undo_stack) < 3:
        return
    
    masks1 = undo_stack[-1].mask
    masks2 = undo_stack[-2].mask

    sum_mask_1 = np.sum(masks1)
    sum_mask_2 = np.sum(masks2)

    if(sum_mask_1>sum_mask_2):
         subtracted_mask = np.absolute(masks1 - masks2)
    else:
         subtracted_mask = np.absolute(masks2 - masks1)
   
    temp = MaskGenerator(predictor)
    temp.setMask(subtracted_mask)
    temp.printMask(img,label)

    undo_stack.pop()
    undo_stack.pop()
    undo_stack.append(temp)

# Save the x and y pixel coordinates in a text file
def save_pts(undo_stack,img,label):

    points = ""
    for i in range(1,len(undo_stack)-1):
        points += str(undo_stack[i].x) + " " + str(undo_stack[i].y) + "\n"
    points += str(undo_stack[-1].x) + " " + str(undo_stack[-1].y)

    file_name = filedialog.asksaveasfilename(defaultextension=".txt",
                                            filetypes=[("text file", ".txt")],
    )
    file = open(file_name,"w")
    file.write(points)
    file.close

# Visualize masks from csv and points from text file
def visualize_masks(img,label,predictor,undo_stack):
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
            # undo_stack.append(temp)

        #print_all_masks(undo_stack,img,label)
        imgtk = cv2_to_imageTK(image)
        label.imgtk = imgtk
        label.configure(image = imgtk)



def redo_event(undo_stack,redo_stack,img,label):
    print("redo")
    # check if the redo_stack is empty
    if len(redo_stack) < 2:
        return
    
    # remove state from redo stack and add to undo stack
    undo_stack.append(redo_stack.pop())

    if len(undo_stack) < 2:
        imgtk = cv2_to_imageTK(img)
        label.imgtk = imgtk
        label.configure(image = imgtk)
        return

    # based on the topmost value of the undo stack display the mask
    undo_stack[-1].printMask(img,label)
    


def undo_event(undo_stack,redo_stack,img,label):
    print("undo")
    # check if the undo_stack is empty

    if len(undo_stack) < 2:
        return
    
    # remove state from undo stack and add to redo stack
    redo_stack.append(undo_stack.pop())

    if len(undo_stack) < 2:
        imgtk = cv2_to_imageTK(img)
        label.imgtk = imgtk
        label.configure(image = imgtk)
        return

    #based on the topmost value of the undo stack display the mask
    undo_stack[-1].printMask(img,label)
   

def click_event(eventorigin, undo_stack, predictor,img,label):
      global x,y
      x = eventorigin.x
      y = eventorigin.y
      print(x,y)

      temp =  MaskGenerator(predictor)
      temp.setCoordinates(x,y)
      undo_stack.append(temp)
      temp.printMask(img,label)
     

def main(args):
    extrinsics1 = args.e1
    extrinsics2 = args.e2
    image2 = cv2.imread(args.i2)
    in_params = args.inparams
    in_model = args.inmodel
    in_coeff = args.incoeff
    in_params2= args.inparams2
    in_model2 = args.inmodel2
    in_coeff2 = args.incoeff2
    depImg = args.dimg
    depArr = args.darr

    rsObj = RealsenseSubscriber(in_params,in_model,in_coeff,depArr,depImg)

    device = "cpu"
    image_path = args.img  
 
    #1. Prepare Image For Inference 
    print("Preparing Model + Images")
    pil_img = Image.open(image_path) 
    
    #2. Prepare Model For Inference
    sam_checkpoint = "sam_vit_h_4b8939.pth"
    model_type = "vit_h"
    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam.to(device=device)
    predictor = SamPredictor(sam)
    predictor.set_image(np.array(pil_img))
    print("Model + Image Ready")

    # #3. Inference
    zeroMask  = MaskGenerator(predictor)
    zeroMask.setCoordinates(0,0)
    undo_stack = [zeroMask]
    redo_stack = [zeroMask] 

    # Main application Window
    window = tk.Tk()

    window.title("Image Segmentation Interface")

     # Load the image using the PhotoImage class
    temp_image = PhotoImage(file=image_path)

    # button_frame = tk.Frame(window)
    # button_frame.grid(row=0, column=0)

    undo_button_image = ImageTk.PhotoImage(Image.open('button_imgs/icons8-undo-50.png'))

    # Undo button
    undo_button = tk.Button(
    window,
    text='Undo',
    font= ('Helvetica 15 bold'),
    image=undo_button_image,
    compound= "left",
    command=lambda: undo_event(undo_stack,redo_stack,img,label)
).grid(row=0,column=0,sticky='nesw')
    
    redo_button_image = ImageTk.PhotoImage(Image.open('button_imgs/icons8-redo-50.png'))

    # Redo button
    redo_button = tk.Button(
    window,
    text='Redo',
    font= ('Helvetica 15 bold'),
    image=redo_button_image,
    compound= "left",
    command=lambda: redo_event(undo_stack,redo_stack,img,label)
).grid(row=0,column=1,sticky='nesw')
    
    clear_button_image = ImageTk.PhotoImage(Image.open('button_imgs/icons8-trash-50.png'))

    # Clear button
    clear_button = tk.Button(
    window,
    text='Clear',
    font= ('Helvetica 15 bold'),
    image=clear_button_image,
    compound= "left",
    command=lambda: clear_stacks(undo_stack,redo_stack,predictor,img,label)
).grid(row=0,column=2,sticky='nesw')
    
    subtract_button_image = ImageTk.PhotoImage(Image.open('button_imgs/icons8-query-inner-join-50.png'))
    
    # Subtract button
    subtract_button = tk.Button(
    window,
    text='Subtract',
    font= ('Helvetica 15 bold'),
    image=subtract_button_image,
    compound= "left",
    command=lambda: subtract_event(undo_stack,img, predictor,label)
).grid(row=0,column=3,sticky='nesw')
    
    union_button_image = ImageTk.PhotoImage(Image.open('button_imgs/icons8-query-outer-join-50.png'))

    #Union button
    union_button = tk.Button(
    window,
    text='Union',
    font= ('Helvetica 15 bold'),
    image=union_button_image,
    compound= "left",
    command=lambda: print_all_masks(undo_stack,img,label)
).grid(row=0,column=4,sticky='nesw')

    save_button_image = ImageTk.PhotoImage(Image.open('button_imgs/icons8-save-all-50.png'))
    #Save to file button
    save_all_button = tk.Button(
    window,
    text='Save All & Transform',
    font= ('Helvetica 15 bold'),
    image=save_button_image,
    compound= "left",
    command=lambda:  save_to_file(undo_stack,extrinsics1,extrinsics2,image2,label,rsObj,in_params2,in_model2,in_coeff2)
).grid(row=0,column=5,sticky='nesw')
    
    save_recent_button_image = ImageTk.PhotoImage(Image.open('button_imgs/icons8-save-50.png'))

    #Save recent to button
    save_button = tk.Button(
    window,
    text='Save & Transform',
    font= ('Helvetica 15 bold'),
    image=save_recent_button_image,
    compound= "left",
    command=lambda:  save_recent_mask_to_file(undo_stack,extrinsics1,extrinsics2,image2,label,rsObj,in_params2,in_model2,in_coeff2)
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
    
    savepts_button = tk.Button(
    window,
    text='Save points',
    font= ('Helvetica 15 bold'),
    image = save_pts_button,
    compound= "left",
    command=lambda: save_pts(undo_stack,img,label)
).grid(row=0,column=8,sticky='nesw')
    
    visualize_button_image = ImageTk.PhotoImage(Image.open('button_imgs/icons8-save-to-grid-50.png'))
    
    visualize_button = tk.Button(
    window,
    text='Visualize',
    font= ('Helvetica 15 bold'),
    image = visualize_button_image,
    compound= "left",
    command=lambda: visualize_masks(img,label,predictor,undo_stack)
).grid(row=0,column=9,sticky='nesw')
    
    # Create a label to display the image
    label = tk.Label(window, image=temp_image)
    img = cv2.imread(image_path)
    label.bind('<Button-1>', lambda e: click_event(e, undo_stack, predictor,img,label))
    label.grid(row=1,column=0,columnspan=10,ipadx=0,padx=0)

    # Run the tkinter event loop
    window.mainloop()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
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
    args = parser.parse_args()
    main(args)