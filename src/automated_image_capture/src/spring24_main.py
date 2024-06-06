#!/usr/bin/env python3
# Python script to generate poses in SE(3) 
# By: Aditya Patankar

# Open3D for point cloud processing and visualization
import numpy as np
from numpy import linalg as la
#import open3d as o3d


import rospy
import rosbag
from std_msgs.msg import Int32, String
import rospy
import numpy as np

import cv2

from PIL import Image
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError



# Matplotlib libraries for plotting and visualization in Python:
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from matplotlib import cm
from geometry_msgs.msg import Pose
from scipy.spatial.transform import Rotation as Rot

from sensor_msgs.msg import JointState, CameraInfo

import pyrealsense2 as rs
import numpy as np
import cv2


bridge = CvBridge()

pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)



    
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
 
   
def captureImagesAlongTrajectory():

    bag_name = f"bag.bag"  # Construct the bag name
    print(bag_name)

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
        
    return 1
      

if __name__ == '__main__':
    captureImagesAlongTrajectory()
    
    
