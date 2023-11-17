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


#Initialize a bridge object of class CvBridge


bag = rosbag.Bag('test_1.bag', 'w')
bridge = CvBridge()
      
#Callback funtion for the subscriber node
def image_callback(ros_image):
    global bridge
    try:
        cv2_image = bridge.imgmsg_to_cv2(ros_image, "rgb8")
        cv2_image_saved = bridge.imgmsg_to_cv2(ros_image, "bgr8")

    except CvBridgeError as error:
        print(error)


'''def depth_callback(data):
    try:
        cv2_depthImage = self.bridge.imgmsg_to_cv2(data, desired_encoding="passthrough")
        cv2_depthArray = np.array(cv2_ bag.write("/camera/image" , cv2_image_saved[0])depthImage, dtype=np.float32)
        bag.write("/camera/image_raw" , cv2_depthImage)
        # Subscribe only once and then unregister.
        self.subDepth.unregister()
    except CvBridgeError  as error:
        print(error)
'''

if __name__ == '__main__':
  rospy.init_node('listenerNode', anonymous=True)
  imageTopic = "/camera/color/image_rect_color"
  depthTopic = "/camera/aligned_depth_to_color/image_raw"
  #cameraInfoTopic = "/camera/aligned_depth_to_color/camera_info"
  #rospy.init_node('image_converter', anonymous=True)
   
  # Subscriber node using rospy.wait_for_message
  image_sub = rospy.wait_for_message("/camera/color/image_rect_color", Image)
  bag.write("/camera/image" , image_sub)
  #image_callback(Image)

  # Without using rospy.wait_for_message and using the standard rospy.Subscriber()
  '''image_sub = rospy.Subscriber(imageTopic, Image, image_callback, queue_size=1)
  rospy.spin()
  image_sub.unregister()'''
  
  #image_sub = rospy.Subscriber(depthTopic, Image, depth_callback, queue_size=1)
  #rospy.spin()
  #image_sub.unregister()
  
  
