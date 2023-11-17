import cv2
import rosbag
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

# Define the path to your ROS bag file
bag_file_path = 'test_1.bag'

# Open the ROS bag file
bag = rosbag.Bag(bag_file_path, 'r')

# Initialize CvBridge
bridge = CvBridge()

# Iterate through the messages in the ROS bag
for topic, msg, t in bag.read_messages(topics=['/camera/image']):
        try:
            # Convert the ROS image message to an OpenCV image
            cv_image = bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
            # Process the image (e.g., display it)
            cv2.imshow('Image from ROS Bag', cv_image)
            cv2.waitKey(700) 

        except Exception as e:
            print(f"Error processing image: {e}")

# Close the ROS bag file
bag.close()

