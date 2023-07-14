
import pyrealsense2 as rs2
import argparse
from cv_bridge import CvBridge, CvBridgeError
import numpy as np
import rospy
from sensor_msgs.msg import Image
from sensor_msgs.msg import CameraInfo

class RealsenseSubscriber:
    
    '''Method to initialize the realsense with the bounding box, intrinsics, and image'''
    def __init__(self, imageTopic, depthTopic, cameraInfoTopic, path):
        self.intrinsics = None
        self.extrinsics = None
        self.bridge = CvBridge()
        self.subImage = rospy.Subscriber(imageTopic, Image, self.imageCallback)
        self.subDepth = rospy.Subscriber(depthTopic, Image, self.depthCallback)
        self.subCamInfo = rospy.Subscriber(cameraInfoTopic, CameraInfo, self.get_intrinsic_parameters)
        self.imagePath = path

        self.depthScale =  0.001 

    ''' Method to get intrinsic camera parameters required for projection and deprojection tasks'''
    def get_intrinsic_parameters(self, cameraInfo):
        self.intrinsics = rs2.intrinsics()
        print('Intrinsics Object Created using Pyrealsense!')
        self.intrinsics.width = cameraInfo.width
        self.intrinsics.height = cameraInfo.height
        self.intrinsics.ppx = cameraInfo.K[2]
        self.intrinsics.ppy = cameraInfo.K[5]
        self.intrinsics.fx = cameraInfo.K[0]
        self.intrinsics.fy = cameraInfo.K[4]

        if cameraInfo.distortion_model == 'plumb_bob':
            self.intrinsics.model = rs2.distortion.brown_conrady
        elif cameraInfo.distortion_model == 'equidistant':
            self.intrinsics.model = rs2.distortion.kannala_brandt4
        self.intrinsics.coeffs = [i for i in cameraInfo.D]

        # Subscribe only once and then unregister.
        self.subCamInfo.unregister()   
        print("Params Loaded!")


    '''Method to get extrinsic camera parameters required for projection and deprojection tasks'''
    def get_extrinsic_parameters(self,to_stream, from_stream):
        self.extrinsics = from_stream.get_extrinsics_to(to_stream)
        return self.extrinsics


    '''Method to save the image only corresponding to the image topic'''
    def imageCallback(self, image):
        try:
            self.img = self.bridge.imgmsg_to_cv2(image, desired_encoding='passthrough')
            print('class of image: ', type(self.img))
            print('shape of image: ', self.img.shape)
            # Subscribe only once and then unregister.
            self.subImage.unregister()
        except CvBridgeError as error:
            print(error)


    '''Method to save the depth data by subscribing to the depth topic'''
    def depthCallback(self, data):
        try:
            self.depthImage = self.bridge.imgmsg_to_cv2(data, desired_encoding="passthrough")
            self.depthArray = np.array(self.depthImage, dtype=np.float32)
            # Subscribe only once and then unregister.
            self.subDepth.unregister()
        except CvBridgeError:
            print('Error!!')


    '''Method to deproject a pixel based on the intrinsic camera parameters'''
    def deproject_pixel_to_point(self,pixel):
        depth = self.depthArray[pixel[1], pixel[0]]
        depthScaled = np.true_divide(depth, self.depthScale)
        self.MostRecentDeprojectedPoint = rs2.rs2_deproject_pixel_to_point(self.intrinsics,pixel,depthScaled)
        return  self.MostRecentDeprojectedPoint 


    '''Method to transform the point from the pov of the first camera location to the second camera location'''
    def transform_point(self,pos1_extrinsics,pos2_extrinsics):

        c1_P = np.reshape(self.MostRecentDeprojectedPoint,[3,1])

        c2_T_c1 = np.matmul(np.linalg.inv(pos2_extrinsics),pos1_extrinsics)
        c2_R_c1 = c2_T_c1[0:3,0:3]
        c2_P_c1 = np.reshape(c2_T_c1[0:3,3],[3,1])
        self.MostRecentTransformedPoint = (np.matmul(c2_R_c1,c1_P) + c2_P_c1)

        return self.MostRecentTransformedPoint

    '''Method to project the newly obtained point to pixel'''
    def project_point_to_pixel(self):
        self.MostRecentProjectedPoint = rs2.rs2_project_point_to_pixel(self.intrinsics,self.MostRecentTransformedPoint)
        return self.MostRecentProjectedPoint
    


def main(args):
    file_path = args.file
    print(file_path)
    # get the pixel coordinates from a file and perform the deprojection and projection tasks

    imageTopic = "/camera/color/image_rect_color"
    depthTopic = "/camera/aligned_depth_to_color/image_raw"
    cameraInfoTopic = "/camera/aligned_depth_to_color/camera_info"
    path = ""
    rsObj = RealsenseSubscriber(imageTopic, depthTopic, cameraInfoTopic, path)

    # Todo: need to get the extrinsic parameters from the first camera position and the second camera position
    extrinsinc_pos1 = rsObj.get_extrinsic_parameters()
    extrinsinc_pos2 = rsObj.get_extrinsic_parameters()

    f = open(file_path, "r")
    f2 = open("output.txt","w")
    tranformed_pts = ""
    for line in f:
        line = line.strip("\n")
        pixel = line.split(" ")
        print("For the pixel", pixel)
        print("Deprojected point is",rsObj.deproject_pixel_to_point(pixel))
        print("Transformed point is", rsObj.transform_point(extrinsinc_pos1,extrinsinc_pos2))
        transformed_pixel = rsObj.project_point_to_pixel()
        print("Projected pixel is", transformed_pixel)
        transformed_pts += str(round(transformed_pixel[0])) + " " + str(round(transformed_pixel[1]))+ "\n"

    f2.write(tranformed_pts)
    f.close()
    f2.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", type=str, default="test.txt", help="load_file")
    args = parser.parse_args()
    main(args)

