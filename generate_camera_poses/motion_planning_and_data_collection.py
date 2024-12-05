import json
import numpy as np
from scipy.spatial.transform import Rotation as R
import actionlib
from geometry_msgs.msg import Pose
from automated_image_capture.msg import MotionExecutionAction, MotionExecutionGoal
import rospy
import rosbag
from sensor_msgs.msg import JointState, Image, CameraInfo


def capture_images_along_trajectory(index):
    bag_name = f"spring24_bag{index}.bag"  # Construct the bag name using the given index
    
    image_topic = "/camera/color/image_rect_color"
    depth_topic = "/camera/aligned_depth_to_color/image_raw"
    camera_info_topic = "/camera/aligned_depth_to_color/camera_info"
    
    # Open the bag file for writing
    with rosbag.Bag(bag_name, 'w') as bag:
        joint_state = rospy.wait_for_message("/joint_states", JointState)
        bag.write("/camera/joint_states", joint_state)
        
        camera_pose = rospy.wait_for_message("/panda_camera_pose", Pose)
        bag.write("/camera/camera_pose", camera_pose)
        
        color_image = rospy.wait_for_message(image_topic, Image)
        bag.write(image_topic, color_image)
        
        depth_image = rospy.wait_for_message(depth_topic, Image)
        bag.write(depth_topic, depth_image)
        
        camera_info = rospy.wait_for_message(camera_info_topic, CameraInfo)
        bag.write(camera_info_topic, camera_info)
    
    return 1

def load_poses_from_json(file_path):
    """Load poses from a JSON file and convert them to a list of Pose objects."""
    with open(file_path, 'r') as f:
        data = json.load(f)

    # Sort keys numerically and extract poses
    ordered_poses = []
    for key in sorted(data.keys(), key=int):  # Sort keys as integers
        for matrix in data[key]:  # Each matrix is a 4x4 transformation
            # Convert 4x4 matrix to Pose
            matrix_np = np.array(matrix)
            pose = Pose()
            pose.position.x = matrix_np[0, 3]
            pose.position.y = matrix_np[1, 3]
            pose.position.z = matrix_np[2, 3]

            rotation = R.from_matrix(matrix_np[:3, :3])
            quaternion = rotation.as_quat()
            pose.orientation.x = quaternion[0]
            pose.orientation.y = quaternion[1]
            pose.orientation.z = quaternion[2]
            pose.orientation.w = quaternion[3]

            ordered_poses.append(pose)

    return ordered_poses

def execute_poses_sequentially(pose_coll):
    """Execute poses in the given order."""
    # Initialize the motion execution client
    motion_execution_client = actionlib.SimpleActionClient('PandaMotionExecutionActionServer', MotionExecutionAction)
    motion_execution_client.wait_for_server()

    print("Starting sequential pose execution...")
    for idx, pose in enumerate(pose_coll):
        print(f"Executing pose {idx + 1}/{len(pose_coll)}")
        goal = MotionExecutionGoal()
        goal.ee_trajectory = [pose]
        goal.gripper_state = [False]  # Adjust gripper state if needed

        # Send the goal to the server
        motion_execution_client.send_goal(goal)
        motion_execution_client.wait_for_result()
        result = motion_execution_client.get_result()

        if result.result == result.SUCCESS:
            print(f"Pose {idx + 1} execution started.")
            capture_images_along_trajectory(idx + 1)
            print(f"Pose {idx + 1} executed successfully.")
        else:
            print(f"Failed to execute pose {idx + 1}.")

    print("Pose execution completed.")

if __name__ == '__main__':
    rospy.init_node('main')
    # Path to the JSON file
    json_file_path = "grouped_end_effector_poses.json"  

    # Load poses from JSON and execute them sequentially
    ordered_poses = load_poses_from_json(json_file_path)
    execute_poses_sequentially(ordered_poses)
