import cv2
import numpy as np
import cv2.aruco as aruco
import glob
import math
import yaml
import time
import argparse
import sys

class CameraModule:
    def __init__(self, cfg):
        """
        Camera module for the soccer environment.
        Return observations, reward and is_terminate from the camera.

        Args:
            cfg: The configuration file for the camera module.    
        """
        self.camera_cfg = cfg['defaults']['camera']
        ball_cfg = self.camera_cfg['ball']
        robot_cfg = self.camera_cfg['robot']
        reward_condition = self.camera_cfg['reward_condition']
        done_condition = self.camera_cfg['done_condition']
        coord_transformer_cfg = self.camera_cfg['coord_transformer']

        self.ball_position, self.robot_a_pos, self.robot_a_angle, self.robot_b_pos, self.robot_b_angle = None, None, None, None, None 

        self.ball_tracker = BallTracker(ball_cfg)
        self.robot_tracker = RobotTracker(robot_cfg)
        self.reward_processor = RewardProcessor(reward_condition)
        self.done_processor = TerminationDetector(done_condition)        
        self.coord_transformer = CoordinateTransformer(coord_transformer_cfg)

        camera_index = self.camera_cfg['camera_index']
        self.capture = cv2.VideoCapture(camera_index)
        self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        # self.capture.set(cv2.CAP_PROP_FPS, 5)

        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

        objp = np.zeros((6*9,3), np.float32)
        objp[:,:2] = np.mgrid[0:9,0:6].T.reshape(-1,2)

        objpoints = [] # 3d point in real world space
        imgpoints = [] # 2d points in image plane.

        images = glob.glob(f'{self.camera_cfg["calibrate_image_path"]}/calibrate*.jpg')
        print(images)
        for fname in images:
            img = cv2.imread(fname)
            gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

            ret, corners = cv2.findChessboardCorners(gray, (9,6),None)
            if ret == True:
                objpoints.append(objp)
                corners2 = cv2.cornerSubPix(gray,corners,(11,11),(-1,-1),criteria)
                imgpoints.append(corners2)
                img = cv2.drawChessboardCorners(img, (7,6), corners2,ret)
                cv2.imshow('img',img)
                cv2.waitKey(500)
        cv2.destroyAllWindows()

        self.ret, self.mtx, self.dist, self.rvecs, self.tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1],None,None)
        self.robot_tracker.mtx, self.robot_tracker.dist = self.mtx, self.dist

    def camera_init(self):
        """
        Initializes the camera.
        """
        # TODO : below is the example code for the camera initialization
        while True:
            frame = self.get_frame()
            ball_position = self.ball_tracker.track_ball(frame)
            robot_a_pos, robot_a_angle, robot_b_pos, robot_b_angle = self.robot_tracker.track_robot(frame)
            print(ball_position, robot_a_pos, robot_b_pos)
            if (robot_a_angle != None) and (robot_b_angle != None) and (ball_position != None):
                self.ball_position, self.robot_a_pos, self.robot_a_angle, self.robot_b_pos, self.robot_b_angle =\
                ball_position, robot_a_pos, robot_a_angle, robot_b_pos, robot_b_angle

                print("Camera initialized")

                break

    def get_frame(self):
        """
        Gets the current frame from the camera.

        Returns:
            The current frame from the camera.
        """
        _, frame = self.capture.read()
        return frame

    def get_obs(self):
        """
        Gets the current observation, reward and done or not from the camera. 
        observation includes the preprocessed frame, the position of the ball in the 
            frame, and the position of the robot in the frame.

        Returns:
            observation: The current observation from the camera.
            reward: The reward from the current frame.
            done: True if the episode has finished, False otherwise.
        """
        frame = self.get_frame()
        ball_position = self.ball_tracker.track_ball(frame)
        robot_a_pos, robot_a_angle, robot_b_pos, robot_b_angle = self.robot_tracker.track_robot(frame)
        # pixel location to real world location
        self.robot_a_pos = robot_a_pos if robot_a_pos != None else self.robot_a_pos
        self.robot_a_angle = robot_a_angle if robot_a_angle != None else self.robot_a_angle
        self.robot_b_pos = robot_b_pos if robot_b_pos != None else self.robot_b_pos
        self.robot_b_angle = robot_b_angle if robot_b_angle != None else self.robot_b_angle
        self.ball_position = ball_position if ball_position != None else self.ball_position

        ball_position_coord, robot_a_pos_coord, robot_b_pos_coord, = \
            [self.coord_transformer.transform_image_to_real(point) for point in [self.ball_position, self.robot_a_pos, self.robot_b_pos]]
        frame = frame[118:430, :]
        observation = [frame, ball_position_coord, robot_a_pos_coord, self.robot_a_angle, robot_b_pos_coord, self.robot_b_angle]
        opponent_observation = self.get_opponent_obs(observation)
        observations = [observation, opponent_observation]
        done = self.done_processor.is_terminate(ball_position_coord)
        reward = self.reward_processor.get_reward(ball_position_coord, done)
        return observations, reward, done
    
    def get_opponent_obs(observation):
        frame, ball_position_coord, robot_a_pos_coord, robot_a_angle, robot_b_pos_coord, robot_b_angle = observation

        frame = cv2.rotate(frame, cv2.ROTATE_180)
        ball_position_coord = [1150 - ball_position_coord[0], 548 - ball_position_coord[1]]
        robot_a_pos_coord = [1150 - robot_b_pos_coord[0], 548 - robot_b_pos_coord[1]]
        robot_a_angle = 360 - robot_b_angle
        robot_b_pos_coord = [1150 - robot_a_pos_coord[0], 548 - robot_a_pos_coord[1]]
        robot_b_angle = 360 - robot_a_angle

        return [frame, ball_position_coord, robot_a_pos_coord, robot_a_angle, robot_b_pos_coord, robot_b_angle]


    
    def test_get_obs(self):
        """
        Draw the observation on the frame and show it.
        """
        observation, reward, done = self.get_obs()
        frame, ball_position, robot_a_pos, robot_a_angle, robot_b_pos, robot_b_angle = observation
        # real world location to pixel location
        ball_position, robot_a_pos, robot_b_pos, = \
            [self.coord_transformer.transform_real_to_image(point) for point in [ball_position, robot_a_pos, robot_b_pos]]
        # draw ball and robot on the frame
        cv2.circle(frame, ball_position, 10, (0, 0, 255), -1)
        cv2.circle(frame, robot_a_pos, 10, (0, 255, 0), -1)
        cv2.circle(frame, robot_b_pos, 10, (0, 255, 0), -1)        
        # write reward and done on the frame
        cv2.putText(frame, 'reward: {}'.format(reward), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)        
        cv2.putText(frame, 'done: {}'.format(done), (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow('frame', frame)
        
    
    def close(self):
        """
        Closes the camera.
        """
        self.capture.release()
        cv2.destroyAllWindows()
    
class RewardProcessor:
    def __init__(self, reward_condition):
        """
        Reward module for the soccer environment.
        Calculates reward from the camera.

        Args:
            reward_condition: The condition to calculate the reward from.
        """
        self.ball_reward = reward_condition['ball_reward']
        self.done_reward = reward_condition['done_reward']

        self.x0, self.r0 = self.ball_reward[0] # 100, -0.01
        self.x1, self.r1 = self.ball_reward[1] # 1050, 0.01
        self.done_x0, self.done_r0 = self.done_reward[0] # 100, -1
        self.done_x1, self.done_r1 = self.done_reward[1] # 1050, 1

    def get_reward(self, ball_position, done):
        """
        Gets the reward from the frame.

        Args:
            ball_position: The position of the ball in the frame.
            done: True if the episode has finished, False otherwise.

        Returns:
            The reward from the frame.
        """
        if done:
            if ball_position[0] < self.done_x0:
                return self.done_r0
            elif ball_position[0] > self.done_x1:
                return self.done_r1
        else:
            return (ball_position[0] - self.x0) / (self.x1 - self.x0) * (self.r1 - self.r0) + self.r0
    
    

class TerminationDetector:
    def __init__(self, done_condition):
        """
        Detection module for the soccer environment.

        Args:
            done_condition: The condition to calculate if the episode should terminate from.
        """
        self.goal_pos_left, self.goal_pos_right = done_condition

    def is_terminate(self, ball_position):
        """
        Checks if the episode should terminate.

        Args:
            ball_position: The position of the ball in the frame.
        """
        if (ball_position[0] > self.goal_pos_right) or (ball_position[0] < self.goal_pos_left):
            return True
        else:
            return False

class BallTracker:
    def __init__(self, ball_cfg):
        """
        Module for tracking the ball in the frame.

        Args:
            ball_template_path: Path to the ball template image.        
        """
        ball_template_paths = ball_cfg['template_path']
        self.boundary_points = ball_cfg['boundary_points']
        self.ball_template_list = []
        for ball_template_path in ball_template_paths:
            self.ball_template_list.append(cv2.imread(ball_template_path))
        # self.ball_template = cv2.imread(ball_template_path) 
        # _, self.w, self.h = ball_template_path[0].shape[::-1]  # Template width and height
        # print(self.w, self.h)

    def track_ball(self, frame):
        """
        Tracks the ball in the frame.

        Args:
            frame: The frame to track the ball in.

        Returns:
            The position of the ball in the frame (center point of the matched area).
        """
        # gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # Convert the frame to grayscale
        
        # List to store the results of all the scales
        results = []

        # Multi-scale template matching
        for ball_template in self.ball_template_list:
            # Loop over the scales of the image
            for scale in np.linspace(0.95, 1.05, 3)[::-1]:
                # Resize the image according to the scale, and keep track
                # of the ratio of the resizing
                _, w, h = ball_template.shape[::-1]
                resized_template = cv2.resize(ball_template, None, fx=scale, fy=scale)
                r = ball_template.shape[1] / float(resized_template.shape[1])

                # If the resized image is smaller than the template, then break
                # from the loop
                if resized_template.shape[0] > frame.shape[0] or resized_template.shape[1] > frame.shape[1]:
                    break
                
                # Apply template matching to find the template in the image
                res = cv2.matchTemplate(frame, resized_template, cv2.TM_CCOEFF_NORMED)
                (_, maxVal, _, maxLoc) = cv2.minMaxLoc(res)
                if (self.boundary_points[0][0] < maxLoc[0] * r < self.boundary_points[0][1]) and (self.boundary_points[1][0] < maxLoc[1] * r < self.boundary_points[1][1]):
                    pass
                else:
                    continue

                # Save the result
                results.append((maxVal, maxLoc, r, w, h))
        if len(results) == 0:
            return None
        # Select the best match with the highest value
        (_, maxLoc, r, w, h) = max(results, key=lambda x: x[0])

        # Compute the (x, y) coordinates of the bounding box for the object
        (startX, startY) = (int(maxLoc[0] * r), int(maxLoc[1] * r))
        (endX, endY) = (int((maxLoc[0] + w) * r), int((maxLoc[1] + h) * r))
        center = (int((startX + endX) / 2), int((startY + endY) / 2))

        return center


class RobotTracker:
    def __init__(self, robot_cfg):
        """
        Module for tracking the robot in the frame.

        Args:
            robot_template_path: Path to the robot template image.
        """
        self.robot_a_aruco_id = robot_cfg['robot_a_aruco_id']
        self.robot_b_aruco_id = robot_cfg['robot_b_aruco_id']
        self.arucoDict = aruco.Dictionary_get(aruco.DICT_5X5_250)
        self.arucoParams = aruco.DetectorParameters_create()
        self.arucoParams.adaptiveThreshConstant = 10
        
        self.mtx, self.dist = None, None

    def track_robot(self, frame):
        """
        Tracks the robot in the frame.

        Args:
            frame: The frame to track the robot in.

        Returns:
            pos: The position of the robot in the frame.
            angle: The angle of the robot in the frame.
        """
        corners, ids, rvec, tvec = self.detect_aruco_marker(frame)
        print(ids)
        robot_a_pos, robot_a_angle = self.detect_aruco_marker_id(self.robot_a_aruco_id, corners, ids, rvec)
        robot_b_pos, robot_b_angle = self.detect_aruco_marker_id(self.robot_b_aruco_id, corners, ids, rvec)
        
        return robot_a_pos, robot_a_angle, robot_b_pos, robot_b_angle
        # return None, None, None, None


    def detect_aruco_marker(self, frame):
        """
        detects the aruco marker in the frame.

        Args:
            frame: The frame to detect the aruco marker in.

        Returns:    
            corners: The corners of the detected aruco marker.
            ids: The ids of the detected aruco marker.
            rvec: The rotation vector of the detected aruco marker.
            tvec: The translation vector of the detected aruco marker.        
        """
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        (corners, ids, rejectedImgPoints) = aruco.detectMarkers(gray_frame, self.arucoDict, parameters=self.arucoParams)
        
        rvec, tvec ,_ = aruco.estimatePoseSingleMarkers(corners, 0.05, self.mtx, self.dist)

        return corners, ids, rvec, tvec
    
    def detect_aruco_marker_id(self, num_id, corners, ids, rvec):
        """
        detect the aruco marker with the given id.

        Args:
            num_id: The id of the aruco marker to detect.
            corners: The corners of the detected aruco marker.
            ids: The ids of the detected aruco marker.
            rvec: The rotation vector of the detected aruco marker.

        Returns:
            center: The center of the detected aruco marker.
            angle: The yaw angle of the detected aruco marker.        
        """
        if num_id in np.ravel(ids) :
            index = np.where(ids == num_id)[0][0] #Extract index of num_id
            cornerUL = corners[index][0][0]
            cornerUR = corners[index][0][1]
            cornerBR = corners[index][0][2]
            cornerBL = corners[index][0][3]

            center = [ (cornerUL[0]+cornerBR[0])/2 , (cornerUL[1]+cornerBR[1])/2 ]

            R, _ = cv2.Rodrigues(rvec[np.where(ids==num_id)[0]][0][0])
            sin_x    = math.sqrt(R[2,0] * R[2,0] +  R[2,1] * R[2,1])    
            singular  = sin_x < 1e-6
            if not singular:
                z1    = math.atan2(R[2,0], R[2,1])     # around z1-axis
                x      = math.atan2(sin_x,  R[2,2])     # around x-axis
                z2    = math.atan2(R[0,2], -R[1,2])    # around z2-axis
            else: # gimbal lock
                z1    = 0                                         # around z1-axis
                x      = math.atan2(sin_x,  R[2,2])     # around x-axis
                z2    = 0                                         # around z2-axis

            # return np.array([[z1], [x], [z2]])

            yawpitchroll_angles = -180*np.array([[z1], [x], [z2]])/math.pi
            yawpitchroll_angles[0,0] = (360-yawpitchroll_angles[0,0])%360 # change rotation sense if needed, comment this line otherwise
            yawpitchroll_angles[1,0] = yawpitchroll_angles[1,0]+90

            pos = center
            angle = yawpitchroll_angles[0]
            print(pos, center)
            return pos, angle
        else:
            return None, None

class CoordinateTransformer:
    def __init__(self, coord_transformer_cfg):
        """
        Module for transforming image coordinates to real-world coordinates.

        Args:
            src_points: List of points in the image (pixel coordinates).
            dst_points: List of corresponding points in the real world.
        """
        src_points, dst_points = coord_transformer_cfg['src_points'], coord_transformer_cfg['dst_points']
        self.transformation_matrix = cv2.getPerspectiveTransform(np.array(src_points, dtype='float32'),
                                                                  np.array(dst_points, dtype='float32'))
        self.inverse_transformation_matrix = np.linalg.inv(self.transformation_matrix)
        

    def transform_image_to_real(self, point):
        """
        Transform a point from image coordinates to real-world coordinates.

        Args:
            point: pixel location in the image.

        Returns:
            The corresponding point in the real world.
        """
        if point:
            point_array = np.array([[[point[0], point[1]]]], dtype='float32')
            transformed_point = cv2.perspectiveTransform(point_array, self.transformation_matrix)
            return transformed_point[0][0]
        else:
            return None
    
    def transform_real_to_image(self, point):
        """
        Transform a point from real-world coordinates to image coordinates.

        Args:
            point: point in the real world.

        Returns:
            The corresponding point in the image.
        """
        # Create an array for the point
        point_array = np.array([[[point[0], point[1]]]], dtype='float32')
        
        # Transform the point using the inverse transformation matrix
        transformed_point = cv2.perspectiveTransform(point_array, self.inverse_transformation_matrix)
        
        return transformed_point[0][0]

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='./config.yaml', help='Path to config file')
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    camera = CameraModule(cfg)
    # frame = cv2.imread('./ball_coord_test/ball0_0.jpg')
    while True:
        frame = camera.get_frame()
        ball_pos = camera.ball_tracker.track_ball(frame)
        ball_coord = camera.coord_transformer.transform_image_to_real(ball_pos)
        ball_position = [camera.coord_transformer.transform_real_to_image(point) for point in [ball_coord]][0]
        # draw ball and robot on the frame
        cv2.circle(frame, ball_pos, 10, (0, 0, 255), -1)
        cv2.imshow('image', frame)
        cv2.waitKey(2000)
        done = camera.done_processor.is_terminate(ball_coord)
        reward = camera.reward_processor.get_reward(ball_coord, done)
        robot_a_pos, robot_a_angle, robot_b_pos, robot_b_angle = camera.robot_tracker.track_robot(frame)
        print(done, reward)
        print(robot_a_pos, robot_a_angle, robot_b_pos, robot_b_angle)

    if frame is None:
        print("Failed to capture frame or load image.")
    else:
        cv2.imshow('image', frame)
        cv2.waitKey(1000)

    ball_pos = camera.ball_tracker.track_ball(frame)
    ball_coord = camera.coord_transformer.transform_image_to_real(ball_pos)
    ball_position = [camera.coord_transformer.transform_real_to_image(point) for point in [ball_coord]][0]
    # draw ball and robot on the frame
    cv2.circle(frame, ball_pos, 10, (0, 0, 255), -1)
    cv2.imshow('image', frame)
    cv2.waitKey(2000)
    done = camera.done_processor.is_terminate(ball_coord)
    reward = camera.reward_processor.get_reward(ball_coord, done)
    print(done, reward)


    # camera.camera_init()
    # # frame = camera.get_frame()
    # # frame = cv2.imread('../assets/frame_1652.jpg')
    # # frame = cv2.imread('../assets/calibrate3.jpg')
    # # if frame is None:
    # #     print("Failed to capture frame or load image.")
    # # else:
    # #     cv2.imshow('image', frame)
    # # cv2.waitKey(0)
    # # cv2.destroyAllWindows()
    # # print(camera.get_obs(frame))
    # while True:
    #     print(camera.get_obs())


    # camera_index = 1

    # # Open the selected camera
    # cap = cv2.VideoCapture(camera_index)
    # if not cap.isOpened():
    #     print("Error: Could not open the selected camera.")
    #     sys.exit()

    # # Set the desired FPS
    # desired_fps = 30
    # cap.set(cv2.CAP_PROP_FPS, desired_fps)
    # # Calculate the time delay between frames based on the desired FPS
    # frame_delay = 1.0 / desired_fps

    # # Get and display the camera's frame size and resolution
    # frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    # frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    # print(f"Camera frame size: {frame_width} x {frame_height}")

    # # Create a window to display the webcam feed
    # cv2.namedWindow("Webcam Feed", cv2.WINDOW_NORMAL)
    # cv2.resizeWindow("Webcam Feed", frame_width, frame_height)

    # frame_count = 0
    # start_time = time.time()
    # prev_frame_time = start_time

    # while True:
    #     # Capture a frame from the selected camera
    #     ret, frame = cap.read()

    #     if not ret:
    #         print("Error: Could not read a frame from the camera.")
    #         break

    #     # Display the frame in the created window
    #     cv2.imshow("Webcam Feed", frame)

    #     # Calculate and display the actual FPS
    #     frame_count += 1
    #     elapsed_time = time.time() - start_time
    #     actual_fps = frame_count / elapsed_time
    #     print(f"Actual FPS: {actual_fps:.2f}", end='\r')

    #     # Wait for the desired time delay between frames
    #     current_time = time.time()
    #     time_diff = current_time - prev_frame_time
    #     if time_diff < frame_delay:
    #         time.sleep(frame_delay - time_diff)
    #     prev_frame_time = current_time

    #     if cv2.waitKey(1) & 0xFF == ord('c'):
    #         cv2.imwrite(f"./frame_{frame_count}.jpg", frame)
    #         print(f"Saved frame_{frame_count}.jpg")

    #     # Exit the loop if the 'q' key is pressed
    #     if cv2.waitKey(1) & 0xFF == ord('q'):
    #         break

    # # Release the camera and close the window
    # cap.release()
    # cv2.destroyAllWindows()