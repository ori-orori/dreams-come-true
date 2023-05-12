"""
Thanks to njanirudh
Reference : Aruco_Tracker (https://github.com/njanirudh/Aruco_Tracker.git)
"""

import cv2
import numpy as np
import cv2.aruco as aruco
import glob
import numpy as np
import math

class CameraModule:
    # def __init__(self, cfg):
    #     """
    #     Camera module for the soccer environment.
    #     Return observations, reward and is_terminate from the camera.

    #     Args:
    #         cfg: The configuration file for the camera module.    
    #     """
    #     camera_cfg = cfg['camera']
    #     ball_template_path = camera_cfg['ball_template_path']
    #     robot_template_path = camera_cfg['robot_template_path']
    #     reward_condition = camera_cfg['reward_condition']
    #     done_condition = camera_cfg['done_condition']

    #     self.ball_tracker = BallTracker(ball_template_path)
    #     self.robot_tracker = RobotTracker(robot_template_path)
    #     self.reward_processor = RewardProcessor(reward_condition)
    #     self.done_processor = TerminationDetector(done_condition)

    def __init__(self):
        self.ball_tracker = 1
        
    def camera_init(self):
        """
        Initializes the camera.
        """
        # TODO : below is the example code for the camera initialization
        self.capture = cv2.VideoCapture(4)
        self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

        objp = np.zeros((6*7,3), np.float32)
        objp[:,:2] = np.mgrid[0:7,0:6].T.reshape(-1,2)

        objpoints = [] # 3d point in real world space
        imgpoints = [] # 2d points in image plane.

        images = glob.glob('calib_images/*.png')

        for fname in images:
            img = cv2.imread(fname)
            gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

            ret, corners = cv2.findChessboardCorners(gray, (7,6),None)

            if ret == True:
                objpoints.append(objp)
                corners2 = cv2.cornerSubPix(gray,corners,(11,11),(-1,-1),criteria)
                imgpoints.append(corners2)
                img = cv2.drawChessboardCorners(img, (7,6), corners2,ret)

        self.ret, self.mtx, self.dist, self.rvecs, self.tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1],None,None)
        # return ret, mtx, dist, rvecs, tvecs
        # raise NotImplementedError

    def get_frame(self):
        """
        Gets the current frame from the camera.

        Returns:
            The current frame from the camera.
        """
        # self.capture = cv2.VideoCapture(0)
        ret, frame = self.capture.read()
        return frame
        # raise NotImplementedError

    def detect_id(self, num_id, frame, corners, ids, rvec, tvec):
        if num_id in np.ravel(ids) :
            
            index = np.where(ids == num_id)[0][0] #Extract index of num_id
            cornerUL = corners[index][0][0]
            cornerUR = corners[index][0][1]
            cornerBR = corners[index][0][2]
            cornerBL = corners[index][0][3]

            center = [ (cornerUL[0]+cornerBR[0])/2 , (cornerUL[1]+cornerBR[1])/2 ]
            
            text='[' + str(center[0]) + ','+ str(center[1])  + ']' 
            org=(num_id, num_id) # (int(center[0]), int(center[1]))
            arrow_i = (int(center[0]), int(center[1]))
            arrow_f = (int(cornerUR[0]), int(cornerUR[1]))
            font=cv2.FONT_HERSHEY_SIMPLEX
            frame = cv2.putText(frame.copy(),text,org,font,1,(255,0,0),2)
            frame = cv2.arrowedLine(frame.copy(), arrow_i, arrow_f, (255,255,0), 2)
            
            # global yawpitchroll_angles
            global pos
            global angle

            if len(np.where(ids==num_id)[0])!=0:
                # if num_id == 200:
                global yawpitchroll_angles
                # print('rvec : ', rvec[np.where(ids==num_id)[0]][0][0]*180/np.pi, ' tvec : ', tvec[np.where(ids==num_id)[0]][0][0])
                frame = cv2.drawFrameAxes(frame.copy(), self.mtx, self.dist, rvec[np.where(ids==num_id)[0]], tvec[np.where(ids==num_id)[0]], 0.1)
        
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
                # global yawpitchroll_angles
                yawpitchroll_angles = -180*np.array([[z1], [x], [z2]])/math.pi
                yawpitchroll_angles[0,0] = (360-yawpitchroll_angles[0,0])%360 # change rotation sense if needed, comment this line otherwise
                yawpitchroll_angles[1,0] = yawpitchroll_angles[1,0]+90
                
                # print('angle : ', yawpitchroll_angles[0])
                # return z1, x, z2
                
                pos = center
                angle = yawpitchroll_angles[0]
        #         else:
        #             # global pos, angle
        #             pos = None
        #             angle = None
        #     else:
        #         # global pos, angle
        #         pos = None
        #         angle = None
        # else:
        #     # global pos, angle
        #     pos = None
        #     angle = None
        return frame, pos, angle

    def get_info(self):
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
        # cropped_frame = self.crop_frame(frame)
        # cropped_frame = frame
        
        ball_position, robot_info = self.get_obs(frame)
        # observation = [cropped_frame, ball_position, robot_info]
        # done = self.done_processor.is_terminate(ball_position)
        # reward = self.reward_processor.get_reward(ball_position, done)
        
        # display the resulting frame
        cv2.imshow('frame',frame)
        cv2.waitKey(1)      
        
        # return observation, reward, done
        return ball_position, robot_info
    
    def get_obs(self, frame):
        """
        Gets the observation from the frame.

        Args:
            frame: The cropped frame to get the observation from.

        Returns:
            ball_position: The position of the ball in the frame.
            robot_info: The position of the robot in the frame.
        """
        # cap, ret, mtx, dist, rvecs, tvecs        
        
        # ret, frame = self.capture.read()
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        aruco_dict = aruco.Dictionary_get(aruco.DICT_5X5_250)
        parameters = aruco.DetectorParameters_create()
        parameters.adaptiveThreshConstant = 10
        corners, ids, rejectedImgPoints = aruco.detectMarkers(gray, aruco_dict, parameters=parameters)

        font = cv2.FONT_HERSHEY_SIMPLEX
        
        if np.all(ids != None):
            global pos_ball 
            rvec, tvec ,_ = aruco.estimatePoseSingleMarkers(corners, 0.05, self.mtx, self.dist)
            frame = aruco.drawDetectedMarkers(frame.copy(), corners, ids)
                
            id_ball = 100 # [70 42 24 66 87]
            id_robot = 120
            
            
            frame, pos_ball, anble_ball = self.detect_id(id_ball, frame.copy(), corners, ids, rvec, tvec)
            frame, pos_robot, angle_robot = self.detect_id(id_robot, frame.copy(), corners, ids, rvec, tvec)
            
            strg = ''
            for i in range(0, ids.size):
                strg += str(ids[i][0])+', '

            cv2.putText(frame, "Id: " + strg, (0,64), font, 1, (0,255,0),2,cv2.LINE_AA)
            
            ball_position = pos_ball
            # print('angle : ', type(angle_robot), angle_robot)
            robot_info = pos_robot + angle_robot.tolist()
            
            

        else:
            frame = frame.copy()
            cv2.putText(frame, "No Ids", (0,64), font, 1, (0,255,0),2,cv2.LINE_AA)
            print('Try again')
            ball_position = []
            robot_info = []

        return ball_position, robot_info
        

    def crop_frame(self, frame, crop_size=(200, 200)):
        """
        Crops the frame to the specified size.
        
        Args:
            frame: The frame to crop.
            crop_size: The size of the frame to crop to. Should be a tuple of (width, height).

        Returns:
            The cropped frame.
        """
        raise NotImplementedError

class RewardProcessor:
    def __init__(self, reward_condition):
        """
        Reward module for the soccer environment.
        Calculates reward from the camera.

        Args:
            reward_condition: The condition to calculate the reward from.
        """
        self.reward_condition = reward_condition


    def get_reward(self, ball_position, done):
        """
        Gets the reward from the frame.

        Args:
            ball_position: The position of the ball in the frame.
            done: True if the episode has finished, False otherwise.

        Returns:
            The reward from the frame.
        """
        raise NotImplementedError

class TerminationDetector:
    def __init__(self, done_condition):
        """
        Detection module for the soccer environment.

        Args:
            done_condition: The condition to calculate if the episode should terminate from.
        """
        self.done_condition = done_condition

    def is_terminate(self, ball_position):
        """
        Checks if the episode should terminate.

        Args:
            ball_position: The position of the ball in the frame.

        """
        raise NotImplementedError

class BallTracker:
    def __init__(self, ball_template_path):
        """
        Module for tracking the ball in the frame.

        Args:
            ball_template_path: Path to the ball template image.        
        """
        self.ball_template = cv2.imread(ball_template_path)

    def track_ball(self, frame):
        """
        Tracks the ball in the frame.

        Args:
            frame: The frame to track the ball in.

        Returns:
            The position of the ball in the frame.
        """
        raise NotImplementedError

class RobotTracker:
    def __init__(self, robot_template_path):
        """
        Module for tracking the robot in the frame.

        Args:
            robot_template_path: Path to the robot template image.
        """
        self.robot_template = cv2.imread(robot_template_path)

    def track_robot(self, frame):
        """
        Tracks the robot in the frame.

        Args:
            frame: The frame to track the robot in.

        Returns:
            The position of the robot in the frame.
        """
        raise NotImplementedError
    


if __name__ == '__main__':
    cam = CameraModule()
    # cam.camera_init()
    # cam.get_obs()
    # cam.get_info()
    cam.camera_init()
    while True:
        ball_position, robot_info = cam.get_info()
        if len(ball_position) != 0 and len(robot_info) != 0:
            print('ball_pos : ', ball_position)
            print('robot_info : ', robot_info)
    
