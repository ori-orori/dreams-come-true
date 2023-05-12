import gym
import numpy as np
from gym import spaces
import cv2
import time

class SoccerEnv:
    def __init__(self,
        cfg,
        models,
        camera,
        communication,        
        ):

        super().__init__()

        env_cfg = cfg.env
        self.mode = env_cfg['mode']
        self.crop_size = env_cfg['crop_size']
        self.robot_fps = env_cfg['robot_fps']

        self.models = models
        self.camera = camera
        self.communication = communication
        self.num_actions = int(cfg.num_actions)

        # Define the action space for the environment
        # self.action_space = spaces.Discrete(5)  # e.g., 0: stop, 1: move forward, 2: move backward, 3: turn left, 4: turn right
        # self.action_space.discrete = True

        # Define the observation space for the environment
        height, width = self.crop_size
        self.observation_space = spaces.Box(low=0, high=255, shape=(height, width), dtype=np.uint8)


    @property
    def action_space(self):
        space = self.env.action_space
        space.discrete = True
        return space
    
    def predict(self, obs_list, state_list):
        if self.mode == 'self_play':
            state1, state2 = state_list
            obs1, obs2 = obs_list
            (robot1_action, state1), (robot2_action, state2) = self.models[0]._policy(obs1, state1), self.models[1]._policy(obs2, state2) # return 2 integers
            action_list, state_list = (robot1_action, robot2_action), (state1, state2)

            return action_list, state_list
        elif self.mode == 'human':
            state1 = state_list
            obs1 = obs_list
            (robot1_action, state1) = self.models[0]._policy(obs1, state1) # return 1 integer
            action_list, state_list = (robot1_action), (state1)
            return action_list, state_list
        else:
            raise NotImplementedError

    def step(self, action):
        if self.mode == 'self_play':
            robot1_action, robot2_action = action
        elif self.mode == 'human':
            robot1_action = action
        else:
            raise NotImplementedError
        # Execute the action in the environment and update the environment state

        # TODO: send the action to the robot through Wi-Fi communication
        if self.mode == 'self_play':
            self.communication.send_action_to_robot('A', robot1_action)
            self.communication.send_action_to_robot('B', robot2_action)
        elif self.mode == 'human':
            self.communication.send_action_to_robot('A', robot1_action)

        # Determine if the episode has finished
        time.sleep(1/self.robot_fps) # wait for the robot to execute the action

        # Get the current camera frame
        obs, reward_self, done = self.camera.get_obs()
        frame_self, ball_position_coord_self, robot_a_pos_coord_self, robot_a_angle_self, robot_b_pos_coord_self, robot_b_angle_self = obs[0]
        frame_oppo, ball_position_coord_oppo, robot_a_pos_coord_oppo, robot_a_angle_oppo, robot_b_pos_coord_oppo, robot_b_angle_oppo = obs[1]   
        frame_self = cv2.resize(frame_self, self.crop_size, interpolation=cv2.INTER_AREA)
        frame_oppo = cv2.resize(frame_oppo, self.crop_size, interpolation=cv2.INTER_AREA)

        reward_oppo = (-1) * reward_self
        done = self.is_done()
        self.step += 1

        if done:
            self.done = True
        else:
            self.done = False

        obs_list = [{"image": frame_self,
             'self_position':(robot_a_pos_coord_self, robot_a_angle_self),
             'opponent_position':(robot_b_pos_coord_self, robot_b_angle_self),
             'ball_position':ball_position_coord_self,
             "is_terminal": self.done,
             "is_first": False},
             {"image": frame_oppo,
             'self_position':(robot_a_pos_coord_oppo, robot_a_angle_oppo),
             'opponent_position':(robot_b_pos_coord_oppo, robot_b_angle_oppo),
             'ball_position':ball_position_coord_oppo,
             "is_terminal": self.done,
             "is_first": False}]
        reward_list = [reward_self, reward_oppo]
        done = self.done
        info = {}
        return obs_list, reward_list, done, info
        

    def reset(self):
        # Reset the environment state and return the initial observation
        print("After reset press Enter to continue...")
        input()
        print("Continuing...")
        # self.camera.close()
        # self.communication.connect()
        # self.camera.camera_init()
        self.camera.camera_init()

        obs, reward_self, _ = self.camera.get_obs()
        frame_self, ball_position_coord_self, robot_a_pos_coord_self, robot_a_angle_self, robot_b_pos_coord_self, robot_b_angle_self = obs[0]
        frame_oppo, ball_position_coord_oppo, robot_a_pos_coord_oppo, robot_a_angle_oppo, robot_b_pos_coord_oppo, robot_b_angle_oppo = obs[1]   
        frame_self = cv2.resize(frame_self, self.crop_size, interpolation=cv2.INTER_AREA)
        frame_oppo = cv2.resize(frame_oppo, self.crop_size, interpolation=cv2.INTER_AREA)
        reward_self, done = 0, False
        reward_oppo = (-1) * reward_self
        self.step = 0

        obs_list = [{"image": frame_self,
             'self_position':(robot_a_pos_coord_self, robot_a_angle_self),
             'opponent_position':(robot_b_pos_coord_self, robot_b_angle_self),
             'ball_position':ball_position_coord_self,
             "is_terminal": False,
             "is_first": True},
             {"image": frame_oppo,
             'self_position':(robot_a_pos_coord_oppo, robot_a_angle_oppo),
             'opponent_position':(robot_b_pos_coord_oppo, robot_b_angle_oppo),
             'ball_position':ball_position_coord_oppo,
             "is_terminal": False,
             "is_first": True}]
        reward_list = [reward_self, reward_oppo]
        info = {}
        return obs_list, reward_list, done, info

    def close(self):
        self.camera.close()
        self.communication.close()

    @property
    def action_space(self):
        space = self.env.action_space
        space.discrete = True
        return space