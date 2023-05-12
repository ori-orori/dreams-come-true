import time

from communication  import ClientCommunication
from camera import CameraModule
from env.soccer_env import SoccerEnv
import datetime
import pathlib
import numpy as np
import io
import uuid
import cv2

from client_dreamer import Client_Dreamer
from tools import Logger

class Client:
    def __init__(self, cfg):
        """
        Class that manage everything on the client side

        Args:
            cfg: The configuration file for the client.    
        """
        self.frequency = cfg.frequency
        self.camera_cfg = cfg.camera
        # self.communication_cfg = cfg['communication']
        # self.env_cfg = cfg.env

        self.camera = CameraModule(cfg)
        self.communication = ClientCommunication(cfg)
        self.models = [Client_Dreamer(cfg), Client_Dreamer(cfg)] 
        self.env = SoccerEnv(
        cfg,
        self.models,
        self.camera,
        self.communication,
        )

    def initialize(self):
        """
        initialize the connection to the robot, server and the camera
        """
        self.communication.connect()
        self.camera.camera_init()

    def run(self):
        """
        Run the client
        """
        while True:
            # Loop every 1/self.frequency seconds
            time.sleep(1/self.frequency)

            # Get observation from camera
            info = self.camera.get_info() # info = (observation, reward, done)
            _, _, done = info

            # Send observation to server
            self.communication.send_info_to_server(info)

            # If done in observation: break
            if done:
                break

            # Receive action from server
            robot, action = self.communication.receive_action_from_server()

            # Send action to robot
            self.communication.send_action_to_robot(robot, action)

    def close(self):
        """
        Close robot and server connection
        """
        self.communication.close()

    def collect_data(self, max_steps, directory='./train_dir'):
        """
        Collect data from the robot
        """
        directory = pathlib.Path(directory).expanduser()
        directory.mkdir(parents=True, exist_ok=True)
        steps = 0
        done = True
        while steps < max_steps:
            if done:
                obs = self.env.reset()
                transition = obs_list[0].copy()
                transition["reward"] = reward_list[0]
                transition["discount"] = 1.0
                self.episode = [transition]
            else:
                action_list, state_list = self.env.predict(obs_list, state_list)
                obs_list, reward_list, done, _ = self.env.step(action_list)
                transition = obs_list[0].copy()
                transition["action"] = np.array(action_list[0])
                transition["reward"] = np.array(reward_list[0])
                transition["discount"] = np.array(0.0) if done else np.array(1.0)
                self.episode.append(transition)
                if done:
                    timestamp = datetime.datetime.now().strftime("%Y%m%dT%H%M%S")
                    episode = {k: [t[k] for t in self.episode] for k in self.episode[0]}
                    identifier = str(uuid.uuid4().hex)
                    length = len(episode["reward"])
                    filename = directory / f"{timestamp}-{identifier}-{length}.npz"
                    with io.BytesIO() as f1:
                        np.savez_compressed(f1, **episode)
                        f1.seek(0)
                        with filename.open("wb") as f2:
                            f2.write(f1.read())
                    # print filename
                    print(f"Episode steps: {length} saved to {filename}")

    def reset(self):
        obs_list, reward_list, done, _ = self.env.reset()
        transition = obs_list[0].copy()
        transition["reward"] = reward_list[0]
        transition["discount"] = 1.0
        self.episode = [transition]

        state_list = ()

        return obs_list, state_list, done

    def get_action(self, obs_list, state_list):
        """
        predict next state and action
        """
        action_list, state_list = self.env.predict(obs_list, state_list)
        return action_list, state_list

    def update_env(self, action_list):
        """
        update obs, reward, done
        and return updated obs, done
        """
        obs_list, reward_list, done, _ = self.env.step(action_list)
        transition = obs_list[0].copy()
        transition["action"] = np.array(action_list[0])
        transition["reward"] = np.array(reward_list[0])
        transition["discount"] = np.array(0.0) if done else np.array(1.0)
        self.episode.append(transition)

        return obs_list, done

    def save_data(self, directory='./train_dir'):

        directory = pathlib.Path(directory).expanduser()
        directory.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.datetime.now().strftime("%Y%m%dT%H%M%S")
        episode = {k: [t[k] for t in self.episode] for k in self.episode[0]}
        identifier = str(uuid.uuid4().hex)
        length = len(episode["reward"])
        filename = directory / f"{timestamp}-{identifier}-{length}.npz"
        with io.BytesIO() as f1:
            np.savez_compressed(f1, **episode)
            f1.seek(0)
            with filename.open("wb") as f2:
                f2.write(f1.read())
        # print filename
        print(f"Episode steps: {length} saved to {filename}")
        
        
if __name__ == "__main__":
    cap = cv2.VideoCapture(1)
    timestamp = datetime.datetime.now().strftime("%Y%m%dT%H%M%S")
    filename = f"./data/{timestamp}-{identifier}-{length}.npz"
    with io.BytesIO() as f1:
        np.savez_compressed(f1, **episode)
        f1.seek(0)
        with filename.open("wb") as f2:
            f2.write(f1.read())
    # print filename
    print(f"Episode steps: {length} saved to {filename}")