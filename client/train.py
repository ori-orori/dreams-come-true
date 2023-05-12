# -*- coding: utf-8 -*-
"""
Created on Tue May  9 14:28:58 2023

@author: jellyho
"""
import time

import argparse
from communication  import ClientCommunication
from camera import CameraModule
from env.soccer_env import SoccerEnv
import datetime
import pathlib
import numpy as np
import io
import uuid
import cv2
import yaml

from client_dreamer import Client_Dreamer
from tools import Logger
from client import Client
import tools

from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
import time, sys
from RCServer import RCServer

# do not modify
app = QApplication(sys.argv)
rc = RCServer()

parser = argparse.ArgumentParser()
parser.add_argument("--configs", nargs="+", required=True)
args, remaining = parser.parse_known_args()
configs = yaml.safe_load(
    (pathlib.Path(sys.argv[0]).parent / "config.yaml").read_text()
)
defaults = {}
for name in args.configs:
    defaults.update(configs[name])
parser = argparse.ArgumentParser()
for key, value in sorted(defaults.items(), key=lambda x: x[0]):
    arg_type = tools.args_type(value)
    parser.add_argument(f"--{key}", type=arg_type, default=arg_type(value))
cfg = parser.parse_args(remaining)

####
class MainWorker(QObject):    
    @pyqtSlot()
    def main(self):
        client = Client(cfg)
        obs_list, state_list, done = client.reset()
        step = 0
        while step < cfg.max_steps:
            action_list, state_list = client.get_action(obs_list, state_list)
            action1, action2 = action_list
            rc.worker.Act('A', action1)
            rc.worker.Act('B', action2)
            time.sleep(1)
            obs_list, done = client.update_env(action_list)
            print(1)
            if done:
                client.save_data()
                break
            step += 1
        
        return
        
        
        # main training Code Here :)
        # while True:
        #     while rc.available():
        #         rc.worker.Act('A', 'W')
        #         time.sleep(1)
        #         rc.worker.Act('A', 'S')
        #         time.sleep(1)
        #     print('Robots are not available')
        #     time.sleep(1)
        return
 ####   
    
# do not modifiy
main = MainWorker()
rc.setMain(main)
rc.show()
sys.exit(app.exec_())