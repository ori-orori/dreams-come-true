# -*- coding: utf-8 -*-
"""
Created on Wed Mar 29 11:39:36 2023

@author: jellyho
"""

import socket
import signal
import sys
import time
import select

class RCServer:
    def __init__(self, carA_ip='192.168.0.201', carB_ip='192.168.0.202', port=1234):
        self.carA_ip = carA_ip
        self.carB_ip = carB_ip
        self.port = port
        self.ServerInit()
        self.bufferSize = 1024
        signal.signal(signal.SIGINT, self.socket_unbinder)

    def socket_unbinder(self, sig, fname):
        self.sock.close()
        print('Socket Released')
        sys.exit(0)
        
    def ServerInit(self):
        # Create a UDP socket
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        host = socket.gethostname()
        host_ip = socket.gethostbyname(host)
        self.sock.bind((host_ip, 1234))
        self.sock.setblocking(0)
        print('RC Server Initialized')
        print('--------------------------------')
        print(f'HOST : {host_ip}:{self.port}')
        print(f'CarA : {self.carA_ip}:{self.port}')
        print(f'CarB : {self.carB_ip}:{self.port}')
        print('--------------------------------')

    def Act(self, robot='A', act=(0, 0)):
        start_time = time.time()
        speed = act
        message = f'{int(speed[0]*256)}&{int(speed[1]*256)}'
        message = message.encode()
        self.sock.sendto(message, (self.carA_ip, self.port))
        ready = select.select([self.sock], [], [], 0.05)
        data, addr = b'', b''
        if ready[0]:
            data, addr = self.sock.recvfrom(self.bufferSize)
        # 데이터 수신 완료 시각 기록
        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"{data.decode('ascii')} >> elapsed:{elapsed_time*1000:.4f}ms")
        
        return elapsed_time

    def HeuristicAct(self, robot='A', act='W', v=1):
        speed = (0, 0)
        if act == 'W':
            speed = (v, v)
        elif act == 'S':
            speed = (-v, -v)
        elif act == 'A':
            speed = (-v, v)
        elif act == 'D':
            speed = (v, -v)
        else:
            pass
        return self.Act(robot, act=speed)
