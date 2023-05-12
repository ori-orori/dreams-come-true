import socket
import struct
import numpy as np

class ClientCommunication:
    def __init__(self, cfg):
        communication_cfg = cfg.communication
        self.local_ip = communication_cfg['local_ip']
        self.server_ip = communication_cfg['server_ip']
        self.carA_ip = communication_cfg['carA_ip']
        self.carB_ip = communication_cfg['carB_ip']
        self.client_port = communication_cfg['client_port']
        self.server_port = communication_cfg['server_port']

        self.client2server = Client2Server(self.local_ip, self.server_ip, self.server_port)
        self.client2car = Client2Car(self.local_ip, self.carA_ip, self.carB_ip, self.client_port)

    def connect(self):
        """
        Connect to the car and server
        """
        # TODO: print for connected or not connected
        self.client2car.server_init()
        # self.client2server.server_init()

    def send_action_to_robot(self, robot, action):
        """
        Send action to robot

        Args:
            robot: robot name, It can be 'A' or 'B'
            action: action, action is a list of 2 float numbers (left wheel speed, right wheel speed)
        """
        self.client2car.send_action(robot, action)

    def send_info_to_server(self, info):
        """
        Send information to server

        Args:
            info: information which contains (observation, reward, done)
        """
        self.client2server.send_info(info)

    def receive_action_from_server(self):
        """
        Receive action from server
        """
        return self.client2server.receive_action()

    def close(self):
        """
        Close robot and server connection
        """
        self.client2server.close()
        self.client2car.close()



class Client2Car():
    def __init__(self, local_ip='127.0.0.1', car_a_ip='192.168.0.201', car_b_ip='192.168.0.202', port=1234):
        """
        UDP Client for communication with car
        
        Args:
            local_ip: local ip address
            car_a_ip: car A ip address
            car_b_ip: car B ip address
            port: client port number        
        """
        self.local_ip = local_ip
        self.carA_ip = car_a_ip
        self.carB_ip = car_b_ip
        self.port = port
        self.bufferSize = 1024

    def server_init(self):
        """
        Initialize UDP server
        """
        # Create a UDP socket
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        # self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.sock.bind((self.local_ip, self.port))
        self.sock.setblocking(True)
        print('RC Server Initialized')
        print('--------------------------------')
        print(f'local ip : {self.local_ip}')
        print(f'carA ip : {self.carA_ip}')
        print(f'carB ip : {self.carB_ip}')
        print('--------------------------------')

    def send_action(self, robot='A', act=0):
        """
        Send action to robot

        Args:
            robot: robot name, It can be 'A' or 'B'
            act: action (left wheel speed, right wheel speed)
        """
        
        speed = act
        message = str(f'{speed[0]}&{speed[1]}').encode()
        if robot == 'A':
            self.sock.sendto(message, (self.carA_ip, self.port))
        else:
            self.sock.sendto()

    def close(self):
        """
        Close robot connection
        """
        self.sock.close()

class Client2Server:
    def __init__(self, local_ip, server_ip, server_port=1234):
        """
        TCP Client for communication with robot
        
        Args:
            local_ip: local ip address
            server_ip: server ip address
            port: server port number        
        """
        self.local_ip = local_ip
        self.server_ip = server_ip
        self.server_port = server_port
        
    def server_init(self):
        """
        Initialize TCP server
        """
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.connect((self.server_ip, self.server_port))
    
    def send_info(self, info):
        """
        Send observation, reward, done to server

        Args:
            info: infomation which contains (observation, reward, done)
        """
        observation, reward, done = info # reward shape [1], done shape [1]
        cropped_frame, ball_position, robot_info = observation # cropped_frame shape [3, 64, 64], ball_position shape [2], robot_info shape [2]
        arrays = [[cropped_frame, ball_position, robot_info], reward, done]

        packed_data = b''
        packed_data += struct.pack('i', len(arrays))
        for array in arrays:
            packed_data += struct.pack('iii', array.nbytes, *array.shape)
            packed_data += array.tobytes()

        self.sock.sendall(packed_data)
    
    def receive_action(self):
        """
        Receive action from server

        Returns: action (left wheel speed, right wheel speed)
        """
        data = self.sock.recv(9) # Receive 9 bytes (1 byte for robot and 8 bytes for action)
        robot_byte, action = struct.unpack('c2f', data) # Unpack the robot type and action
        robot = robot_byte.decode() # Decode the robot type back to a string
        return robot, action
    
    def close(self):
        """
        Close server connection
        """
        self.sock.close()