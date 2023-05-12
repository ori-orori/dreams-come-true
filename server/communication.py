import socket
import struct
import numpy as np

class ServerCommunication:
    def __init__(self, cfg):
        communication_cfg = cfg['communication']
        self.server_ip = communication_cfg['server_ip']
        self.server_port = communication_cfg['server_port']

        self.server2client = Server2Client(self.server_ip, self.server_port)

    def connect(self):
        """
        Connect to the client
        """
        # TODO: print for connected or not connected
        self.server2client.server_init()

    def send_action_to_client(self, robot, action):
        """
        Send action to client

        Args:
            robot: robot name, It can be 'A' or 'B'
            action: action of robot
        """
        self.server2client.send_action(robot, action)

    def recieve_info_from_client(self, info):
        """
        Send information from client

        Args:
            info: information which contains (observation, reward, done)
        """
        return self.server2client.receive_info()

    def close(self):
        """
        Close robot and server connection
        """
        self.server2client.close()
    
class Server2Client:
    def __init__(self, server_ip, port=1234):
        """
        TCP Client for communication with client
        
        Args:
            server_ip: server ip address
            port: server port number        
        """
        self.server_ip = server_ip
        self.port = port
        
    def server_init(self):
        """
        Initialize TCP server
        """
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.bind((self.server_ip, self.port))
        self.sock.listen()
    
    def send_action(self, robot, action):
        """
        Send action of specific robot type to client

        Args:
            robot: It can be 'A' or 'B'
            action: action is a list of 2 float numbers            
        """
        robot_byte = robot.encode() # Encode robot type as a single byte
        action_data = struct.pack('c2f', robot_byte, *action) # Pack robot type and action together
        self.sock.sendall(action_data)
    
    def receive_info(self):
        """
        Receive infomation which contains (observation, reward and done) from client
        """
        conn, addr = self.sock.accept()

        # Receive the bundled data
        conn, addr = self.sock.accept()

        # Receive the bundled data
        data = conn.recv(4)
        num_arrays = struct.unpack('i', data)[0]

        arrays = []
        for i in range(num_arrays):
            # Receive the shape information
            data = conn.recv(12)
            array_shape = struct.unpack('iii', data)

            # Receive the array data
            array_data = conn.recv(array_shape[0])

            # Convert the array data to numpy array
            array = np.frombuffer(array_data, dtype=np.float32).reshape(array_shape[1:])
            arrays.append(array)

        cropped_frame, ball_position, robot_info, reward, done = arrays

        return (cropped_frame, ball_position, robot_info), reward, done
    
    def close(self):
        """
        Close server connection
        """
        self.sock.close()