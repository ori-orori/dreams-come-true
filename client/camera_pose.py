### Thanks to https://python-academia.com/en/opencv-aruco/

import cv2
from cv2 import aruco
import time
import numpy as np

dict_aruco = aruco.Dictionary_get(aruco.DICT_5X5_100)
parameters = aruco.DetectorParameters_create()

cap = cv2.VideoCapture(0)

try:
    while True:
        ret, frame = cap.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)

        corners, ids, rejectedImgPoints = aruco.detectMarkers(gray, dict_aruco, parameters=parameters)

        frame_markers = aruco.drawDetectedMarkers(frame.copy(), corners, ids)
        
        num_id = 66 # [70 42 24 66 87]
        if num_id in np.ravel(ids) :
            index = np.where(ids == num_id)[0][0] #Extract index of num_id
            cornerUL = corners[index][0][0]
            cornerUR = corners[index][0][1]
            cornerBR = corners[index][0][2]
            cornerBL = corners[index][0][3]

            center = [ (cornerUL[0]+cornerBR[0])/2 , (cornerUL[1]+cornerBR[1])/2 ]

            # print('Upper left : {}'.format(cornerUL))
            # print('Upper right : {}'.format(cornerUR))
            # print('Lower right : {}'.format(cornerBR))
            # print('Lower Left : {}'.format(cornerBL))
            # print('Center : {}'.format(center))
            # print(corners[index])
            # frame_markers_C = frame_markers.copy()
            
            text='[' + str(center[0]) + ','+ str(center[1])  + ']' #"Hello OpenCV!(한글)"
            org=(50, 100) # (int(center[0]), int(center[1]))
            arrow_i = (int(center[0]), int(center[1]))
            arrow_f = (int(cornerUR[0]), int(cornerUR[1]))
            font=cv2.FONT_HERSHEY_SIMPLEX
            frame_markers_C = cv2.putText(frame_markers.copy(),text,org,font,1,(255,0,0),2)
            frame_markers_C = cv2.arrowedLine(frame_markers_C.copy(), arrow_i, arrow_f, (255,255,0), 2)
            # frame_markers_c = aruco.drawDetectedMarkers(frame_markers.copy(), corners, ids)
            # return center
        else:
            frame_markers_C = frame_markers.copy()
        cv2.imshow('frame', frame_markers_C)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cv2.destroyWindow('frame')
    cap.release()
except KeyboardInterrupt:
    cv2.destroyWindow('frame')
    cap.release()
