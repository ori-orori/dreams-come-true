import cv2
import sys
import time

def find_cameras(max_cameras=10):
    cameras = []
    for i in range(max_cameras):
        try:
            cap = cv2.VideoCapture(i)
            if cap is None or not cap.isOpened():
                cap.release()
            else:
                cameras.append(i)
                cap.release()
        except Exception as e:
            print(f"Error while trying camera {i}: {e}")
            continue
    return cameras

def main():
    # List available cameras
    # cameras = find_cameras()
    # if not cameras:
    #     print("No cameras found.")
    #     sys.exit()

    # # Display the list of cameras
    # print("Available cameras:")
    # for i, camera_index in enumerate(cameras):
    #     print(f"{i}: Camera {camera_index}")

    # # Prompt the user to select a camera
    # camera_choice = int(input("Enter the index of the camera you want to use: "))
    # camera_index = cameras[camera_choice]
    camera_index = 1

    # Open the selected camera
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        print("Error: Could not open the selected camera.")
        sys.exit()

    # Set the desired FPS
    desired_fps = 30
    cap.set(cv2.CAP_PROP_FPS, desired_fps)
    # Calculate the time delay between frames based on the desired FPS
    frame_delay = 1.0 / desired_fps

    # Get and display the camera's frame size and resolution
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"Camera frame size: {frame_width} x {frame_height}")

    # Create a window to display the webcam feed
    cv2.namedWindow("Webcam Feed", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Webcam Feed", frame_width, frame_height)

    frame_count = 0
    start_time = time.time()
    prev_frame_time = start_time

    while True:
        # Capture a frame from the selected camera
        ret, frame = cap.read()

        if not ret:
            print("Error: Could not read a frame from the camera.")
            break

        # Display the frame in the created window
        cv2.imshow("Webcam Feed", frame)

        # Calculate and display the actual FPS
        frame_count += 1
        elapsed_time = time.time() - start_time
        actual_fps = frame_count / elapsed_time
        print(f"Actual FPS: {actual_fps:.2f}", end='\r')

        # Wait for the desired time delay between frames
        current_time = time.time()
        time_diff = current_time - prev_frame_time
        if time_diff < frame_delay:
            time.sleep(frame_delay - time_diff)
        prev_frame_time = current_time

        # Exit the loop if the 'q' key is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the camera and close the window
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
