import os
import cv2
import numpy as np
import torch
import pickle
import csv

# https://docs.opencv.org/4.x/dc/dbb/tutorial_py_calibration.html
# Testing object detection with YOLOv5: https://github.com/ultralytics/yolov5

class DistortionCorrection():
    """
    Camera Calibration & Lens Distortion Correction.
    """

    def __init__(self, SR=1):
        self.SR = SR


    def create_manual_data(self, source, export_path):
        cap = cv2.VideoCapture(source)
        image_count = 0

        while cap.isOpened():
            ret, img = cap.read()
            cv2.imshow('Save image by pressing \'s\' key. Press \'q\' to quit.', img)

            key = cv2.waitKey(1) & 0xFF  # Wait for key press (1 ms delay)
            if key == ord('s'): # Press 's' to save the image
                cv2.imwrite(f'{export_path}/calibration_image_{image_count}.png', img)
                print(f"Image {image_count} saved!")
                image_count += 1
            elif key == ord('q'): # Press 'q' to quit
                break

        # Release and destroy all windows before termination
        cap.release()
        cv2.destroyAllWindows()


    def create_calibration_data(self, import_path, export_path):
        """
        Create calibration images from a video.
        """
        # Open the camera
        cap = cv2.VideoCapture(import_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        sample_rate = int(fps/self.SR) # sample x times every seconds
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        num_images = int(total_frames / sample_rate)

        # Check if the folder exists
        if not os.path.exists(export_path):
            # If it doesn't exist, create it recursively
            os.makedirs(export_path)

        frame_count = 0
        image_count = 0

        while True:
            # Capture frame-by-frame
            ret, frame = cap.read()
            frame_count += 1

            # Check if the frame was successfully captured
            if not ret:
                break
            
            if frame_count % sample_rate == 0:
                # Save frame image
                cv2.imwrite(f'{export_path}/calibration_image_{image_count}.png', frame)
                print(f'Data creation: Image {image_count+1}/{num_images}.')
                image_count += 1

        # Release the camera and close all OpenCV windows
        cap.release()
        cv2.destroyAllWindows()


    def calibrate_camera(self, path, board_size=(7,7), square_size_mm=26):
        """
        Camera calibration: Get camera matrix and distortion coefficients using calibration images.
        """
        # Define the number of inner corners in the chessboard pattern
        board_size = board_size  # (columns-1, rows-1)
        square_size_mm = square_size_mm # size of one chessboard square in mm

        # Prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(8,5,0)
        objp = np.zeros((board_size[0] * board_size[1], 3), np.float32)
        objp[:, :2] = np.mgrid[0:board_size[0], 0:board_size[1]].T.reshape(-1, 2)
        objp = objp * square_size_mm

        # Arrays to store object points and image points from all the images
        obj_points = []  # 3D points in real world space
        img_points = []  # 2D points in image plane

        # Read all images in the data folder
        images = [f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f)) and (f.endswith('.jpg') or f.endswith('.png'))]
        num_images = len(images)

        # Read calibration images and detect chessboard corners
        for i in range(num_images):
            print(f'Calibration: Image {i+1}/{num_images}.')
            img = cv2.imread(f'{path}/calibration_image_{i}.png')
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            # Find chessboard corners
            ret, corners = cv2.findChessboardCorners(gray, board_size, None)
            
            # If corners are found, add object points and image points
            if ret:
                obj_points.append(objp)
                img_points.append(corners)
                # Draw and display the corners (Debug visualization)
                #cv2.drawChessboardCorners(img, board_size, corners, ret)
                #cv2.imshow('Chessboard Corners', img)
                #cv2.waitKey(500) # Display each image for 500 ms      
        #cv2.destroyAllWindows()

        # Calibrate the camera
        ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(obj_points, img_points, gray.shape[::-1], None, None)

        # Save calibration data
        calibration_data = {
            'camera_matrix': camera_matrix,
            'dist_coeffs': dist_coeffs
        }
        pickle.dump(calibration_data, open("calibration.pkl", "wb"))

        # Print reprojection error of calibration process
        mean_error = self.calculate_error(camera_matrix, dist_coeffs, obj_points, img_points, rvecs, tvecs)
        print(f"Total error: {mean_error}")
        
        return camera_matrix, dist_coeffs
    

    def calculate_error(self, camera_matrix, dist_coeffs, obj_points, img_points, rvecs, tvecs):
        """
        Calculate the reprojection error of the camera calibration.
        """
        cumulative_error = 0
        for i in range(len(obj_points)):
            imgpoints2, _ = cv2.projectPoints(obj_points[i], rvecs[i], tvecs[i], camera_matrix, dist_coeffs)
            error = cv2.norm(img_points[i], imgpoints2, cv2.NORM_L2)/len(imgpoints2)
            cumulative_error += error
        mean_error = cumulative_error / len(obj_points)

        return mean_error
    

    def undistort_image(self, image, camera_matrix, dist_coeffs):
        """
        Perform lens distortion correction on an image.
        """
        h, w = image.shape[:2]
        new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(camera_matrix, dist_coeffs, (w, h), 0, (w, h))
        undistorted_img = cv2.undistort(image, camera_matrix, dist_coeffs, None, new_camera_matrix)
        # image cropping
        x, y, w, h = roi
        undistorted_img = undistorted_img[y:y+h, x:x+w]

        return undistorted_img
    

    def export_data(self, img, results, classnames, writer, frame_idx, detections):
        # Draw bounding boxes and labels on the frame
        for *xyxy, conf, cls in results.xyxy[0]:
            label = f'{classnames[int(cls)]} {conf:.2f}'
            cv2.rectangle(img, (int(xyxy[0]), int(xyxy[1])), (int(xyxy[2]), int(xyxy[3])), (0, 255, 0), 2)
            cv2.putText(img, label, (int(xyxy[0]), int(xyxy[1]) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            detections.append([frame_idx, int(cls), classnames[int(cls)], conf.item(), int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3])]) # CSV

        # write image to video
        writer.write(img)
        return detections



def main(recreate=False, manual=False, recalibrate=False):
    """
    1. Create Calibration data ('recreate') either manually ('manual', requires webcam and a printed out 8x8 chessboard image) or use existing data.
    2. Calibrate the camera ('recalibrate') and optionally perform Lens Distortion Correction.
    3. Test the LDC on existing test data (e.g. object detection) and export BB visualization + csv.
    """
    ### Instantiate Correcter
    sample_rate = 1
    correcter = DistortionCorrection(SR=sample_rate)
    calibration_video = "./data/calibration/calib5.mov"
    calibration_images = f"./data/calibration/sr0_calib0" if manual else f"./data/calibration/sr{sample_rate}_{calibration_video[-10:-4]}"

    # MEASUREMENTS
    # calibration: calib2, SR=3, error: 15425056744370919, testing: 2_lights_both, detections: 50.0% (hands)
    # calibration: manual, SR=0, error: 0.10335521807128667, testing: wohnzimmer_insta, detections: 99.8% (objects)

    ### Create calibration data (only once)
    if recreate:
        if manual:
            correcter.create_manual_data(source=0, export_path=calibration_images)
        else:
            correcter.create_calibration_data(calibration_video, calibration_images)

    ### Calibrate camera with calibration data (only once)
    if recalibrate:
        camera_matrix, dist_coeffs = correcter.calibrate_camera(calibration_images, (7,7), 26)
    else:
        # Load calibration data from pickle file (last calibration)
        calibration_data = pickle.load(open("calibration.pkl", "rb"))
        camera_matrix = calibration_data['camera_matrix']
        dist_coeffs = calibration_data['dist_coeffs']

    # Load data
    video = 'highflow.MP4'
    test_video_path = f'./data/test/frame/{video}'
    cap = cv2.VideoCapture(test_video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Load model
    path = '../yolov5/hand.pt'
    #path = '../yolov5s.pt'
    hand_detector = torch.hub.load('ultralytics/yolov5', 'custom', path=path)
    hand_detector.eval()
    classnames = hand_detector.names

    # Export data
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    #width, height = 640, 640
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    model = 'hands' if 'hand' in path else 'objects'
    output_video_path = f'./data/outputs/{video[:-4]}/{model}.mp4'
    os.makedirs(os.path.dirname(output_video_path), exist_ok=True)
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
    output_csv_path = f'./data/outputs/{video[:-4]}/{model}.csv'
    os.makedirs(os.path.dirname(output_csv_path), exist_ok=True)
    detections = []

    # Helper variables
    det_counter = 0 # count bounding box detections
    frame_count = 0
    undistort = False # whether to use undistorted frames

    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()
        frame_count += 1

        # Check if the frame was successfully captured
        if not ret:
            break

        # Undistort the frame
        try:
            if camera_matrix is None or dist_coeffs is None:
                raise ValueError(f"Camera matrix is {camera_matrix}) and distortion coefficients is {dist_coeffs}. Please calibrate first.")
            
            # Undistort frame and get BB predictions
            frame = correcter.undistort_image(frame, camera_matrix, dist_coeffs) if undistort else frame
            #frame = cv2.resize(frame, (640, 640)) # resizing to match training data image size does not seem to work (probably already resized pre-inference)
            results = hand_detector(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            detections = correcter.export_data(frame, results, classnames, out, frame_count, detections)

            # accumulate number of frames in which hand(s) were detected as performance measure
            if results.pred[0].size()[0] > 0:
                det_counter += 1
            results.print()
            print(f"Frame {frame_count}/{total_frames}.\n")

            # Prediction visualization: Display the annotated image with bounding boxes
            img = results.render()
            img = np.array(img)
            img = np.squeeze(img, axis=0) # remove extra dimension for displaying
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            cv2.imshow('Hand Detection', img)

            # Introduce a delay and check for 'q' key press to exit
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        except ValueError as e:
            print("Error:", e)
            break

    # Release the camera and close all OpenCV windows
    print(f"\nNumber of detections is {det_counter}/{frame_count} ({round(det_counter/frame_count, 3) * 100}%).")
    cap.release()
    out.release()
    cv2.destroyAllWindows()

    # Write detections to CSV
    with open(output_csv_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Frame', 'ClassID', 'ClassName', 'Confidence', 'X1', 'Y1', 'X2', 'Y2'])
        writer.writerows(detections)


# Use main idiom for importing class to detection script of object detector.
if __name__ == "__main__":
    main(recreate=False, manual=True, recalibrate=False)