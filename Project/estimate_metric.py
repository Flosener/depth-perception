"""
MIT License

Copyright (c) 2022 Intelligent Systems Lab Org

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

Citation:
@misc{https://doi.org/10.48550/arxiv.2302.12288,
  doi = {10.48550/ARXIV.2302.12288},
  url = {https://arxiv.org/abs/2302.12288},
  author = {Bhat, Shariq Farooq and Birkl, Reiner and Wofk, Diana and Wonka, Peter and Müller, Matthias},
  keywords = {Computer Vision and Pattern Recognition (cs.CV), FOS: Computer and information sciences, FOS: Computer and information sciences},
  title = {ZoeDepth: Zero-shot Transfer by Combining Relative and Metric Depth},
  publisher = {arXiv},
  year = {2023},
  copyright = {arXiv.org perpetual, non-exclusive license}
}

GitHub: https://github.com/isl-org/MiDaS, https://github.com/isl-org/ZoeDepth
"""

classes = {
    0: 'bottle',
    1: 'bowl_close',
    2: 'bowl_far',
    3: 'clock',
    4: 'cup_close',
    5: 'cup_far',
    6: 'hand_close',
    7: 'hand_far',
    8: 'hand_medium',
    9: 'plant',
    10: 'glass_close',
    11: 'glass_far'
}

# region Imports
import sys
from pathlib import Path

# Use the project file packages instead of the conda packages, i.e. add to system path for import
file = Path(__file__).resolve()
root = file.parents[0]
modules = ['yolov5', 'MiDaS']
for m in modules:
    path = root / m
    if path.exists() and str(path) not in sys.path:
        sys.path.append(str(path))
        #print(f"Added {path} to sys.path")
    elif not path.exists():
        print(f"Error: {path} does not exist.")
    elif str(path) in sys.path:
        print(f"{path} already exists in sys.path")

import cv2
import numpy as np
import os
import csv
import torch
import open3d as o3d
from PIL import Image
from MiDaS.run import create_side_by_side
from MiDaS.ZoeDepth.zoedepth.utils.misc import colorize
# endregion

class DepthEstimator:
    def __init__(self, model_type, device=None):
        self.device = self.get_device(device)
        self.model_type = model_type
        self.model = self.load_model()
    
    def get_device(self, device):
        if device in ['cpu', 'cuda', 'mps']:
            if device == 'cuda' and torch.cuda.is_available():
                return torch.device("cuda")
            if device == 'mps' and torch.backends.mps.is_available():
                return torch.device("mps")
        return torch.device("cpu")
    
    def load_model(self):
        model = torch.hub.load("isl-org/ZoeDepth", self.model_type, pretrained=True)
        model.to(self.device)
        return model
    
    def preprocess(self, image):
        image = Image.open(image).convert("RGB")
        return image
    
    def predict_depth(self, image):
        self.model.eval()
        input = self.preprocess(image)
        with torch.no_grad():
            depth = self.model.infer_pil(input)
        return depth

    def visualize_depth(self, image, depth, grayscale):
        return create_side_by_side(image, depth, grayscale) / 255
        
    def create_point_cloud(self, image, depth_map, name, outdir):
        """
        Code by @ Subhransu Sekhar Bhattacharjee (Rudra) "1ssb"
        """
        height, width = image.shape[:2]
        focal_length_x = 470.4 # adjust according to camera
        focal_length_y = 470.4 # adjust according to camera

        x, y = np.meshgrid(np.arange(width), np.arange(height))
        x = (x - width / 2) / focal_length_x
        y = (y - height / 2) / focal_length_y
        z = np.array(depth_map)

        points = np.stack((np.multiply(x, z), np.multiply(y, z), z), axis=-1).reshape(-1, 3)
        colors = image.reshape(-1, 3) / 255.0

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        pcd.colors = o3d.utility.Vector3dVector(colors)

        out_path = os.path.join(outdir, f'{name}_point_cloud.ply')
        o3d.io.write_point_cloud(out_path, pcd)
        print(f'Point cloud saved to {out_path}')

    def target_depth(self, label_file, depth_map):
        """
        Extract depth from a target ROI (e.g. bounding box).

        Parameters:
        label_file (str): Path to the YOLO format label file
        depth_map (np.array): The depth map of the image

        Returns:
        float: The average error between the predicted mean depth and the true depth across all bounding boxes
        """
        height, width = depth_map.shape[:2] # if depth_map.shape == frame.shape[:2]
        total_error = 0
        count = 0

        # Read YOLO labels from the file
        with open(label_file, 'r') as f:
            lines = f.readlines()

        # Store results for CSV output
        results = []

        for line in lines:
            parts = line.strip().split()
            # YOLO format: class x_center y_center width height (all normalized 0-1)
            # Here, we assume the true depth is also given in the label (for demonstration purposes)
            class_id, x_center, y_center, bbox_width, bbox_height, true_depth = map(float, parts)
            
            # Convert normalized coordinates to absolute pixel values
            x_center *= width
            y_center *= height
            bbox_width *= width
            bbox_height *= height

            x = int(x_center - bbox_width / 2)
            y = int(y_center - bbox_height / 2)
            w = int(bbox_width)
            h = int(bbox_height)

            # Ensure the bounding box coordinates are within the image dimensions
            x_start = max(x, 0)
            y_start = max(y, 0)
            x_end = min(x + w, depth_map.shape[1])
            y_end = min(y + h, depth_map.shape[0])

            # Extract the ROI from the depth map
            roi_depth = depth_map[y_start:y_end, x_start:x_end]

            # Calculate the mean depth within the ROI
            mean_depth = np.mean(roi_depth)

            # Calculate the absolute difference between the mean depth and the true depth
            depth_difference = abs(mean_depth - true_depth)

            # Accumulate the error and count
            total_error += depth_difference
            count += 1

            # Print the mean depth and compare it with the true depth
            print(f"{classes[class_id]}: (x: {x}, y: {y}, w: {w}, h: {h})")
            print(f"True Depth: {true_depth}")
            print(f"Mean Depth: {mean_depth}")
            print(f"Depth Difference: {depth_difference}\n")

            # Store result for CSV output
            results.append([os.path.splitext(os.path.basename(label_file[:-44]))[0], classes[class_id], mean_depth, true_depth])

        # Calculate the average error
        average_error = total_error / count if count != 0 else 0

        # Print and return the average error
        print(f"Average Error: {average_error}")
        return average_error, results


def main():

    # Load OD models
    weights_objects = './yolov5/yolov5x.pt'
    weights_hands = './yolov5/hand.pt'
    od_model_objs = torch.hub.load('ultralytics/yolov5', 'custom', path=weights_objects).eval()
    od_model_hands = torch.hub.load('ultralytics/yolov5', 'custom', path=weights_hands).eval()

    # Load DE model
    depth_estimator = DepthEstimator(
        model_type = 'ZoeD_N', # ZoeN (nyu, indoor), ZoeK (kitti, outdoor), ZoeNK
        device='cpu'
    )

    # Initialize CSV file
    csv_file = 'estimated_depths.csv'
    with open(csv_file, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['filename', 'object', 'estimated_depth', 'true_depth'])

    # Load DS
    ds = './datasets/HaND_augmented/'
    images = os.listdir(ds+'images/')
    labeldir = ds+'labels/'
    outdir = ds+'visualization/'
    n = len(images)
    
    # Loop over testing data
    for i, file in enumerate(images):

        # Read the image
        frame = cv2.imread(ds + 'images/' + file)
        
        if frame is None:
            print(f"Failed to load image: {file}")
            continue

        # Perform object detection
        #rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        #detections_objs = od_model_objs(frame)
        #detections_hands = od_model_hands(frame)
        # combine detections
        #detections = 

        # Perform depth estimation
        depth = depth_estimator.predict_depth(ds + 'images/' + file)
        average_error, results = depth_estimator.target_depth(labeldir + file[:-3] + 'txt', depth)

        # Append results to CSV
        with open(csv_file, mode='a', newline='') as f:
            writer = csv.writer(f)
            writer.writerows(results)

        visual = depth_estimator.visualize_depth(frame, depth, False)
        print(f'\nDepth Image {i+1}/{n}, Min = {depth.min()}, Max = {depth.max()}')
        #depth_estimator.create_point_cloud(frame, depth, file[:-40], outdir)
        cv2.imshow("Depth map", visual)
        if cv2.waitKey(100) & 0xFF == ord('q'):
            break


if __name__ == "__main__":
    main()