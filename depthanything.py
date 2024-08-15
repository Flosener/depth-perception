"""
Copyright 2024 Lihe Yang

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.

Citation:
@article{depth_anything_v2,
  title={Depth Anything V2},
  author={Yang, Lihe and Kang, Bingyi and Huang, Zilong and Zhao, Zhen and Xu, Xiaogang and Feng, Jiashi and Zhao, Hengshuang},
  journal={arXiv:2406.09414},
  year={2024}
}

GitHub: https://github.com/DepthAnything/Depth-Anything-V2
"""

import cv2
import matplotlib
import numpy as np
import os
import torch
import open3d as o3d
import time
import sys
from pathlib import Path

# Add project files to PATH before importing DAv2
file = Path(__file__).resolve()
root = file.parents[0]
modules = ['depthanythingv2']
for m in modules:
    path = root / m
    if path.exists() and str(path) not in sys.path:
        sys.path.append(str(path))
    elif not path.exists():
        print(f"Error: {path} does not exist.")

from depthanythingv2.metric_depth.depth_anything_v2.dpt import DepthAnythingV2 as metricV2
from depthanythingv2.depth_anything_v2.dpt import DepthAnythingV2 as relativeV2

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


class DepthAnythingEstimator:
    def __init__(self, model_type='vits', dataset='hypersim', max_depth=None, device=None):
        self.device = device if device is not None else torch.device('cuda') if torch.cuda.is_available() else torch.device('mps') if torch.backends.mps.is_available() else torch.device('cpu')
        print(f'Device: {self.device}')
        self.encoder = model_type if model_type in ['vits', 'vitb', 'vitl', 'vitg'] else 'vits'
        self.configs = {
            'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
            'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
            'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
            'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
        }
        self.dataset = dataset
        self.max_depth = max_depth # 20 for indoor model, 80 for outdoor model
        self.model = self.load_model()
    
    def load_model(self):
        if self.max_depth is not None:
            model = metricV2(**{**self.configs[self.encoder], 'max_depth': self.max_depth})
            model.load_state_dict(torch.load(f'depthanythingv2/checkpoints/depth_anything_v2_metric_{self.dataset}_{self.encoder}.pth', map_location=self.device))
        else:
            model = relativeV2(**self.configs[self.encoder])
            model.load_state_dict(torch.load(f'depthanythingv2/checkpoints/depth_anything_{self.encoder}14.pth', map_location=self.device))
        model.to(self.device) # already done by map_location, but well
        return model
    
    def preprocess(self, image):
        # pre-processing already done in infer_image()
        return image
    
    def predict_depth(self, image):
        self.model.eval()
        input = self.preprocess(image)
        with torch.no_grad():
            start = time.time()
            depth = self.model.infer_image(input, input_size=518)
            end = time.time()
        inference_time = end - start
        return depth, inference_time

    def create_depthmap(self, image, depth, grayscale, name=None, outdir=None):
        # Normalize depth map
        depth_normalized = (depth - depth.min()) / (depth.max() - depth.min()) * 255.0
        depth_normalized = depth_normalized.astype(np.uint8)

        if grayscale:
            depth_colored = np.repeat(depth_normalized[..., np.newaxis], 3, axis=-1)
        else:
            cmap = matplotlib.colormaps.get_cmap('Spectral_r')
            depth_colored = (cmap(depth_normalized)[:, :, :3] * 255)[:, :, ::-1].astype(np.uint8)

        if depth_colored is None:
            return image
        else:
            split_region = np.ones((image.shape[0], 20, 3), dtype=np.uint8) * 255
            combined_frame = cv2.hconcat([image, split_region, depth_colored])
            # Save to output directory if specified
            if outdir is not None and name is not None:
                if not os.path.exists(outdir):
                    os.makedirs(outdir)
                output_path = os.path.join(outdir, name+'.png')
                cv2.imwrite(output_path, combined_frame)
                #print(f'Saved depth visualization to {output_path}')
            return combined_frame
        
    def create_pointcloud(self, image, depth, name=None, outdir=None):
        """
        Code by @ Subhransu Sekhar Bhattacharjee (Rudra) "1ssb"
        """
        height, width = image.shape[:2]
        focal_length_x = 470.4 # adjust according to camera
        focal_length_y = 470.4 # adjust according to camera

        x, y = np.meshgrid(np.arange(width), np.arange(height))
        x = (x - width / 2) / focal_length_x
        y = (y - height / 2) / focal_length_y
        z = np.array(depth)

        points = np.stack((np.multiply(x, z), np.multiply(y, z), z), axis=-1).reshape(-1, 3)
        colors = image.reshape(-1, 3) / 255.0

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        pcd.colors = o3d.utility.Vector3dVector(colors)
        # Plot point cloud
        #o3d.visualization.draw_geometries([pcd])

        if outdir is not None and name is not None:
            if not os.path.exists(outdir):
                os.makedirs(outdir)
            out_path = os.path.join(outdir, name+'.ply')
            o3d.io.write_point_cloud(out_path, pcd)
            #print(f'Point cloud saved to {out_path}')

    def create_csv(self, label_file, depth, time):
        """
        Extract depth from a target ROI (e.g. bounding box).

        Parameters:
        label_file (str): Path to the YOLO format label file
        depth (np.array): The depth map of the image

        Returns:
        float: The average error between the predicted mean depth and the true depth across all bounding boxes
        """
        height, width = depth.shape[:2] # if depth.shape == frame.shape[:2]
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
            x_end = min(x + w, depth.shape[1])
            y_end = min(y + h, depth.shape[0])

            # Extract the ROI from the depth map
            roi_depth = depth[y_start:y_end, x_start:x_end]

            # Calculate the mean depth within the ROI (method should probably be changed later)
            mean_depth = np.mean(roi_depth)

            # Calculate the absolute difference between the mean depth and the true depth
            depth_difference = abs(mean_depth - true_depth)

            # Accumulate the error and count
            total_error += depth_difference
            count += 1

            # Print the mean depth and compare it with the true depth
            #print(f"{classes[class_id]}: (x: {x}, y: {y}, w: {w}, h: {h})")
            #print(f"True Depth: {true_depth}")
            #print(f"Mean Depth: {mean_depth}")
            #print(f"Depth Difference: {depth_difference}\n")

            # Store result for CSV output
            id = '_' + label_file[-36:-33] # take 3 letters as ID hash for each image
            results.append([os.path.basename(label_file[:-44]) + id, classes[class_id], mean_depth, true_depth, time])

        # Calculate the average error
        average_error = total_error / count if count != 0 else 0

        # Print the average error and return results
        #print(f"Average Error: {average_error}")
        return results
