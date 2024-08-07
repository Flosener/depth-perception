"""
BSD 2-Clause License

Copyright (c) 2024, Wei Yin and Mu Hu

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this
   list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

Citation:
@article{hu2024metric3dv2,
  title={Metric3D v2: A Versatile Monocular Geometric Foundation Model for Zero-shot Metric Depth and Surface Normal Estimation},
  author={Hu, Mu and Yin, Wei and Zhang, Chi and Cai, Zhipeng and Long, Xiaoxiao and Chen, Hao and Wang, Kaixuan and Yu, Gang and Shen, Chunhua and Shen, Shaojie},
  journal={arXiv preprint arXiv:2404.15506},
  year={2024}
}

GitHub: https://github.com/yvanyin/metric3d
"""

import cv2
import numpy as np
import os
import torch
import open3d as o3d
import sys
from pathlib import Path

# Use the project file packages instead of the conda packages, i.e. add to system path for import
file = Path(__file__).resolve()
root = file.parents[0]
modules = ['Metric3D']
for m in modules:
    path = root / m
    if path.exists() and str(path) not in sys.path:
        sys.path.append(str(path))
        #print(f"Added {path} to sys.path")
    elif not path.exists():
        print(f"Error: {path} does not exist.")
    elif str(path) in sys.path:
        print(f"{path} already exists in sys.path")

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

class MetricDepthEstimator:
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
        model = torch.hub.load('yvanyin/metric3d', self.model_type, pretrain=True)
        model.to(self.device)
        return model
    
    def preprocess(self, image):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0).float()
        return image
    
    def predict_depth(self, image):
        self.model.eval()
        input = self.preprocess(image)
        with torch.no_grad():
            depth, confidence, output_dict = self.model.inference({'input': input})
        return depth

    def create_depthmap(self, image, depth, grayscale):
        """ From Midas.run """
        depth_min = depth.min()
        depth_max = depth.max()
        normalized_depth = 255 * (depth - depth_min) / (depth_max - depth_min)
        normalized_depth *= 3

        right_side = np.repeat(np.expand_dims(normalized_depth, 2), 3, axis=2) / 3
        if not grayscale:
            right_side = cv2.applyColorMap(np.uint8(right_side), cv2.COLORMAP_INFERNO)

        if image is None:
            return right_side
        else:
            return np.concatenate((image, right_side), axis=1)
        
    def create_pointcloud(self, image, depth_map, name=None, outdir=None):
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
        # Plot point cloud
        #o3d.visualization.draw_geometries([pcd])

        if outdir is not None and name is not None:
            if not os.path.exists(outdir):
                os.makedirs(outdir)
            out_path = os.path.join(outdir, f'{name[:-41]}.ply')
            o3d.io.write_point_cloud(out_path, pcd)
            print(f'Point cloud saved to {out_path}')

    def create_csv(self, label_file, depth_map, time):
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

            # Calculate the mean depth within the ROI (method should probably be changed later)
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
            results.append([os.path.splitext(os.path.basename(label_file[:-44]))[0], classes[class_id], mean_depth, true_depth, time])

        # Calculate the average error
        average_error = total_error / count if count != 0 else 0

        # Print and return the average error
        print(f"Average Error: {average_error}")
        return results
