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
import matplotlib
import numpy as np
import os
import torch
import open3d as o3d
import sys
from pathlib import Path
import time

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
        self.device = device if device is not None else torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu') # torch.device('mps') if torch.backends.mps.is_available()
        print(f'Device: {self.device}')
        self.model_type = model_type
        self.model = self.load_model()
    
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
        input = input.to(self.device)
        with torch.no_grad():
            start = time.time()
            depth, confidence, output_dict = self.model.inference({'input': input})
            end = time.time()
        inference_time = end - start
        depth = depth.squeeze().cpu().numpy()
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
            # resize image instead of depthmap (size mismatch due to input processing in decoder)
            h_d, w_d = depth_colored.shape[:2]
            image = cv2.resize(image, (w_d, h_d))
            split_region = np.ones((h_d, 20, 3), dtype=np.uint8) * 255
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
        #height, width = image.shape[:2]
        height, width = depth.shape[:2]
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
            print(f'Point cloud saved to {out_path}')


    def create_csv(self, label_file, depth, frame, time, metric):
        """
        Extract depth from a target ROI (e.g. bounding box).

        Parameters:
        label_file (str): Path to the YOLO format label file
        depth (np.array): The depth map of the image
        time: frame inference time
        metric (bool): Flag for choosing evaluation method

        Returns:
        float: The average error between the predicted mean depth and the true depth across all bounding boxes
        """
        
        # Get W and H
        if depth.shape[:2] == frame.shape[:2]:
            height, width = depth.shape[:2]
        else:
            depth = cv2.resize(depth, frame.shape[:2])
            height, width = depth.shape[:2]
            print(f'Resizing depthmap to fit BBs in original frame...')

        # Read YOLO labels from the file
        with open(label_file, 'r') as f:
            lines = f.readlines()

        # Store results for CSV output
        results = []
        true_depths = [] # relative
        estimated_depths = [] # relative
        id = '_' + label_file[-36:-33] # take 3 letters as ID hash for each image
        total_error = 0
        count = 0

        for line in lines:
            parts = line.strip().split()
            class_id, x_center, y_center, bbox_width, bbox_height, true_depth = map(float, parts)
            
            # Convert normalized coordinates to absolute pixel values
            x_center *= width
            y_center *= height
            bbox_width *= width
            bbox_height *= height

            # ROI depth method
            x = int(x_center - bbox_width / 2)
            y = int(y_center - bbox_height / 2)
            w = int(bbox_width)
            h = int(bbox_height)

            # Ensure the bounding box coordinates are within the image dimensions
            x_start = max(x, 0)
            y_start = max(y, 0)
            x_end = min(x + w, depth.shape[1])
            y_end = min(y + h, depth.shape[0])

            # Extract the ROI from the depth map and calculate mean depth
            roi_depth = depth[y_start:y_end, x_start:x_end]
            mean_depth = np.mean(roi_depth)

            # Center point depth method
            center_depth = depth[int(y_center), int(x_center)]
            
            if metric:
                # Calculate the absolute difference between the mean depth and the true depth
                #depth_difference = abs(mean_depth - true_depth)
                depth_difference = abs(center_depth - true_depth)
                total_error += depth_difference
                results.append([os.path.basename(label_file[:-44]) + id, classes[class_id], mean_depth, center_depth, true_depth, time])
            else:
                # Store the depth and object
                true_depths.append(true_depth)
                estimated_depths.append(center_depth)

            count += 1

        if not metric:
            # Calculate the true and estimated proportional depths
            true_proportions = self.compute_proportional_depths(true_depths)
            estimated_proportions = self.compute_proportional_depths(estimated_depths)

            # Calculate error between true and estimated proportions
            for key in true_proportions:
                if key in estimated_proportions:
                    total_error += abs(true_proportions[key] - estimated_proportions[key])

        # Calculate the average error
        average_error = total_error / count if count > 0 else 0
        print(f"Average Error: {average_error}")

        if not metric:
            results.append([os.path.basename(label_file[:-44]) + id, average_error, time])
        
        return results
    

    def compute_proportional_depths(self, depth_values):
        """
        Computes proportional depths between all pairs of objects in a scene.
        
        Parameters:
        depth_values (list): List of depth values for all objects in the scene
        
        Returns:
        dict: Dictionary containing proportional depths for each object pair
        """
        n = len(depth_values)
        proportional_depths = {}
        
        for i in range(n):
            for j in range(i + 1, n):
                key = f"{i}-{j}"
                if depth_values[j] != 0:  # Avoid division by zero
                    proportional_depths[key] = depth_values[i] / depth_values[j]
                else:
                    proportional_depths[key] = np.inf
        
        return proportional_depths
