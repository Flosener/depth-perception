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

import cv2
import numpy as np
import os
import torch
import open3d as o3d
from PIL import Image
import sys
from pathlib import Path
import time

# Use the project file packages instead of the conda packages, i.e. add to system path for import
file = Path(__file__).resolve()
root = file.parents[0]
modules = ['midas']
for m in modules:
    path = root / m
    if path.exists() and str(path) not in sys.path:
        sys.path.append(str(path))
        #print(f"Added {path} to sys.path")
    elif not path.exists():
        print(f"Error: {path} does not exist.")
    elif str(path) in sys.path:
        print(f"{path} already exists in sys.path")

from midas.run import create_side_by_side, process
from midas.ZoeDepth.zoedepth.utils.misc import colorize
from midas.midas.model_loader import load_model

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


class MidasDepthEstimator:
    def __init__(self, model_type, device=None):
        self.device = device if device is not None else torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu') # torch.device('mps') if torch.backends.mps.is_available()
        print(f'Device: {self.device}')
        self.model_type = model_type
        self.metric = True if 'Zoe' in self.model_type else False
        self.model = self.load_model()

    def load_model(self):
        if self.metric:
            torch.hub.help("intel-isl/MiDaS", "DPT_BEiT_L_384", force_reload=True)
            model = torch.hub.load("isl-org/ZoeDepth", self.model_type, pretrained=True)
        else:
            model, self.transform, self.net_w, self.net_h = load_model(self.device, f'MiDaS/weights/{self.model_type}.pt', self.model_type)
        model.to(self.device)
        return model
    
    def preprocess(self, image):
        if self.metric:
            image = Image.fromarray(image) #.convert("RGB")
        else:
            original_image_rgb = np.flip(image, 2)  # in [0, 255] (flip required to get RGB)
            image = self.transform({"image": original_image_rgb/255})["image"]
        return image
    
    def predict_depth(self, image):
        self.model.eval()
        input = self.preprocess(image)
        with torch.no_grad():
            start = time.time()
            depth = self.model.infer_pil(input) if self.metric else process(self.device, self.model, self.model_type, input, (self.net_w, self.net_h), image.shape[1::-1], True, False)
            end = time.time()
        inference_time = end - start
        return depth, inference_time

    def create_depthmap(self, image, depth, grayscale, name=None, outdir=None):
        if self.metric:
            image = self.preprocess(image)
            size = image.size
            depth_image = Image.fromarray(colorize(depth))
            # Stack img and pred side by side for comparison and save
            depth_image = depth_image.resize(size)
            combined_frame = Image.new("RGB", (size[0]*2, size[1]))
            combined_frame.paste(image, (0, 0))
            combined_frame.paste(depth_image, (size[0], 0))
        else:
            combined_frame = create_side_by_side(image, depth, grayscale)

        # Save to output directory if specified
        if outdir is not None and name is not None:
            if not os.path.exists(outdir):
                os.makedirs(outdir)
            output_path = os.path.join(outdir, name+'.png')
            if self.metric:
                combined_frame.save(output_path)
            else:
                cv2.imwrite(output_path, combined_frame)
            print(f'Saved depth visualization to {output_path}')
        
        if self.metric:
            return create_side_by_side(image, depth, grayscale) / 255
        else:
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
            print(f'Point cloud saved to {out_path}')

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

        # Print and return the average error
        #print(f"Average Error: {average_error}")
        return results