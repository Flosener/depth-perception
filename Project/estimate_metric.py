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

# region Imports
import sys
from pathlib import Path

# Use the project file packages instead of the conda packages, i.e. add to system path for import
file = Path(__file__).resolve()
root = file.parents[0]
modules = ['yolov5', 'depthanythingv2']
for m in modules:
    path = root / m
    if path.exists() and str(path) not in sys.path:
        sys.path.append(str(path))
        #print(f"Added {path} to sys.path")
    elif not path.exists():
        print(f"Error: {path} does not exist.")
    elif str(path) in sys.path:
        print(f"{path} already exists in sys.path")


import argparse
import cv2
import glob
import matplotlib
import numpy as np
import os
import torch
import open3d as o3d

from depthanythingv2.metric_depth.depth_anything_v2.dpt import DepthAnythingV2
# endregion

class DepthEstimator:
    def __init__(self, encoder, dataset='hypersim', max_depth=20, device=None):
        self.device = self.get_device(device)
        self.encoder = encoder if encoder in ['vits', 'vitb', 'vitl', 'vitg'] else 'vitb'
        self.configs = {
        'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
        'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
        'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
        'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
        }
        self.dataset = dataset
        self.max_depth = max_depth # 20 for indoor model, 80 for outdoor model
        self.model = self.load_model()
    
    def get_device(self, device):
        if device in ['cpu', 'cuda', 'mps']:
            if device == 'cuda' and torch.cuda.is_available():
                return torch.device("cuda")
            if device == 'mps' and torch.backends.mps.is_available():
                return torch.device("mps")
        return torch.device("cpu")
    
    def load_model(self):
        model = DepthAnythingV2(**{**self.configs[self.encoder], 'max_depth': self.max_depth})
        model.load_state_dict(torch.load(f'depthanythingv2/checkpoints/depth_anything_v2_metric_{self.dataset}_{self.encoder}.pth', map_location=self.device))
        model.to(self.device)
        return model
    
    def preprocess(self, image):
        # pre-processing already done in infer_image()
        return image
    
    def predict_depth(self, image):
        self.model.eval()
        input = self.preprocess(image)
        with torch.no_grad():
            depth = self.model.infer_image(input, input_size=518)
        return depth

    def visualize_depth(self, image, depth, grayscale):
        h,w = image.shape[:2]

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
            split_region = np.ones((h, 20, 3), dtype=np.uint8) * 255
            combined_frame = cv2.hconcat([image, split_region, depth_colored])
            return combined_frame
        
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


def main():

    # Load OD models
    weights_objects = './yolov5/yolov5x.pt'
    weights_hands = './yolov5/hand.pt'
    od_model_objs = torch.hub.load('ultralytics/yolov5', 'custom', path=weights_objects).eval()
    od_model_hands = torch.hub.load('ultralytics/yolov5', 'custom', path=weights_hands).eval()

    # Load DE model
    depth_estimator = DepthEstimator(
        encoder='vitb',
        dataset='hypersim', # 'hypersim' for indoor model, 'vkitti' for outdoor model
        max_depth=20, # 20 for indoor model, 80 for outdoor model
        device='cpu' # change dpt.py line 220 to use 'cuda'
    )

    # Load DS
    ds = os.listdir('./datasets/HaND_augmented/images/')
    outdir = './datasets/HaND_augmented/visualization/pointclouds/'
    n = len(ds)
    
    # Loop over testing data
    for i, file in enumerate(ds):

        # Read the image
        frame = cv2.imread('datasets/HaND_augmented/images/' + file)
        
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
        depth = depth_estimator.predict_depth(frame)
        visual = depth_estimator.visualize_depth(frame, depth, False)
        #depth_estimator.create_point_cloud(frame, depth, file, outdir)
        print(f'\nDepth Image {i+1}/{n}, Min = {depth.min()}, Max = {depth.max()}')
        cv2.imshow("Depth map", visual)
        if cv2.waitKey(100) & 0xFF == ord('q'):
            break


if __name__ == "__main__":
    main()