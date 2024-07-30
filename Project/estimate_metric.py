"""
Copyright 2024 LiheYoung

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
@inproceedings{depthanything,
      title={Depth Anything: Unleashing the Power of Large-Scale Unlabeled Data}, 
      author={Yang, Lihe and Kang, Bingyi and Huang, Zilong and Xu, Xiaogang and Feng, Jiashi and Zhao, Hengshuang},
      booktitle={CVPR},
      year={2024}
}

GitHub: https://github.com/LiheYoung/Depth-Anything
"""

# region Imports
import sys
from pathlib import Path

# Use the project file packages instead of the conda packages, i.e. add to system path for import
file = Path(__file__).resolve()
root = file.parents[0]
modules = ['yolov5', 'depthanything']
for m in modules:
    path = root / m
    if path.exists() and str(path) not in sys.path:
        sys.path.append(str(path))
        #print(f"Added {path} to sys.path")
    elif not path.exists():
        print(f"Error: {path} does not exist.")
    elif str(path) in sys.path:
        print(f"{path} already exists in sys.path")

import os
from PIL import Image
import numpy as np
import cv2
import torch
from torchvision.transforms import Compose
from depthanything.depth_anything.dpt import DepthAnything
from depthanything.depth_anything.util.transform import Resize, NormalizeImage, PrepareForNet
# endregion

class DepthEstimator:
    def __init__(self, metric_weights, encoder, device=None):
        self.device = self.get_device(device)
        self.metric_weights = metric_weights
        self.encoder = encoder if encoder in ['vits', 'vitb', 'vitl'] else 'vitb'
        self.config = {
            'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
            'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
            'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]}
        }
        self.model = self.load_model()
    
    def get_device(self, device):
        if device in ['cpu', 'cuda', 'mps']:
            if device == 'cuda' and torch.cuda.is_available():
                return torch.device("cuda")
            if device == 'mps' and torch.backends.mps.is_available():
                return torch.device("mps")
        return torch.device("cpu")
    
    def load_model(self):
        # online method
        model = DepthAnything.from_pretrained(f'LiheYoung/depth_anything_{self.encoder}14')

        # offline method (requires weights download!)
        #model = DepthAnything(self.config[self.encoder])
        #model.load_state_dict(torch.load(f'./checkpoints/depth_anything_{self.encoder}14.pth'))

        # Load model weights for metric depth (requires weights download!)
        metric_weights = torch.load(self.metric_weights, map_location=self.device)
        model.load_state_dict(metric_weights, strict=False)
        model.to(self.device)
        return model
    
    def preprocess(self, image):
        h, w = image.shape[:2]
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        transform = Compose([
            Resize(
                width=w,
                height=h,
                resize_target=False,
                keep_aspect_ratio=True,
                ensure_multiple_of=14,
                resize_method='lower_bound',
                image_interpolation_method=cv2.INTER_CUBIC,
            ),
            NormalizeImage(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            PrepareForNet(),
        ])
        image = transform({'image': image})['image']
        image = torch.from_numpy(image).unsqueeze(0).to(self.device)
        return image
    
    def predict_depth(self, image):
        self.model.eval()
        input = self.preprocess(image)
        with torch.no_grad():
            depth = self.model(input) # (1,H,W)
        return depth

    def visualize_depth(self, image, depth, grayscale):
        h, w = image.shape[:2]
        depth_image = torch.nn.functional.interpolate(depth[None], (h, w), mode='bilinear', align_corners=False)[0, 0]
        depth_image = (depth_image - depth_image.min()) / (depth_image.max() - depth_image.min()) * 255.0
        depth_image = depth_image.cpu().numpy().astype(np.uint8) # (H,W)
        depth_image = cv2.applyColorMap(depth_image, cv2.COLORMAP_INFERNO) # (H,W,3)

        split_region = np.ones((image.shape[0], 20, 3), dtype=np.uint8) * 255
        combined_results = cv2.hconcat([image, split_region, depth_image])
        return combined_results


def main():

    # Load OD models
    weights_objects = './yolov5/yolov5x.pt'
    weights_hands = './yolov5/hand.pt'
    od_model_objs = torch.hub.load('ultralytics/yolov5', 'custom', path=weights_objects).eval()
    od_model_hands = torch.hub.load('ultralytics/yolov5', 'custom', path=weights_hands).eval()

    # Load DE model
    depth_estimator = DepthEstimator(
        metric_weights="./checkpoints/depth_anything_metric_depth_indoor.pt",
        encoder='vits',
        device='cpu'
    )

    # Load DS
    ds = os.listdir('./datasets/HaND_augmented/images/')
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
        print(f'\nDepth Image {i+1}/{n}, Min = {depth.min()}, Max = {depth.max()}')
        cv2.imshow("Depth map", visual)
        if cv2.waitKey(100) & 0xFF == ord('q'):
            break


if __name__ == "__main__":
    main()