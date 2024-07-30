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
import cv2
import torch
import numpy as np
from transformers import pipeline
from PIL import Image
# endregion


class DepthEstimator:
    def __init__(self, weights, device=None):
        self.device = self.get_device(device)
        self.weights = weights
        self.model = self.load_model()
    
    def get_device(self, device):
        if device in ['cpu', 'cuda', 'mps']:
            if device == 'cuda' and torch.cuda.is_available():
                return torch.device("cuda")
            if device == 'mps' and torch.backends.mps.is_available():
                return torch.device("mps")
        return torch.device("cpu")
    
    def load_model(self):
        model = pipeline(task="depth-estimation", model=self.weights, device=self.device) # transformers pipeline
        return model
    
    def preprocess(self, image):
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        pil = Image.fromarray(rgb)
        return pil
    
    def predict_depth(self, image):
        input_tensor = self.preprocess(image)
        with torch.no_grad():
            output = self.model(input_tensor)
        depth = output["predicted_depth"]
        return depth

    def visualize_depth(self, image, depth, grayscale):
        h,w = image.shape[:2]
        # Part of transformers.pipelines.depth_estimation.DepthEstimationPipeline.postprocess(model_outputs)
        prediction = torch.nn.functional.interpolate(depth.unsqueeze(1), size=(h,w), mode="bicubic", align_corners=False)
        output = prediction.squeeze().cpu().numpy()
        formatted = (output * 255 / np.max(output)).astype("uint8")
        depth = Image.fromarray(formatted)

        # Part of MiDaS.run.create_side_by_side(image, depth, grayscale)
        right_side = np.repeat(np.expand_dims(depth, 2), 3, axis=2)
        if not grayscale:
            right_side = cv2.applyColorMap(np.uint8(right_side), cv2.COLORMAP_INFERNO)
        if image is None:
            return right_side
        else:
            return np.concatenate((image, right_side), axis=1)


def main():

    # Load OD models
    weights_objects = 'yolov5/yolov5x.pt'
    weights_hands = 'yolov5/hand.pt'
    od_model_objs = torch.hub.load('ultralytics/yolov5', 'custom', path=weights_objects).eval()
    od_model_hands = torch.hub.load('ultralytics/yolov5', 'custom', path=weights_hands).eval()

    # Load DE model
    depth_estimator = DepthEstimator(
        weights="LiheYoung/depth-anything-small-hf", 
        device='cuda' if torch.cuda.is_available() else 'cpu' # 'mps' currently not supported in DAv1
        )

    # Load DS
    ds = os.listdir('datasets/HaND_augmented/images/')
    n = len(ds)
    
    # Loop over testing data
    for i, file in enumerate(ds):

        # Read the image
        frame = cv2.imread('datasets/HaND_augmented/images/' + file)
        
        if frame is None:
            print(f"Failed to load image: {file}")
            continue

        # Perform object detection (important later on)
        #rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        #detections_objs = od_model_objs(frame)
        #detections_hands = od_model_hands(frame)
        # combine detections
        #detections = 

        # Perform depth estimation and visualize relative depth map
        depth = depth_estimator.predict_depth(frame)
        visual = depth_estimator.visualize_depth(frame, depth, False)
        print(f'Depth Image {i+1}/{n}, Min = {depth.min()}, Max = {depth.max()}')
        cv2.imshow("Depth map", visual)
        if cv2.waitKey(50) & 0xFF == ord('q'):
            break


if __name__ == "__main__":
    main()