import os
import sys
import cv2
import csv
import torch
import numpy as np

def evaluate(depth_estimator=None, model='', model_type=''):

    # Setup data I/O
    outdir = f'./data/{model}/{model_type}/'
    if not os.path.exists(outdir):
        os.makedirs(outdir)

    # Load dataset
    dataset = './datasets/HaND_augmented/'
    images = dataset + 'images/'
    labels = dataset + 'labels/'
    # Initialize CSV
    csv_file = outdir + 'estimated_depth.csv'
    with open(csv_file, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['filename', 'object', 'estimated_depth', 'true_depth', 'inference_time'])

    
    # Loop over dataset
    for i, file in enumerate(os.listdir(images)):

        print(f'Image {i+1}/477') # 477 images in testing dataset

        # Read the image
        frame = cv2.imread(images+file)
        if frame is None:
            print(f"Failed to load image: {file}")
            continue

        # Perform depth estimation
        depth, inference_time = depth_estimator.predict_depth(frame)
        labelfile = labels + file[:-3] + 'txt'
        results = depth_estimator.create_csv(labelfile, depth, inference_time)

        # Append results to CSV
        with open(csv_file, mode='a', newline='') as f:
            writer = csv.writer(f)
            writer.writerows(results)

        # Visualize results
        id = '_' + file[-36:-33] # take 3 letters as ID hash for each image
        name = file[:-44]+id
        depth_estimator.create_pointcloud(frame, depth, name, outdir+'pointclouds/')
        depth_image = depth_estimator.create_depthmap(frame, depth, False, name, outdir+'depthmaps/')
        depth_image = cv2.resize(depth_image, (960, 300)) # W, H
        cv2.imshow("Depth map", depth_image)

        if cv2.waitKey(100) & 0xFF == ord('q'):
            break


def run(depth_estimator=None, model='', model_type='', source=0):

    # Data I/O
    cap = cv2.VideoCapture(source)
    times = [] # list for tracking average inference time
    i = 0
    
    # Loop over datastream
    while True:

        # Read the frame
        ret, frame = cap.read()
        i += 1

        if not ret:
            break

        # Perform depth estimation and visualize
        depth, inference_time = depth_estimator.predict_depth(frame)
        times.append(inference_time)
        depth_image = depth_estimator.create_depthmap(frame, depth, False)
        depth_image = cv2.resize(depth_image, (960, 300)) # W, H
        cv2.imshow("Depth map", depth_image)
        print(f'Frame {i}, time: {round(inference_time, 2)}, avg: {round(np.mean(times), 2)}')

        if cv2.waitKey(100) & 0xFF == ord('q'):
            break


if __name__ == "__main__":
    
    metric = False
    eval = False
    source = 1

    # Choose depth estimator and model type name (for saving)
    model = 'depthanything'
    model_type = 'vits'

    assert model in ['depthanything', 'zoedepth', 'metric3d', 'unidepth']
    
    if model == 'depthanything':
        from depthanything import DepthAnythingEstimator
        if metric: # CPU: vits ~1.25s, vitb ~2.8s, vitl ~8.3s
            depth_estimator = DepthAnythingEstimator(
                model_type='vits', # vits, vitb, vitl
                dataset='hypersim', # 'hypersim' for indoor model, 'vkitti' for outdoor model
                max_depth=20, # 20 for indoor model, 80 for outdoor model
                device=torch.device('cpu')
            )
        else: # CPU: same as above
            depth_estimator = DepthAnythingEstimator(
                model_type='vits', # vits, vitb, vitl
                device=torch.device('cpu'),
            )
    elif model == 'zoedepth':
        if metric: # ~7.3s on CPU (N), 
            from zoedepth_metric import ZoeDepthEstimator
            depth_estimator = ZoeDepthEstimator(
                model_type = 'ZoeD_N', # ZoeN (nyu, indoor), ZoeK (kitti, outdoor), ZoeNK
            )
        else:
            pass
    elif model == 'metric3d':
        if metric:
            # REQUIRES CUDA
            from metric3d_metric import MetricDepthEstimator
            depth_estimator = MetricDepthEstimator(
                model_type = 'metric3d_vit_small', # metric3d_vit_small, (metric3d_vit_large, metric3d_vit_giant2 --> too slow)
            )
        else:
            pass
    elif model == 'unidepth':
        if metric:
            # REQUIRES CUDA
            from unidepth_metric import UniDepthEstimator
            depth_estimator = UniDepthEstimator(
                model_type = 'v2-vits14', # , v2-vits14, v2-vitl14, v2old-vitl14, v1-vitl4, v1-cnvnxtl, v1-convnext-large
            )
        else:
            pass
    else:
        print('Model is not available.')
        sys.exit()
    
    if eval:
        evaluate(depth_estimator, model, model_type)
    else:
        run(depth_estimator, model, model_type, source)