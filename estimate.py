import os
import sys
import cv2
import csv
import torch
import numpy as np

def evaluate(depth_estimator=None, model='', model_type='', metric=False):

    # Setup data I/O
    outdir = f'./data/{model}/{model_type}_{depth_estimator.dataset}/' if model=='depthanything' and metric else f'./data/{model}/{model_type}/'
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
        writer.writerow(['filename', 'object', 'estimated_mean_depth', 'estimated_center_depth', 'true_depth', 'inference_time'])

    
    # Loop over dataset
    for i, file in enumerate(os.listdir(images)):

        print(f'Image {i+1}/{len(os.listdir(images))}') # dynamically get the number of images in the dataset

        # Read the image
        frame = cv2.imread(images+file)
        if frame is None:
            print(f"Failed to load image: {file}")
            continue

        # Perform depth estimation
        depth, inference_time = depth_estimator.predict_depth(frame)
        labelfile = labels + file[:-3] + 'txt'
        results = depth_estimator.create_csv(labelfile, depth, frame, inference_time)

        # Append results to CSV
        with open(csv_file, mode='a', newline='') as f:
            writer = csv.writer(f)
            writer.writerows(results)

        # Visualize results
        id = '_' + file[-36:-33] # take 3 letters as ID hash for each image
        name = file[:-44]+id
        if i < 10: # save 10 example images
            #depth_estimator.create_pointcloud(frame, depth, name, outdir+'pointclouds/')
            depth_image = depth_estimator.create_depthmap(frame, depth, False, name, outdir+'depthmaps/')
            depth_image = cv2.resize(depth_image, (960, 300)) # W, H
            cv2.imshow("Depth map", depth_image)

            if cv2.waitKey(100) & 0xFF == ord('q'):
                break


def run(depth_estimator=None, source=0):

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
    
    metric = True
    eval = True
    source = 0

    # Choose depth estimator and model size
    model = 'unidepth' # 'midas'
    model_type = 'cnvnxtl' # 'dpt_swin2_tiny_256'

    assert model in ['depthanything', 'midas', 'metric3d', 'unidepth'], f"Model '{model}' is not available."
    
    if model == 'depthanything':
        assert model_type in ['vits', 'vitb', 'vitl', 'vitg'], f"Model type '{model_type}' is not available. Available: vits, vitb, vitl, vitg"
        max_depth = 20 if metric else None

        # CPU: vits ~1.25s, vitb ~2.8s, vitl ~8.3s (both, metric and relative)
        from depthanything import DepthAnythingEstimator
        depth_estimator = DepthAnythingEstimator(
            model_type=model_type,
            max_depth=max_depth, # 20 for indoor model (-> hypersim), 80 for outdoor model (-> vkitti), None for relative depth
        )

    elif model == 'midas':
        if metric:
            assert model_type in ['ZoeD_N', 'ZoeD_K', 'ZoeD_NK'], f"Model type '{model_type}' is not available. Available: ZoeD_N, ZoeD_K, ZoeD_NK"
        else:
            # Downgrade to timm == 0.6.12, e.g. for swin and levit (https://github.com/isl-org/MiDaS/issues/225#issuecomment-2211808309)
            valid_models = ['dpt_beit_large_512', 'dpt_beit_large_384', 'dpt_beit_base_384', 'dpt_swin2_large_384', 'dpt_swin2_base_384',
                            'dpt_swin2_tiny_256', 'dpt_swin_large_384', 'dpt_levit_224', 'dpt_large_384',
                            'dpt_hybrid_384', 'midas_v21_384', 'midas_v21_small_256']
            available_models = ", ".join(valid_models)
            assert model_type in valid_models, f"Model type '{model_type}' is not available. Available: {valid_models}"
        
        # CPU: metric ~7.4s, 
        # relative -- dpt_levit_224 ~0.06s, dpt_swin2_tiny_256 ~0.26s ***, midas_v21_small_256 ~0.38s, midas_v21_384 ~0.39s, 
        # dpt_beit_base_384 ~1.45s, dpt_swin2_large_384 ~ 1.9s, dpt_swin_large_384 ~1.2s, dpt_swin2_base_384 ~1.22s, dpt_hybrid_384 ~2s, 
        # dpt_large_384 ~3.1s, dpt_beit_large_384 ~3.5s, dpt_beit_large_512 ~8.8s
        from midas import MidasDepthEstimator
        depth_estimator = MidasDepthEstimator(
            model_type=model_type,
        )
    
    elif model == 'metric3d': # frame shape and depth shape differ -- depth shape is changed in decoder
        print('Metric3D supports only metric depth estimation (CUDA required). Switching to metric estimation.') if not metric else None
        assert model_type in ['vits', 'vitl', 'vitg'], f"Model type '{model_type}' is not available. Available: vits, vitl, vitg"
        types = {'vits': 'metric3d_vit_small', 'vitl': 'metric3d_vit_large', 'vitg': 'metric3d_vit_giant2'}

        from metric3d import MetricDepthEstimator
        depth_estimator = MetricDepthEstimator(
            model_type=types[model_type],
        )

    elif model == 'unidepth':
        print('UniDepth supports only metric depth estimation (CUDA required). Switching to metric estimation.') if not metric else None
        assert model_type in ['vits', 'vitl', 'cnvnxtl'], f"Model type '{model_type}' is not available. Available: vits, vitl, cnvnxtl"
        types = {'vits': 'v2-vits14', 'vitl': 'v2-vits14', 'cnvnxtl': 'v1-cnvnxtl'}

        from unidepth import UniDepthEstimator
        depth_estimator = UniDepthEstimator(
            model_type=types[model_type],
        )
    
    # Run evaluation on test dataset or run camera stream estimation
    if eval:
        evaluate(depth_estimator, model, model_type, metric)
    else:
        run(depth_estimator, source)