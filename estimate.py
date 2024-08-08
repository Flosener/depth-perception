import os
import sys
import cv2
import csv

def run(depth_estimator=None, model='', model_type=''):

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
    
    
    # Loop over testing data
    for i, file in enumerate(os.listdir(images)):

        print(f'Image {i+1}/477') # 477 images in testing dataset

        # Read the image
        frame = cv2.imread(images+file)
        if frame is None:
            print(f"Failed to load image: {file}")
            continue

        # Perform depth estimation
        input = images+file if model == 'zoedepth' else frame # change to image path for zoedepth
        depth, inference_time = depth_estimator.predict_depth(input)
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
        depth_image = depth_estimator.create_depthmap(input, depth, False, name, outdir+'depthmaps/')
        depth_image = cv2.resize(depth_image, (960, 300)) # W, H
        #"""
        cv2.imshow("Depth map", depth_image)
        if cv2.waitKey(100) & 0xFF == ord('q'):
            break
        #"""


if __name__ == "__main__":
    
    # Choose depth estimator and model type name (for saving)
    model = 'unidepth'
    model_type = 'v2-vits14'

    assert model in ['depthanything', 'zoedepth', 'metric3d', 'unidepth']
    
    if model == 'depthanything':
        from depthanything import DepthAnythingEstimator
        depth_estimator = DepthAnythingEstimator(
            encoder='vits', # vits, vitb, vitl
            dataset='hypersim', # 'hypersim' for indoor model, 'vkitti' for outdoor model
            max_depth=20, # 20 for indoor model, 80 for outdoor model
            device='cuda' # change dpt.py line 220 to use 'cpu' instead of not supported 'mps' on Mac
        )
    elif model == 'zoedepth':
        from zoedepth import ZoeDepthEstimator
        depth_estimator = ZoeDepthEstimator(
            model_type = 'ZoeD_N', # ZoeN (nyu, indoor), ZoeK (kitti, outdoor), ZoeNK
            device='cuda'
        )
    elif model == 'metric3d':
        from metric3d import MetricDepthEstimator
        depth_estimator = MetricDepthEstimator(
            model_type = 'metric3d_vit_small', # metric3d_vit_small, metric3d_vit_large, metric3d_vit_giant2 (all require cuda)
            device='cuda'
        )
    elif model == 'unidepth':
        from unidepthv2 import UniDepthEstimator
        depth_estimator = UniDepthEstimator(
            model_type = 'v2-vits14', # , v2-vits14, v2-vitl14, v2old-vitl14, v1-vitl4, v1-cnvnxtl, v1-convnext-large
            device='cuda'
        )
    else:
        print('Model is not available.')
        sys.exit()
    
    run(depth_estimator, model, model_type)