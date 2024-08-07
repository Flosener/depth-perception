import os
import cv2
import csv
from metric3d import MetricDepthEstimator

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
        writer.writerow(['filename', 'object', 'estimated_depth', 'true_depth'])
    
    
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
        results = depth_estimator.create_csv(labels + file[:-3]+'txt', depth, inference_time)

        # Append results to CSV
        with open(csv_file, mode='a', newline='') as f:
            writer = csv.writer(f)
            writer.writerows(results)

        # Visualize results
        depth_estimator.create_pointcloud(frame, depth, file[:-3], outdir+'pointclouds/')
        depth_image = depth_estimator.create_depthmap(frame, depth, False, file[:-3], outdir+'depthmaps/')
        cv2.imshow("Depth map", depth_image)

        if cv2.waitKey(100) & 0xFF == ord('q'):
            break


if __name__ == "__main__":
    
    depth_estimator = MetricDepthEstimator(
        model_type = 'metric3d_vit_small', # metric3d_vit_small, metric3d_vit_large, metric3d_vit_giant2 (all require xformers --> cuda)
        device='cpu'
    )

    model = 'depthanything'
    model_type = 'vits_hypersim'

    run(depth_estimator, model, model_type)