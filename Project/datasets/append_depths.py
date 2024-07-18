"""
ChatGPT Prompt:
"Now I have my dataset with ds/images/ and ds/labels/ folders. Each label is yolov5 format, great. Now I have measured the objects' depths and saved in depths.csv. 
I want to add to all labels the respective depth for each detection. how?" + depths.csv file upload
---> still needed a lot of additional changes because the prompt was not very specific (e.g. there are two bowls with two depths but only one 'bowl' class in the labels)
"""

import os
import pandas as pd

# Paths
labels_path = 'HaND_augmented/labels'
depths_file = 'HaND_augmented/depths.csv'

# Read depths.csv
depths_df = pd.read_csv(depths_file)


def add_depth_to_labels(label_file, depths_df):
    label_name = os.path.basename(label_file) # 'az_front_complex_artificial_close_png.rf.b272f069b0b510bf167832ebf8b306a9.txt'
    filename = os.path.splitext(label_name)[0] # split off extension -> 'az_front_complex_artificial_close_png.rf.b272f069b0b510bf167832ebf8b306a9'
    parts = filename.split('_') # split by underscores
    scene_name = '_'.join(parts[:4]) # join first four parts -> 'az_front_complex_artificial'
    depths = depths_df[depths_df['scene'] == scene_name] # filter for relevant row

    # Get ground truth depth of each object
    dbottle = float(depths.iloc[0, 1]) 
    dbowl_close = float(depths.iloc[0, 2]) if float(depths.iloc[0, 2]) < float(depths.iloc[0, 3]) else float(depths.iloc[0, 3])
    dbowl_far = float(depths.iloc[0, 3]) if float(depths.iloc[0, 2]) < float(depths.iloc[0, 3]) else float(depths.iloc[0, 2])
    dclock = float(depths.iloc[0, 4])
    dcup_close = float(depths.iloc[0, 5]) if float(depths.iloc[0, 5]) < float(depths.iloc[0, 6]) else float(depths.iloc[0, 6])
    dcup_far = float(depths.iloc[0, 6]) if float(depths.iloc[0, 5]) < float(depths.iloc[0, 6]) else float(depths.iloc[0, 5])
    dplant = float(depths.iloc[0, 7])
    dglass_close = float(depths.iloc[0, 8]) if float(depths.iloc[0, 8]) < float(depths.iloc[0, 9]) else float(depths.iloc[0, 9])
    dglass_far = float(depths.iloc[0, 9]) if float(depths.iloc[0, 8]) < float(depths.iloc[0, 9]) else float(depths.iloc[0, 8])
    dhand_close = float(depths.iloc[0, 10])
    dhand_medium = float(depths.iloc[0, 11])
    dhand_far = float(depths.iloc[0, 12])

    # Append respective depth in each line
    new_lines = []
    with open(label_file, 'r') as f:
        for line in f:
            parts = line.strip().split() # '4 0.33298429319371725 0.8282122905027933 0.23350785340314137 0.3435754189944134'
            cls = int(parts[0]) # 4

            if cls == 0:
                depth = dbottle
            elif cls == 1:
                depth = dbowl_close
            elif cls == 2:
                depth = dbowl_far
            elif cls == 3:
                depth = dclock
            elif cls == 4:
                depth = dcup_close
            elif cls == 5:
                depth = dcup_far
            elif cls == 6:
                depth = dhand_close
            elif cls == 7:
                depth = dhand_far
            elif cls == 8:
                depth = dhand_medium
            elif cls == 9:
                depth = dplant
            elif cls == 10:
                depth = dglass_close
            elif cls == 11:
                depth = dglass_far

            if depth:
                parts.append(str(depth))
            new_lines.append(' '.join(parts))
    
    return new_lines


def process_labels(labels_path, depths_df):
    for label_file in os.listdir(labels_path):
        if label_file.endswith('.txt'):
            full_path = os.path.join(labels_path, label_file)
            new_labels = add_depth_to_labels(full_path, depths_df)
            with open(full_path, 'w') as f:
                f.write('\n'.join(new_labels))


if __name__ == "__main__":
    process_labels(labels_path, depths_df)