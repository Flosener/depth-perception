"""
ChatGPT Prompt:
"I have a folder 'datasets/HaND/images' with testing images and on the same level a folder '/labels' containing labels in YOLOv5 format (.txt with classID, x, y, w, h, confidence) for hands in folder 'labels/hands' 
and objects in folder 'labels/objects'. Now, I want to re-number the class IDs of the hands from 0,1,2,3 to 100, 101, 102, 103, then filter out detections with unwanted class IDs from the object labels, 
then merge each hand and object label for each image and export it again, again with the image name .txt in a new folder 'labels/merged'. Can you give me the code?"
"""

import os

# Paths
images_path = 'images'
hands_labels_path = 'labels/hands'
objects_labels_path = 'labels/objects'
merged_labels_path = 'labels/merged'

# Create merged labels directory if it doesn't exist
os.makedirs(merged_labels_path, exist_ok=True)

# Allowed object class IDs (modify this list according to your requirements)
allowed_object_class_ids = {39, 45, 74, 41, 58, 40}  # bottle, bowl, clock, cup, potted plant, wine glass

def re_number_hand_class_ids(label_file):
    new_lines = []
    with open(label_file, 'r') as f:
        for line in f:
            parts = line.strip().split()
            class_id = int(parts[0])
            # Change hand class IDs to 0
            if class_id in {0, 1, 2, 3}:
                parts[0] = '0'
            # Remove confidence if present (assuming it's the 6th element)
            if len(parts) == 6:
                parts = parts[:5]
            new_lines.append(' '.join(parts))
    return new_lines

def filter_object_labels(label_file):
    new_lines = []
    with open(label_file, 'r') as f:
        for line in f:
            parts = line.strip().split()
            class_id = int(parts[0])
            if class_id in allowed_object_class_ids:
                # Remove confidence if present (assuming it's the 6th element)
                if len(parts) == 6:
                    parts = parts[:5]
                new_lines.append(' '.join(parts))
    return new_lines

def merge_labels(hand_labels, object_labels):
    return hand_labels + object_labels

def process_labels(image_name):
    hand_label_file = os.path.join(hands_labels_path, f"{image_name}.txt")
    object_label_file = os.path.join(objects_labels_path, f"{image_name}.txt")
    merged_label_file = os.path.join(merged_labels_path, f"{image_name}.txt")

    # Re-number hand class IDs
    hand_labels = re_number_hand_class_ids(hand_label_file) if os.path.exists(hand_label_file) else []
    
    # Filter unwanted class IDs from object labels
    object_labels = filter_object_labels(object_label_file) if os.path.exists(object_label_file) else []

    # Merge labels
    merged_labels = merge_labels(hand_labels, object_labels)

    # Write merged labels to file
    with open(merged_label_file, 'w') as f:
        f.write('\n'.join(merged_labels))

def main():
    # Get list of image names without extension
    image_names = [os.path.splitext(f)[0] for f in os.listdir(images_path) if f.endswith(('.jpg', '.jpeg', '.png'))]

    for image_name in image_names:
        process_labels(image_name)

if __name__ == "__main__":
    main()