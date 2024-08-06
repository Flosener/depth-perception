# HAnd Navigation in Depth

The dataset contains single images (extracted from videos) containing scenarios of grasping objects with the hand from various places indoor and outdoor. Objects include the COCO classes. 
Videos are captured with a Logitech C922. Ground truth labels for distance between hand and target object is measured with the Rockseed S2 laser distance meter.
Uploaded here: https://universe.roboflow.com/optivist/hand-rmcj2 

Keywords: hand, object; grasping, reaching; depth, estimation, prediction; navigation, guidance, assistance; tactile bracelet

 ## Details

Indoor:
- kitchen, living room, office
- lighting: direct, indirect, artificial (room light)

Outdoor:
- garden table
- lighting: direct (sunny), indirect (cloudy)


Variables:
- depth: close, medium, far (~35-120cm)
- location: indoor, outdoor
- lighting: direct, indirect, artificial
- complexity: with/without occlusions
- view: front, side (does not refer to perspective on hand)

## Dataset information

HaND - v4 allclasses_augmented
https://universe.roboflow.com/optivist/hand-rmcj2
Provided by Florian Pätzold
License: CC BY 4.0
==============================

This dataset was exported via roboflow.com on July 18, 2024 at 2:30 PM GMT
The dataset includes 477 images.
Coco-hands are annotated in YOLO v5 PyTorch format.

The following pre-processing was applied to each image:
* Auto-orientation of pixel data (with EXIF-orientation stripping)

The following augmentation was applied to create 3 versions of each source image:
* 50% probability of horizontal flip
* 50% probability of vertical flip
* Randomly crop between 0 and 30 percent of the image
* Random brigthness adjustment of between -15 and +15 percent
* Random exposure adjustment of between -10 and +10 percent
* Random Gaussian blur of between 0 and 2.5 pixels