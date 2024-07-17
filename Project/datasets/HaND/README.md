# HAnd Navigation in Depth

The dataset contains single images (extracted from videos) containing scenarios of grasping objects with the hand from various places indoor and outdoor. Objects include the COCO classes. 
Videos are captured with a Logitech C922. Ground truth labels for distance between hand and target object is measured with the Rockseed S2 laser distance meter.

keywords: hand, object; grasping, reaching; depth, estimation, prediction; navigation, guidance, assistance; tactile bracelet

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

Analysis:
- distance: depth estimation
- illuminance in ROI
- depth maps