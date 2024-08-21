# Depth Estimation Evaluation

I am here sharing the code for evaluation of four depth estimators, namely DepthAnythingV2, MiDaS/ZoeDepth, Metric3D, UniDepth.

## Prerequisites
- CUDA for Metric3D and UniDepth

## Setup

- Create separate environments from the respective requirements files for each estimator as they all rely on different packages and versions
- If you want to use DepthAnythingV2 you have to download the weights first from https://github.com/DepthAnything/Depth-Anything-V2/tree/main/metric_depth (metric) and https://huggingface.co/spaces/LiheYoung/Depth-Anything/tree/main/checkpoints (relative) into depthanythingv2/checkpoints
- Same for relative depth estimation with MiDaS: https://github.com/isl-org/MiDaS?tab=readme-ov-file#setup (into MiDaS/weights)
- Specifiy the model name 'model' and weights 'model_type' of the depth estimator in the 'estimate.py' file's main idiom
- run estimate.py (no CLI command available as args parser is not implemented yet) -> this will import the respective class and call eval() to predict depth of each dataset image and save depthmap, pointcloud and predicted depths in a csv file OR call run() on datastream from a camera 'source'

## Remarks

- When creating a conda environment from the zoedepth environment.yml, solving the environment could take very long. Faster solution: https://www.anaconda.com/blog/a-faster-conda-for-a-growing-community
- To use GPU ('cuda') you have to use compatible cuda, pytorch, torchvision and xformers package versions. Check cuda version with $ nvcc -V and download corresponding pytorch versions here: https://pytorch.org/get-started/previous-versions/ (I think torch==2.4.0 is currently buggy)

## Citations

@inproceedings{piccinelli2024unidepth,
    title     = {{U}ni{D}epth: Universal Monocular Metric Depth Estimation},
    author    = {Piccinelli, Luigi and Yang, Yung-Hsu and Sakaridis, Christos and Segu, Mattia and Li, Siyuan and Van Gool, Luc and Yu, Fisher},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    year      = {2024}
}

@article{hu2024metric3dv2,
  title={Metric3D v2: A Versatile Monocular Geometric Foundation Model for Zero-shot Metric Depth and Surface Normal Estimation},
  author={Hu, Mu and Yin, Wei and Zhang, Chi and Cai, Zhipeng and Long, Xiaoxiao and Chen, Hao and Wang, Kaixuan and Yu, Gang and Shen, Chunhua and Shen, Shaojie},
  journal={arXiv preprint arXiv:2404.15506},
  year={2024}
}

@misc{https://doi.org/10.48550/arxiv.2302.12288,
  doi = {10.48550/ARXIV.2302.12288},
  url = {https://arxiv.org/abs/2302.12288},
  author = {Bhat, Shariq Farooq and Birkl, Reiner and Wofk, Diana and Wonka, Peter and Müller, Matthias},
  keywords = {Computer Vision and Pattern Recognition (cs.CV), FOS: Computer and information sciences, FOS: Computer and information sciences},
  title = {ZoeDepth: Zero-shot Transfer by Combining Relative and Metric Depth},
  publisher = {arXiv},
  year = {2023},
  copyright = {arXiv.org perpetual, non-exclusive license}
}

@article{depth_anything_v2,
  title={Depth Anything V2},
  author={Yang, Lihe and Kang, Bingyi and Huang, Zilong and Zhao, Zhen and Xu, Xiaogang and Feng, Jiashi and Zhao, Hengshuang},
  journal={arXiv:2406.09414},
  year={2024}
}