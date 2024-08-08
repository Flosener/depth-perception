# Depth Estimation Evaluation

I am here sharing the code for evaluation of four depth estimators, namely DepthAnythingV2, MiDaS/ZoeDepth, Metric3Dv2, UniDepth.

## Setup

- Create separate environments from the respective requirements files for each estimator as they all rely on different packages and versions
- If you want to use DepthAnythingV2 you have to download the weights first from https://github.com/DepthAnything/Depth-Anything-V2/tree/main/metric_depth into depthanythingv2/checkpoints
- Specifiy the model name 'model' and weights 'model_type' of the depth estimator in the 'estimate.py' file's main idiom (also set them in the Estimator loader!)
- run estimate.py -> this will import the respective class and call run() to predict depth of each dataset image and save depthmap, pointcloud and predicted depths in a csv file

## Remarks

- When creating a conda environment from the zoedepth environment.yml, solving the environment could take very long. To solve: https://www.anaconda.com/blog/a-faster-conda-for-a-growing-community
- To use GPU ('cuda') you have to use the correct torch, torchvision and xformers package versions. Check cuda version with $ nvcc -V and download specific pytorch versions here: https://pytorch.org/get-started/previous-versions/
- above point may result in a xformer warning (depthanything and metric3d) which can be resolved by running: pip install torch==2.0.1+cu118 torchvision==0.15.2+cu118 torchaudio==2.0.2 torchtext==0.15.2 torchdata==0.6.1 --index-url https://download.pytorch.org/whl/cu118

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