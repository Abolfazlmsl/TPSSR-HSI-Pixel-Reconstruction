# PFGIN

The implementation of paper ["Predictive Filtering Integrated Generative Remote Sensing Hyperspectral Image Inpainting"](https://ieeexplore.ieee.org/document/10829644)


## Introduction
 An HSI inpainting method based on predictive image filtering integrated generative restoration is proposed, which can adaptively handle different scenes with dynamically predicted filtering kernels according to different inputs. 
 The generator of the proposed predictive filtering integrated generative HSI inpainting network (PFGIN) comprises two interconnected collaborative branches: the kernel prediction network (KPN) and the filtering-guided generative network (FGN).
 FGN provides features to KPN, and the KPN dynamically predicts the kernels of designed spatial-spectral filtering according to its input.  

 ## Getting Started

### 1. Preparing your training/testing datasets

Here, we give an example for single scale data preparation of Washington DC Mall data and c data.

Firstly, download WDCM  hyperspectral images into ```data/tr_data```, then download Indian-pines data to ```data/val_data```,

```angular2html
data/ICVL
├── tr_data
│    └──WDCM (*.tif)
├── val_data
│    └──Indian-pines (*.tif)
```
Then run the python code ```main.py``` to train your model.

## Citation

```angular2html
@ARTICLE{10829644,
  author={Wu, Yinhu and Zhang, Junping and Liu, Dongyang},
  journal={IEEE Geoscience and Remote Sensing Letters}, 
  title={Predictive Filtering Integrated Generative Remote Sensing Hyperspectral Image Inpainting}, 
  year={2025},
  volume={22},
  number={},
  pages={1-5},
  keywords={Filtering;Kernel;Image filtering;Generative adversarial networks;Generators;Image restoration;Training;Sensors;Semantics;Filters;Filtering guided generative inpainting;hyperspectral images (HSIs) inpainting;predictive spatial-spectral filtering (PSSF);remote sensing},
  doi={10.1109/LGRS.2025.3526177}}
```
## Acknowledgement

This repo is built mainly based on [MISF](https://github.com/tsingqguo/misf). We thank a lot for their contributions to the community.
