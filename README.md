[![License CC BY-NC-SA 4.0](https://img.shields.io/badge/license-CC4.0-blue.svg)](https://raw.githubusercontent.com/NVIDIA/FastPhotoStyle/master/LICENSE.md)
![Python 2.7](https://img.shields.io/badge/python-2.7-green.svg)
![Python 3.5](https://img.shields.io/badge/python-3.5-green.svg)

## FastPhotoStyle

### License
Copyright (C) 2018 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).

<img src="https://raw.githubusercontent.com/NVIDIA/FastPhotoStyle/master/teaser.png" width="800" title="Teaser results"> 


### Setup environment
```bash
cd FastPhotoStyle
conda create -n cv python==3.7
conda activate cv
python -m pip install --upgrade pip
pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu117
pip install cupy-cuda11x
conda install -c conda-forge scikit-umfpack
pip install -U setuptools
cd ..
pip download pynvrtc==9.2 --no-deps
tar -xzvf pynvrtc-9.2.tar.gz
cd pynvrtc-9.2
pip install .
cd ../FastPhotoStyle
pip install scipy==1.2.1
```

### Run on CPU

##### Without segmentation
```bash
python demo.py --content_image_path pictures/content/in34.png --style_image_path pictures/style/tar34.png --output_image_path pictures/results/result34.png --cuda 0
```

##### With segmentation
```bash
python demo.py --content_image_path pictures/content/in34.png --content_seg_path pictures/content_segment/in34.png --style_image_path pictures/style/tar34.png --style_seg_path pictures/style_segment/tar34.png --output_image_path pictures/results/result34.png --cuda 0
```

### About

Given a content photo and a style photo, the code can transfer the style of the style photo to the content photo. The details of the algorithm behind the code is documented in our arxiv paper. Please cite the paper if this code repository is used in your publications.

[A Closed-form Solution to Photorealistic Image Stylization](https://arxiv.org/abs/1802.06474) <br> 
[Yijun Li (UC Merced)](https://sites.google.com/site/yijunlimaverick/), [Ming-Yu Liu (NVIDIA)](http://mingyuliu.net/), [Xueting Li (UC Merced)](https://sunshineatnoon.github.io/), [Ming-Hsuan Yang (NVIDIA, UC Merced)](http://faculty.ucmerced.edu/mhyang/), [Jan Kautz (NVIDIA)](http://jankautz.com/) <br>
European Conference on Computer Vision (ECCV), 2018 <br>


### Tutorial

Please check out the [tutorial](TUTORIAL.md).


