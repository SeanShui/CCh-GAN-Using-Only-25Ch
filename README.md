# ChFontGAN25: Chinese Handwritten Style Transfer Using Only 25 Target


## Abstract
Chinese handwritten style transfer is a challenging task because there are more than ten
thousands of different characters and at least 3,000 of them are commonly used. Most
of the existing methods requires more than 100 characters as target examples. In this
paper we propose a Chinese character style transfer system that use 25 target examples.
First, a fixed set of the 25 characters is carefully selected. Then, Fréchet Inception
Distance (FID) is used to find the reference font for style transfer. Third, skeleton
information is added to a typical CycleGAN model to improve the quality of transferred
characters. FID is used again to evaluate the results and our method outperforms other
compared methods.


## Network Structure
![alt network](assets/Model.png)

We propose a two-stage Chinese font generative system. Stage 1 classify the nearest source style for stage 2 style transfe.

## Stage 1 Usage

Install from [pip](https://pypi.org/project/pytorch-fid/):

```
pip install pytorch-fid
```

Requirements:
- python3
- pytorch
- torchvision
- pillow
- numpy
- scipy

To compute the FID score between two styles of font images, where images of each style are contained in an individual folder:
```
python -m pytorch_fid path/to/source path/to/target
```

## Acknowledgements
Stage 1 code derived and rehashed from:

* [FID score for PyTorch](https://github.com/mseitzer/pytorch-fid) by [mseitzer](https://github.com/mseitzer)

Stage 2 code derived and rehashed from:

* [Generating handwritten Chinese characters using CycleGAN](https://github.com/ZC119/Handwritten-CycleGAN) by [ZC119](https://github.com/ZC119)

## License

Stage 1: Apache 2.0

Stage 2: GPL v3

Util: Apache 2.0
