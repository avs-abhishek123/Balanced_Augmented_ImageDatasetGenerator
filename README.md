# Balanced Augmented Image Dataset Generator using AugStatic - A lightwight Image Augmentation Library

<img src = 'https://github.com/avs-abhishek123/AugStatic/blob/8809c0700d3a8900fbe3e92ebc47ca39f2304922/augstatic.png' width = 1020 height = 450 align = "center">

<!--<h1 align ="center" style="font-size: 1400px"> AugStatic - A Light-Weight Image Augmentation Library </h1>-->
<h1 align ="center" style="color: purple; font-size: 80px;"><b><u>AugStatic - A Light-Weight Image Augmentation Library</u></b></h1>

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/augstatic-a-light-weight-image-augmentation/augstatic-image-augmentation-on-cifar10&color=072c79)](https://paperswithcode.com/sota/augstatic-image-augmentation-on-cifar10?p=augstatic-a-light-weight-image-augmentation)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/augstatic-a-light-weight-image-augmentation/augstatic-image-augmentation-on-coco&color=ff3131)](https://paperswithcode.com/sota/augstatic-image-augmentation-on-coco?p=augstatic-a-light-weight-image-augmentation)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/augstatic-a-light-weight-image-augmentation/augstatic-image-augmentation-on-imagenet&color=#FFED00)](https://paperswithcode.com/sota/augstatic-image-augmentation-on-imagenet?p=augstatic-a-light-weight-image-augmentation)

## Abstract
<p> <a href = 'https://www.ijrar.org/papers/IJRAR22B1901.pdf'> [ Paper Link ] </a> </p>
 The mixed data consists of various structured and unstructured data. The exponential boom of the amount of data has made the datasets of varying samples. This paper focuses on the image dataset generator that balances an imbalanced dataset using the AugStatic augmentation library. The datasets, including various classes, are said to be balanced if the number of samples in the classes is equal. This gives a fair chance for the model to learn about all the classes. An augmented image dataset balances an imbalanced image dataset. It is useful when the data is less in a specific category, generating new data with it. There are multiple augmentation techniques supported by the AugStatic library that helps in developing the augmented balanced library by the iterative implementation of the augmentations on the generated dataset. It takes an input of the existing dataset, majority and minority classes sample count that returns a balanced image dataset by iteratively applying the augmentations on the generated augmented images in the minority class. This generator is efficient and can be used for any image dataset.


---

## Background Research Work

![Aug_types](https://github.com/avs-abhishek123/AugStatic/blob/7f33a6d188b09489af5dc092dc5c77f96724bcd0/Aug_types.PNG)

### Various types of augmentations were researched and compiled into a compact, lightweight, and practical library.

* ### **Imgaug**  [ [Library-GitHub-Link](https://github.com/aleju/imgaug) | [Documentation-Link](https://imgaug.readthedocs.io/en/latest/) ] 


  <p>
    <img src = 'https://github.com/avs-abhishek123/AugStatic/blob/fed5dd54e822dac6ffc3dd37ede0e2b39ecfef8d/imgaug2.png' width = 350 height = 200 align = "right">
    <h4> The salient feature are – </h4>
    
    <ul>
      <li> It contains over forty image augmentation techniques. </li>
      <li> Functionality to augment images with masks, key points, bounding boxes, and heat maps. </li>
      <li> Easier to augment the image dataset for object detection and segmentation problems. </li>
      <li> Complex augmentation pipelines. </li>
      <li> Many helper functions for augmentation visualization, conversion, and more. </li> 
    </ul>
  </p>
  
* ### **Augmentor** [ [Library-GitHub-Link](https://github.com/mdbloice/Augmentor) | [Documentation-Link](https://augmentor.readthedocs.io/en/stable/) ]
  <p>
  <img src = 'https://github.com/avs-abhishek123/AugStatic/blob/8809c0700d3a8900fbe3e92ebc47ca39f2304922/augmentor.png' width = 350 height = 200 align = "right">
  <h4> The salient feature are – </h4>
  
  <ul>
    <li> It has fewer possible augmentations compared to other packages. </li>
    <li> It supports extra features like size-preserving shearing, size-preserving rotations, and cropping, which is beneficial for machine learning pipelines. </li>
    <li> It supports to compose augmentation pipelines. </li>
    <li> It supports usage with PyTorch [8] and Tensorflow [1]. </li>
  </ul>
  </p>
  
* ### **Albumentations** [ [Library-GitHub-Link](https://github.com/albumentations-team/albumentations) | [Documentation-Link](https://albumentations.ai/docs/) ] 

  <p>
  <img src = 'https://github.com/avs-abhishek123/AugStatic/blob/8809c0700d3a8900fbe3e92ebc47ca39f2304922/albumentations.png' width = 350 height = 200 align = "right">
  <h4> The salient feature are – </h4>
  
  <ul>
    <li> It contains over forty image augmentation techniques. </li>
    <li> Functionality to augment images with masks, key points, bounding boxes, and heat maps. </li>
    <li> Easier to augment the image dataset for object detection and segmentation problems. </li>
    <li> Complex augmentation pipelines. </li>
    <li> Many helper functions for augmentation visualization, conversion, and more. </li>
    </ul>
  </p>
  
---

## Methods & Results 

| Augmentation Technique | Input Image | Output Image |
| --- | --- | --- |
| **Blur** | ![Input_image](https://github.com/avs-abhishek123/AugStatic/blob/c563cf249fd0f28453eb63d47634c3279b426031/input_image.jpg) | ![Blur](https://github.com/avs-abhishek123/AugStatic/blob/0f12cb0adf7b0c9bd68074f8891907067604e79f/OutputImages/Blur.jpg) |
| **CLAHE (Contrast Stretched Adaptive Histogram Equalization)** | ![Input_image](https://github.com/avs-abhishek123/AugStatic/blob/c563cf249fd0f28453eb63d47634c3279b426031/input_image.jpg) | ![CLAHE](https://github.com/avs-abhishek123/AugStatic/blob/0f12cb0adf7b0c9bd68074f8891907067604e79f/OutputImages/CLAHE.jpg) |
| **ChannelDropout** | ![Input_image](https://github.com/avs-abhishek123/AugStatic/blob/c563cf249fd0f28453eb63d47634c3279b426031/input_image.jpg) | ![ChannelDropout](https://github.com/avs-abhishek123/AugStatic/blob/0f12cb0adf7b0c9bd68074f8891907067604e79f/OutputImages/ChannelDropout.jpg) |
| **ChannelShuffle** | ![Input_image](https://github.com/avs-abhishek123/AugStatic/blob/c563cf249fd0f28453eb63d47634c3279b426031/input_image.jpg) | ![ChannelShuffle](https://github.com/avs-abhishek123/AugStatic/blob/0f12cb0adf7b0c9bd68074f8891907067604e79f/OutputImages/ChannelShuffle.jpg) |
| **ColorJitter** | ![Input_image](https://github.com/avs-abhishek123/AugStatic/blob/c563cf249fd0f28453eb63d47634c3279b426031/input_image.jpg) | ![ColorJitter](https://github.com/avs-abhishek123/AugStatic/blob/0f12cb0adf7b0c9bd68074f8891907067604e79f/OutputImages/ColorJitter.jpg) |
| **Downscale** | ![Input_image](https://github.com/avs-abhishek123/AugStatic/blob/c563cf249fd0f28453eb63d47634c3279b426031/input_image.jpg) | ![Downscale](https://github.com/avs-abhishek123/AugStatic/blob/0f12cb0adf7b0c9bd68074f8891907067604e79f/OutputImages/Downscale.jpg) |
| **Emboss** | ![Input_image](https://github.com/avs-abhishek123/AugStatic/blob/c563cf249fd0f28453eb63d47634c3279b426031/input_image.jpg) | ![Emboss](https://github.com/avs-abhishek123/AugStatic/blob/0f12cb0adf7b0c9bd68074f8891907067604e79f/OutputImages/Emboss.jpg) |
| **FancyPCA** | ![Input_image](https://github.com/avs-abhishek123/AugStatic/blob/c563cf249fd0f28453eb63d47634c3279b426031/input_image.jpg) | ![FancyPCA](https://github.com/avs-abhishek123/AugStatic/blob/0f12cb0adf7b0c9bd68074f8891907067604e79f/OutputImages/FancyPCA.jpg) |
| **GaussNoise** | ![Input_image](https://github.com/avs-abhishek123/AugStatic/blob/c563cf249fd0f28453eb63d47634c3279b426031/input_image.jpg) | ![GaussNoise](https://github.com/avs-abhishek123/AugStatic/blob/0f12cb0adf7b0c9bd68074f8891907067604e79f/OutputImages/GaussNoise.jpg) |
| **GaussianBlur** | ![Input_image](https://github.com/avs-abhishek123/AugStatic/blob/c563cf249fd0f28453eb63d47634c3279b426031/input_image.jpg) | ![GaussianBlur](https://github.com/avs-abhishek123/AugStatic/blob/0f12cb0adf7b0c9bd68074f8891907067604e79f/OutputImages/GaussianBlur.jpg) |
| **GlassBlur** | ![Input_image](https://github.com/avs-abhishek123/AugStatic/blob/c563cf249fd0f28453eb63d47634c3279b426031/input_image.jpg) | ![GlassBlur](https://github.com/avs-abhishek123/AugStatic/blob/0f12cb0adf7b0c9bd68074f8891907067604e79f/OutputImages/GlassBlur.jpg) |
| **HueSaturationValue** | ![Input_image](https://github.com/avs-abhishek123/AugStatic/blob/c563cf249fd0f28453eb63d47634c3279b426031/input_image.jpg) | ![HueSaturationValue](https://github.com/avs-abhishek123/AugStatic/blob/0f12cb0adf7b0c9bd68074f8891907067604e79f/OutputImages/HueSaturationValue.jpg) |
| **ISONoise** | ![Input_image](https://github.com/avs-abhishek123/AugStatic/blob/c563cf249fd0f28453eb63d47634c3279b426031/input_image.jpg) | ![ISONoise](https://github.com/avs-abhishek123/AugStatic/blob/0f12cb0adf7b0c9bd68074f8891907067604e79f/OutputImages/ISONoise.jpg) |
| **InvertImg** | ![Input_image](https://github.com/avs-abhishek123/AugStatic/blob/c563cf249fd0f28453eb63d47634c3279b426031/input_image.jpg) | ![InvertImg](https://github.com/avs-abhishek123/AugStatic/blob/0f12cb0adf7b0c9bd68074f8891907067604e79f/OutputImages/InvertImg.jpg) |
| **MedianBlur** | ![Input_image](https://github.com/avs-abhishek123/AugStatic/blob/c563cf249fd0f28453eb63d47634c3279b426031/input_image.jpg) | ![MedianBlur](https://github.com/avs-abhishek123/AugStatic/blob/0f12cb0adf7b0c9bd68074f8891907067604e79f/OutputImages/MedianBlur.jpg) |
| **MotionBlur** | ![Input_image](https://github.com/avs-abhishek123/AugStatic/blob/c563cf249fd0f28453eb63d47634c3279b426031/input_image.jpg) | ![MotionBlur](https://github.com/avs-abhishek123/AugStatic/blob/0f12cb0adf7b0c9bd68074f8891907067604e79f/OutputImages/MotionBlur.jpg) |
| **MultiplicativeNoise** | ![Input_image](https://github.com/avs-abhishek123/AugStatic/blob/c563cf249fd0f28453eb63d47634c3279b426031/input_image.jpg) | ![MultiplicativeNoise](https://github.com/avs-abhishek123/AugStatic/blob/0f12cb0adf7b0c9bd68074f8891907067604e79f/OutputImages/MultiplicativeNoise.jpg) |
| **Posterize** | ![Input_image](https://github.com/avs-abhishek123/AugStatic/blob/c563cf249fd0f28453eb63d47634c3279b426031/input_image.jpg) | ![Posterize](https://github.com/avs-abhishek123/AugStatic/blob/0f12cb0adf7b0c9bd68074f8891907067604e79f/OutputImages/Posterize.jpg) |
| **RGBShift** | ![Input_image](https://github.com/avs-abhishek123/AugStatic/blob/c563cf249fd0f28453eb63d47634c3279b426031/input_image.jpg) | ![RGBShift](https://github.com/avs-abhishek123/AugStatic/blob/0f12cb0adf7b0c9bd68074f8891907067604e79f/OutputImages/RGBShift.jpg) |
| **Sharpen** | ![Input_image](https://github.com/avs-abhishek123/AugStatic/blob/c563cf249fd0f28453eb63d47634c3279b426031/input_image.jpg) | ![Sharpen](https://github.com/avs-abhishek123/AugStatic/blob/0f12cb0adf7b0c9bd68074f8891907067604e79f/OutputImages/Sharpen.jpg) |
| **Solarize** | ![Input_image](https://github.com/avs-abhishek123/AugStatic/blob/c563cf249fd0f28453eb63d47634c3279b426031/input_image.jpg) | ![Solarize](https://github.com/avs-abhishek123/AugStatic/blob/0f12cb0adf7b0c9bd68074f8891907067604e79f/OutputImages/Solarize.jpg) |
| **Superpixels** | ![Input_image](https://github.com/avs-abhishek123/AugStatic/blob/c563cf249fd0f28453eb63d47634c3279b426031/input_image.jpg) | ![Superpixels](https://github.com/avs-abhishek123/AugStatic/blob/0f12cb0adf7b0c9bd68074f8891907067604e79f/OutputImages/Superpixels.jpg) |
| **ToGray** | ![Input_image](https://github.com/avs-abhishek123/AugStatic/blob/c563cf249fd0f28453eb63d47634c3279b426031/input_image.jpg) | ![ToGray](https://github.com/avs-abhishek123/AugStatic/blob/0f12cb0adf7b0c9bd68074f8891907067604e79f/OutputImages/ToGray.jpg) |
| **ToSepia** | ![Input_image](https://github.com/avs-abhishek123/AugStatic/blob/c563cf249fd0f28453eb63d47634c3279b426031/input_image.jpg) | ![ToSepia](https://github.com/avs-abhishek123/AugStatic/blob/0f12cb0adf7b0c9bd68074f8891907067604e79f/OutputImages/ToSepia.jpg) |
| **VerticalFlip** | ![Input_image](https://github.com/avs-abhishek123/AugStatic/blob/c563cf249fd0f28453eb63d47634c3279b426031/input_image.jpg) | ![VerticalFlip](https://github.com/avs-abhishek123/AugStatic/blob/0f12cb0adf7b0c9bd68074f8891907067604e79f/OutputImages/VerticalFlip.jpg) |
| **HorizontalFlip** | ![Input_image](https://github.com/avs-abhishek123/AugStatic/blob/c563cf249fd0f28453eb63d47634c3279b426031/input_image.jpg) | ![HorizontalFlip](https://github.com/avs-abhishek123/AugStatic/blob/0f12cb0adf7b0c9bd68074f8891907067604e79f/OutputImages/HorizontalFlip.jpg) |
| **Transpose** | ![Input_image](https://github.com/avs-abhishek123/AugStatic/blob/c563cf249fd0f28453eb63d47634c3279b426031/input_image.jpg) | ![Transpose](https://github.com/avs-abhishek123/AugStatic/blob/0f12cb0adf7b0c9bd68074f8891907067604e79f/OutputImages/Transpose.jpg) |
| **OpticalDistortion** | ![Input_image](https://github.com/avs-abhishek123/AugStatic/blob/c563cf249fd0f28453eb63d47634c3279b426031/input_image.jpg) | ![OpticalDistortion](https://github.com/avs-abhishek123/AugStatic/blob/0f12cb0adf7b0c9bd68074f8891907067604e79f/OutputImages/OpticalDistortion.jpg) |
| **GridDistortion** | ![Input_image](https://github.com/avs-abhishek123/AugStatic/blob/c563cf249fd0f28453eb63d47634c3279b426031/input_image.jpg) | ![GridDistortion](https://github.com/avs-abhishek123/AugStatic/blob/0f12cb0adf7b0c9bd68074f8891907067604e79f/OutputImages/GridDistortion.jpg) |
| **JpegCompression** | ![Input_image](https://github.com/avs-abhishek123/AugStatic/blob/c563cf249fd0f28453eb63d47634c3279b426031/input_image.jpg) | ![JpegCompression](https://github.com/avs-abhishek123/AugStatic/blob/0f12cb0adf7b0c9bd68074f8891907067604e79f/OutputImages/JpegCompression.jpg) |
| **Cutout** | ![Input_image](https://github.com/avs-abhishek123/AugStatic/blob/c563cf249fd0f28453eb63d47634c3279b426031/input_image.jpg) | ![Cutout](https://github.com/avs-abhishek123/AugStatic/blob/0f12cb0adf7b0c9bd68074f8891907067604e79f/OutputImages/Cutout.jpg) |
| **CoarseDropout** | ![Input_image](https://github.com/avs-abhishek123/AugStatic/blob/c563cf249fd0f28453eb63d47634c3279b426031/input_image.jpg) | ![CoarseDropout](https://github.com/avs-abhishek123/AugStatic/blob/0f12cb0adf7b0c9bd68074f8891907067604e79f/OutputImages/CoarseDropout.jpg) |
| **GridDropout** | ![Input_image](https://github.com/avs-abhishek123/AugStatic/blob/c563cf249fd0f28453eb63d47634c3279b426031/input_image.jpg) | ![GridDropout](https://github.com/avs-abhishek123/AugStatic/blob/0f12cb0adf7b0c9bd68074f8891907067604e79f/OutputImages/GridDropout.jpg) |

---

## Conclusion:

* There are multiple augmentation techniques supported by the AugStatic library that helps in developing the augmented balanced library by the iterative implementation of the augmentations on the generated dataset. 
* It takes an input of the existing dataset, majority and minority classes sample count that returns a balanced image dataset by iteratively applying the augmentations on the generated augmented images in the minority class. 
* This generator is computationally efficient and can be used for any image dataset.
* With the advancement in augmentation, there is a lot of scope in making the AugStatic library for audio, NLP, and time-series data. 

---

## To cite my paper: 
|Citing Text|
|---|
|   Allena Venkata Sai Abhishek,  Dr. Venkateswara Rao Gurrala,   "AUGMENTED BALANCED IMAGE DATASET GENERATOR USING AUGSTATIC LIBRARY", IJRAR - International Journal of Research and Analytical Reviews (IJRAR), E-ISSN 2348-1269, P- ISSN 2349-5138, Volume.9, Issue 2, Page No pp.536-543, May 2022, Available at : http://www.ijrar.org/IJRAR22B1901.pdf   |
