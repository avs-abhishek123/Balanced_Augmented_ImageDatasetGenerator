import os
import pprint as pp
import numpy as np
import cv2
import os
import json
import random
import PIL
import urllib
from PIL import Image
from torchvision import transforms
from PIL import Image as PILImage
from PIL import ImageDraw as PILImageDraw
import math
from typing import Any, Callable,Dict, List, Optional, Sequence, Tuple, Union
import glob
import matplotlib.pyplot as plt
import shutil 
import os 
import base64
import torch
import albumentations as A
from functools import wraps


from AugmentedDatasetCreator import AugDataCreator 

# buildings is constant for all datasets
#sea is our minority class

# Normal is constant for all datasets
# Pneumonia is our minority class

#It is read from original unbalanced dataset

transformQuantity=input("Enter the type of transform quantity you want : ")

majorityclass_Size=2000
classname="sea"
minclass="sea"
majorityclass="buildings"

# ub_dataset=[
#     #"2_1",
#     #"4_1",
#     #"8_1",
#     "16_1",
#     #"20_1"
#     #"Unbalanced_2_1_DecreasedB",
#     #"Unbalanced_4_1_DecreasedB",
#     #"Unbalanced_8_1_DecreasedB",
#     #"Unbalanced_16_1_DecreasedB",
#     #"Unbalanced_20_1_DecreasedB"
#     ]
# for ubratio in ub_dataset:
MajorityImage_Dir="C:/Users/HP/Desktop/Final Project Documents/Mtech-Project/Datasets/unbalancedDataset/train/buildings"
MinorityImage_Dir="C:/Users/HP/Desktop/Final Project Documents/Mtech-Project/Datasets/unbalancedDataset/train/sea"
finalMinorityImageDir="C:/Users/HP/Desktop/Final Project Documents/Mtech-Project/Demo4review/Output_Images"


minorityclass_Size=majorityclass_Size//2
# if ubratio=="2_1":
#     minorityclass_Size=majorityclass_Size//2
# elif ubratio=="4_1":
#     minorityclass_Size=majorityclass_Size//4
# elif ubratio=="8_1":
#     minorityclass_Size=majorityclass_Size//8
# elif ubratio=="16_1":
#     minorityclass_Size=majorityclass_Size//16
# elif ubratio=="20_1":
#     minorityclass_Size=majorityclass_Size//20
# else:
#     break     


AugType_list=[]

if transformQuantity=="single":
    AugType=input("Give the type of augmentation you want for all the images of the minority class : ")
    AugType_list=[AugType]
elif transformQuantity=="bulk":
    AugType_list=["Blur",
    "CLAHE",
    "ChannelDropout",
    "ChannelShuffle",
    "ColorJitter",
    "Downscale",
    "Emboss",
    "FancyPCA",
    "GaussNoise",
    "GaussianBlur",
    "GlassBlur",
    "HueSaturationValue",
    "ISONoise",
    "InvertImg",
    "MedianBlur",
    "MotionBlur",
    "MultiplicativeNoise",
    "Posterize",
    "RGBShift",
    "Sharpen",
    "Solarize",
    "Superpixels",
    "ToGray",
    "ToSepia",
    "VerticalFlip",
    "HorizontalFlip",
    "Transpose",
    "OpticalDistortion",
    "GridDistortion",
    "JpegCompression",
    "Cutout",
    "CoarseDropout",
    "GridDropout"
    ]
else:
    print("Enter single or bulk for generating respective augmented datasets")


for AugType in AugType_list:
    #AugType=input("Give the type of augmentation you want for all the images of the minority class : ")

    newMinorityDirectoryname=classname+AugType
    print("newMinorityDirectoryname",newMinorityDirectoryname)
    x = os.path.join(finalMinorityImageDir, newMinorityDirectoryname)
    #print(x)
    newAugmentedMinorityClasspath=x
    print("newAugmentedMinorityClasspath",newAugmentedMinorityClasspath)

    if not os.path.exists(newAugmentedMinorityClasspath):
        print("THE DIRECTORY DONT EXIST SO MAKING A NEW ONE")
        os.mkdir(newAugmentedMinorityClasspath)
        annotation_path="/mc2/SaiAbhishek/Improve_Model_Accuracy_using_Aug_Approaches/Single_Static_Augmentation/chest_X_ray_pneumonia/Augmented_Dataset/COCO4Each/"+newMinorityDirectoryname+".json"


    elif not os.path.exists(newAugmentedMinorityClasspath):
        print("THE DIRECTORY EXIST")
        annotation_path="/mc2/SaiAbhishek/Improve_Model_Accuracy_using_Aug_Approaches/Single_Static_Augmentation/chest_X_ray_pneumonia/Augmented_Dataset/COCO4Each/"+newMinorityDirectoryname+"1.json"
        
        os.mkdir(newAugmentedMinorityClasspath+"1")

    elif not os.path.exists(newAugmentedMinorityClasspath+"1"):
        print("THE DIRECTORY EXIST")
        annotation_path="/mc2/SaiAbhishek/Improve_Model_Accuracy_using_Aug_Approaches/Single_Static_Augmentation/chest_X_ray_pneumonia/Augmented_Dataset/COCO4Each/"+newMinorityDirectoryname+"2.json"
        
        os.mkdir(newAugmentedMinorityClasspath+"2")

    else:
        print("THE DIRECTORY EXIST")
        annotation_path="/mc2/SaiAbhishek/Improve_Model_Accuracy_using_Aug_Approaches/Single_Static_Augmentation/chest_X_ray_pneumonia/Augmented_Dataset/COCO4Each/"+newMinorityDirectoryname+"3.json"
        
        os.mkdir(newAugmentedMinorityClasspath+"3")


    # add more static augmentations

    an_object = AugDataCreator(MajorityImage_Dir,MinorityImage_Dir,newAugmentedMinorityClasspath,AugType,minclass,majorityclass,minorityclass_Size,majorityclass_Size)

    newfilenameList=an_object.singleAugmented_imageCreator()
    print(newfilenameList)
    #print(len(newfilenameList))
    #augmented_dict=dict()
    #for newfilename in newfilenameList:
    #    augmented_dict[newfilename]=2


    #file_pointer=open(annotation_path,'w+')
    #json.dump(augmented_dict, file_pointer, indent = 4)
    #file_pointer.close()  


