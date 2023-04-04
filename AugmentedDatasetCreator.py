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
import requests
from io import BytesIO
import math
from typing import Any, Callable,Dict, List, Optional, Sequence, Tuple, Union
import glob
import matplotlib.pyplot as plt
import shutil 
import torch
import albumentations as A
from functools import wraps

from staticTransforms import *
from staticTransforms import StaticTransformDataGenerator as AugStatic

class AugDataCreator():
    def __init__(self,MajorityImage_Dir,MinorityImage_Dir,finalMinorityImageDir,AugType,minclass,majorityclass,minorityclass_Size,majorityclass_Size):

        self.MajorityImage_Dir=MajorityImage_Dir
        self.MinorityImage_Dir=MinorityImage_Dir
        self.finalMinorityImageDir=finalMinorityImageDir
        self.AugType=AugType
        self.num_sample=1
        self.minclass=minclass
        self.majorityclass=majorityclass
        self.minorityclass_Size=minorityclass_Size
        self.majorityclass_Size=majorityclass_Size
        # just make a dataset for the input string given to us for the static augmentation applied on our dataset
        self.OriginalMajorityClassfileStore()
        self.OriginalMinorityClassfileStore()
        self.TransformList()
        
    
    def TransformList(self):
        self.allsupportedTransformList=[]
        lst_train=[]
        text_file_path="C:/Users/HP/Desktop/Final Project Documents/Mtech-Project/Demo4review/supportedAugmentations.txt"
        with open(text_file_path) as f:
            lines = f.readlines()
            for line in lines:
                ele=line.strip('\n')
                lst_train.append(ele)
        self.allsupportedTransformList=lst_train
                   
    def OriginalMajorityClassfileStore(self):
        dst=self.finalMinorityImageDir
        destination=dst+"/"+self.majorityclass+"/"
        os.mkdir(destination)        
        
        for (root, dirs, files) in os.walk(self.MajorityImage_Dir):
            for filename in files:    
                #print(filename)
                source=self.MajorityImage_Dir+"/"+filename
                

                #final_storing_path4originalImage =self.finalMinorityImageDir+"/"+filename

                #OriginalImage2beCopied=Image.fromarray(img)
                #OriginalImage2beCopied.save(final_storing_path4originalImage)
                shutil.copy(source, destination)
                #print("source",source)
                #print("destination",destination)
                
            
            self.originalCopiedMajorityImageList=files

            print("Majority class files copied successfully.")

    def OriginalMinorityClassfileStore(self):
        destination=self.finalMinorityImageDir+"/"+self.minclass+"/"
        os.mkdir(destination)
        for (root, dirs, files) in os.walk(self.MinorityImage_Dir):
            for filename in files:    
                #print(filename)
                source=self.MinorityImage_Dir+"/"+filename
                #destination=self.finalMinorityImageDir+"/"
                #final_storing_path4originalImage =self.finalMinorityImageDir+"/"+filename

                #OriginalImage2beCopied=Image.fromarray(img)
                #OriginalImage2beCopied.save(final_storing_path4originalImage)
                shutil.copy(source, destination)
                #print("source",source)
                #print("destination",destination)            
            self.originalCopiedMinoityImageList=files
            

            print("Minority class files copied successfully.")

    def augmentassigner(self,img2):    
        
        for augmentationtype in self.allsupportedTransformList:
            if self.AugType==augmentationtype:
                augm_obj=AugStatic(img2,1)
                aug_img = eval("augm_obj."+augmentationtype+"()")
                # REFER
                # https://java2blog.com/python-string-to-function/
                # aug_img=augm               
                #aug_img=AugStatic.eval(AugType)(self, img2)
            else:
                continue
        print("The new augmented dataset has this augmentation :",self.AugType)
        return aug_img


    def singleAugmented_imageCreator(self):
        #for loop on the imagefilenames and then read it and then create a copy of each image,  
        newID=0
        new_augmented_fileName_list=[]
        count=self.minorityclass_Size
        # path, dirs, files = next(os.walk("/usr/lib"))
        # file_count = len(files)
        new_augmented_fileName_list_final=[]
        #a=count
        """
            insert the number of examples you need in the end of balancing
            put 
                count=250 if ur unbalanced set has 250 & uwant make it 2K
                count<500
        """        
        """
            insert the number of examples you need in the end of balancing
            put 
                count=500 if ur unbalanced set has 500 & uwant make it 2K
                count<1000
        """
        total_of_min_copied_and_1stAugImgs=2*count
        newAugmentedMinorityClasspath=self.finalMinorityImageDir+"/"+self.minclass
        #os.mkdir(newAugmentedMinorityClasspath)
        while count<self.majorityclass_Size :
            if count<total_of_min_copied_and_1stAugImgs:
                MinorityImage_Dir=self.MinorityImage_Dir
            else:
                MinorityImage_Dir=newAugmentedMinorityClasspath
            for (root, dirs, files) in os.walk(MinorityImage_Dir):
                for filename in files:    
                    print(filename)
                    
                    path=MinorityImage_Dir+"/"+filename
                    print(path)
                    img_BGR=cv2.imread(path)
                    img = cv2.cvtColor(img_BGR, cv2.COLOR_BGR2RGB)
                    #self.singleAugmented_imageCreator()
                    # copying image to another image object
                    img2 = img.copy()
                    augmented_Image=self.augmentassigner(img2)

                    # we store this in a the respective cass of a dataset
                    new_Augmented_Image_name=self.augmentedImageSaver(augmented_Image,newID)
                    #break
                    newID+=1
                    #storing all the augmented filenames in a list
                    #new_augmented_fileName_list.append(filename)
                    new_augmented_fileName_list.append(new_Augmented_Image_name)
                    count+=1
                    print(count)
                    new_augmented_fileName_list_final+=new_augmented_fileName_list
                    if count>=self.majorityclass_Size:
                        break
                    else:
                        continue
        newDirFilesList=self.originalCopiedMinoityImageList+ new_augmented_fileName_list_final     

        return count
    

    def augmentedImageSaver(self,augmented_Image,newID):
        new_Augmented_Image_name=self.AugType+str(newID)+".jpg"
        #print("we are saving file with this name", new_Augmented_Image_name)
        newAugmentedMinorityClasspath=self.finalMinorityImageDir+"/"+self.minclass
        #os.mkdir(newAugmentedMinorityClasspath)
        final_storing_path =newAugmentedMinorityClasspath + "/" +new_Augmented_Image_name
        #print("we are saving file with this path", final_storing_path)
        



        augmentedFinal_Image=Image.fromarray(augmented_Image)
        augmentedFinal_Image.save(final_storing_path)

        return new_Augmented_Image_name


    

    # for 1 image u did, just run a loop somehwere and read all the directory files