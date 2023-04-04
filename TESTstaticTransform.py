from staticTransforms import StaticTransformDataGenerator as AugStatic
import PIL
import cv2
from PIL import Image

augmentationtype=input()
img=cv2.imread("AugmentationLibrary/StaticTransforms/dog.jpg")
image=cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

an_object= AugStatic(image,1)

# return number of sample, store them in the folder

print("AugStatic."+augmentationtype+"()")


augmented_image=eval("an_object."+augmentationtype+"()")


Image.fromarray(augmented_image).save("C:/Users/abhis/OneDrive/Desktop/Mtech-Project/Demo4review/Output_Images/"+augmentationtype+".jpg")

