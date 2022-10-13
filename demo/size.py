import cv2,os
import numpy as np

array_of_img = [] # this if for store all of the image data
# this function is for read image,the input is directory name
def read_directory(directory_name):
    for filename in os.listdir(directory_name):
        print(filename)
        img = cv2.imread(directory_name+"/"+filename)
        img = cv2.resize(img, (512, 512))
        img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        cv2.imwrite(f'./predict/Chimei/9/{filename}', img)
read_directory(r'C:\Users\a0907\Desktop\fracture\demo\predict\Chimei\9')