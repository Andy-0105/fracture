import cv2
import os
array_of_img = []
def read_directory(directory_name):
    count=0
    for filename in os.listdir(r"./"+directory_name):
        img = cv2.imread(directory_name + "/" + filename)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        cv2.imwrite(f"./123/{count}.jpg",img)
        count+=1
read_directory("./test_8")