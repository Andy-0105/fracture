import cv2
from PIL import Image
import numpy as np
img = cv2.imread('101106434107.jpg', 0)
ret, th1 = cv2.threshold(img, 0.5, 255, cv2.THRESH_BINARY)
th1=cv2.resize(th1,(512,512))
cv2.imwrite('1234.jpg',th1)