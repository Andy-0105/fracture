import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

ans = Image.open('111.jpg')
ans = ans.convert('1')
plt.imsave('123.png',ans,cmap='gray')
plt.imshow(ans,cmap='gray')
plt.show()
