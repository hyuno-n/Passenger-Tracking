import matplotlib.pylab
import matplotlib.pyplot
import  numpy as np
from sklearn.cluster import  DBSCAN
from sklearn.preprocessing import  StandardScaler
from ultralytics import  YOLO
import cv2
import  os
import itertools
import  matplotlib.pyplot as plt
import  matplotlib




image = np.zeros(shape=(600,600),dtype=np.uint8)
cv2.circle(image , center=(325,120),radius=5,color=(0,0,255),thickness=-1)
cv2.imshow("test",image)
plt.scatter(325,120 ,color = 'red')
plt.show()
cv2.waitKey(0)