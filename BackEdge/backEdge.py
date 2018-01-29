import cv2
import numpy as np 
from matplotlib import pyplot as plt
import math
from numpy import linalg 
def rearedge(x,y) :
	vec1 = x[1]-x[0]
	vec2 = y[1]-y[0]
	z  = np.dot(vec1, vec2)
	z = z/(linalg.norm(vec1)*linalg.norm(vec2))
	if (z< math.cos((np.pi/2))) :
		return True
	else :
		return False


img1 = cv2.imread('FRM1.jpg')
img2 = cv2.imread('FRM2.jpg')
# Converting images to grayscale
gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
# A 3x3 operating kernel
kernel = np.ones((3, 3), np.uint8)

# Opening operation to remove possible noises
opening1 = cv2.morphologyEx(gray1, cv2.MORPH_OPEN, kernel, iterations = 2) 
opening2 = cv2.morphologyEx(gray2, cv2.MORPH_OPEN, kernel, iterations = 2)
# dilating the gray scale image
dilate1 = cv2.dilate(opening1, kernel, iterations=1) 
dilate2 = cv2.dilate(opening2, kernel, iterations=1) 

# conversion to binary
ret, bin1 = cv2.threshold(dilate1, 0, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU) 
ret, bin2 = cv2.threshold(dilate2, 0, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU) 

#Edge detection using Canny's Algorithm. It requires two thresholds
thrs1 = 923
thrs2 = 2962
edge1 = cv2.Canny(bin1, thrs1, thrs2, apertureSize=5)
edge2 = cv2.Canny(bin2, thrs1, thrs2, apertureSize=5)
# Finding all possible contours in the image
img_c1, contours1, hierarchy1 = cv2.findContours(edge1, cv2.RETR_TREE, cv2.CHAIN_APPROX_TC89_KCOS)
img_c2, contours2, hierarchy2 = cv2.findContours(edge2, cv2.RETR_TREE, cv2.CHAIN_APPROX_TC89_KCOS)

# Centers of all contours
centers1 = np.zeros((len(contours1),2))
centers2 = np.zeros((len(contours2),2))


i=0
for cnt1, cnt2 in zip(contours1, contours2):
	M1 = cv2.moments(cnt1)
	M2 = cv2.moments(cnt2)
	centers1[i] = (int(M1["m10"] / M1["m00"]), int(M1["m01"] / M1["m00"]))
	centers2[i] = (int(M2["m10"] / M2["m00"]), int(M2["m01"] / M2["m00"]))
	i = i + 1

centr_i1 = centers2[0]
centr_i0 = centers1[0]
# Points in contour 1 in image 2
pt2 = np.float32([t[-1] for t in contours2[0]])
Di = np.array([centr_i0,centr_i1])
back_edges = []
print(len(pt2))
j = 0
for i in range(0,len(pt2)-1):
	Vi = np.array([centr_i1,pt2[i]])
	z = rearedge(Di,Vi)
	if z:
		back_edges.append(pt2[i])
		print(i)
		point = back_edges[j]
		j = j + 1
		cv2.circle(img2, (point[0],point[1]), 6, (255,0,0),3)
	else :
		print('False')


for i in range(0,len(back_edges)-1):
	st = back_edges[i]
	end = back_edges[i+1]
	cv2.line(img2, (st[0],st[1]), (end[0], end[1]), (0, 0, 255), 4)

cv2.imshow('',img2)
ch = cv2.waitKey(0)
# cv2.circle(frame, center, 5, (0,255,255),-1)