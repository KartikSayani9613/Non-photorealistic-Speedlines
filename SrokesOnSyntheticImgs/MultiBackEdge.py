import cv2
import numpy as np 
from matplotlib import pyplot as plt
import math
from numpy import linalg 
from common import draw_str

def drawline (frame,start,end):
	x1, y1 = start
	x2, y2 = end
	dx = abs(x2 - x1)
	dy = abs(y2 - y1)
	#initial radius assumed
	slope = dy/float(dx)
	in_rad = 3

	x, y = x1, y1

	if slope > 1:
		print('yes')
		dx, dy = dy, dx
		x, y = y, x
		x1, y1 = y1, x1
		x2, y2 = y2, x2

	p = 2*dy - dx

	points = []
	point = [x,y]
	points.append(point)

	for i in range(2,int(dx)):
		if p > 0:
			y = y + 1 if y < y2 else y - 1
			p = p + 2*(dy - dx)
		else:
			p = p + 2*dy

		x = x + 1 if x < x2 else x - 1
		point = [x,y]
		points.append(point)

	points= np.asarray(points)
	if slope > 1:
		points[:,[0,1]] = points[:,[1,0]]
	num = len(points)
	step= in_rad/num
	rad = in_rad
	for i in range (num):    
		td=points[i]
		x=td[0]
		y=td[1]
		cv2.circle(frame, (int(x),int(y)) , int(rad), (220, 220, 220), -1)
		rad = rad-step
    
	# cv2.imshow("image",frame)
	# cv2.waitKey(0)
	# cv2.destroyAllWindows()
	#top three lines are for drawing and viewing the image 
	return frame 
	#return the image with the strokes
	#if multiple strokes are needed then add the for loop in the program code. and in the function return the points
	


def rearedge(x,y) :
	vec1 = x[1]-x[0]
	vec2 = y[1]-y[0]
	z  = np.dot(vec1, vec2)
	z = z/(linalg.norm(vec1)*linalg.norm(vec2))
	if (z< math.cos((np.pi/2))) :
		return True
	else :
		return False



img1 = cv2.imread('multi_1.jpg')
img2 = cv2.imread('multi_6.jpg')


gray1 = cv2.cvtColor(img1,cv2.COLOR_BGR2GRAY)
gray2 = cv2.cvtColor(img2,cv2.COLOR_BGR2GRAY)
ret, thresh1 = cv2.threshold(gray1,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
ret, thresh2 = cv2.threshold(gray2,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
# cv2.imshow('',thresh1)
# ch = cv2.waitKey(0)

# noise removal

kernel = np.ones((3,3),np.uint8)
# kernel[1,1] = -8
opening1 = cv2.morphologyEx(thresh1,cv2.MORPH_OPEN,kernel,iterations=1)
opening2 = cv2.morphologyEx(thresh2,cv2.MORPH_OPEN,kernel,iterations=1)
# cv2.imshow('',opening1)
# ch = cv2.waitKey(0)


# sure background
sure_bg1 = cv2.dilate(opening1,kernel,iterations=1)
sure_bg2 = cv2.dilate(opening2,kernel,iterations=1)
# cv2.imshow('',sure_bg1)
# ch = cv2.waitKey(0)
# sure foreground
dist_transform1 = cv2.distanceTransform(opening1,cv2.DIST_L2,3)
ret, sure_fg1 = cv2.threshold(dist_transform1,0.5*dist_transform1.max(),255,0)
dist_transform2 = cv2.distanceTransform(opening2,cv2.DIST_L2,3)
ret, sure_fg2 = cv2.threshold(dist_transform2,0.5*dist_transform2.max(),255,0)

# Finding unknown region
sure_fg1 = np.uint8(sure_fg1)
unknown1 = cv2.subtract(sure_bg1,sure_fg1)
sure_fg2 = np.uint8(sure_fg2)
unknown2 = cv2.subtract(sure_bg2,sure_fg2)


# Marker labelling
ret, markers1 = cv2.connectedComponents(sure_fg1)
ret, markers2 = cv2.connectedComponents(sure_fg2)

# Add one to all labels so that sure background is not 0 but 1
markers1 = markers1 + 1
markers2 = markers2 + 1

# Mark unknown region with 0
markers1[unknown1==255] = 0
markers2[unknown2==255] = 0

markers1 = cv2.watershed(img1,markers1)
markers2 = cv2.watershed(img2,markers2)
img_1 = np.zeros(img1.shape,np.uint8)
img_2 = np.zeros(img2.shape,np.uint8)
img_1[markers1== -1] = [0,0,255]
img_2[markers2== -1] = [0,0,255]


gray_1 = cv2.cvtColor(img_1,cv2.COLOR_BGR2GRAY)
gray_2 = cv2.cvtColor(img_2,cv2.COLOR_BGR2GRAY)

# img_c1, contours1, hierarchy1 = cv2.findContours(gray_1, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
img_c2, contours2, hierarchy2 = cv2.findContours(gray_2, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)


centers1 = np.zeros((len(contours2)/2,2),np.int32)
centers2 = np.zeros((len(contours2)/2,2),np.int32)

i=0
j=0
for cnt1 in contours2:
	if i%2 is 0:
		M2 = cv2.moments(cnt1)
		centers2[j] = (int(M2["m10"] / M2["m00"]), int(M2["m01"] / M2["m00"]))
		j = j+1
	i = i + 1

pts_1 = []
pts_2 = []


k=0
for i in range(0,len(contours2)):
	if i%2 is 0:
		points = np.float32([t[-1] for t in contours2[i]])
		pts_2.append(points)
		cnt = contours2[i]
		x,y,w,h= cv2.boundingRect(cnt)
		patch = gray2[y:y+h,x:x+w]
		w, h = patch.shape[::-1]
		res = cv2.matchTemplate(gray1,patch,cv2.TM_SQDIFF_NORMED)
		minVal, maxVal, minLoc, maxLoc = cv2.minMaxLoc(res)
		topLeft = minLoc
		w, h = patch.shape[::-1]
		dabba = img1[topLeft[1]-3:topLeft[1]+h+3,topLeft[0]-3:topLeft[0]+w+3]
		dabba_th = thresh1[topLeft[1]-3:topLeft[1]+h+3,topLeft[0]-3:topLeft[0]+w+3]
		dabba_op = cv2.morphologyEx(dabba_th,cv2.MORPH_OPEN,kernel,iterations=1)
		dabba_bg = cv2.dilate(dabba_op,kernel,iterations=1)
		dabba_dst = cv2.distanceTransform(dabba_op,cv2.DIST_L2,3)
		ret, dabba_fg = cv2.threshold(dabba_dst,0.5*dist_transform1.max(),255,0)
		dabba_fg = np.uint8(dabba_fg)
		dabba_un = cv2.subtract(dabba_bg,dabba_fg)
		ret, dabba_mrk = cv2.connectedComponents(dabba_fg)
		dabba_mrk = dabba_mrk + 1
		dabba_mrk[dabba_un==255] = 0
		dabba_mrk = cv2.watershed(dabba,dabba_mrk)
		dabba_1 = np.zeros(dabba.shape,np.uint8)
		dabba_1[dabba_mrk== -1] = [0,0,255]
		dabba_1g = cv2.cvtColor(dabba_1,cv2.COLOR_BGR2GRAY)
		
		ret, cont1, hier = cv2.findContours(dabba_1g, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
		M1 = cv2.moments(cont1[0])

		centers1[k] = (int(M1["m10"] / M1["m00"]) + topLeft[0] - 3, int(M1["m01"] / M1["m00"]) + topLeft[1] - 3)
		k=k+1
		points = np.float32([t[-1] for t in cont1[0]])
		for j in range(0,len(points)):
			points[j,1] = points[j,1] + topLeft[1] - 3
			points[j,0] = points[j,0] + topLeft[0] - 3
		pts_1.append(points)

Di = np.zeros((2,2),np.int32)
back_edges = []
end_point = []
img2_c = img2.copy()
i=0
k=0
for c1, c0 in zip(centers2,centers1):
	Di[0] = c0
	Di[1] = c1
	pt1 = pts_1[i]
	pt2 = pts_2[i]
	i = i+1
	for j in range(0,len(pt1)-1):
		Vi = np.array([c1,pt2[j]])
		z = rearedge(Di,Vi)
		if z:
			back_edges.append(pt2[j])
			end_point.append(pt1[j])
			# print(j)
			point = back_edges[k]
			k = k + 1



for i in range(0,len(back_edges),50):
	for j in range(0,1):
		if i + j >= len(back_edges):
			break
		start = back_edges[i+j]
		end = end_point[i+j]
		# cv2.circle(img2_c, (start[0],start[1]), 3, (255,0,0),3)
		# cv2.circle(img2_c, (end[0],end[1]), 3, (255,0,0),3)
		drawline(img2_c,start,end)

cv2.imshow('',img1)
cv2.waitKey(0)
cv2.imshow('',img2)
cv2.waitKey(0)
cv2.imshow('',img2_c)
cv2.waitKey(0)
cv2.destroyAllWindows()