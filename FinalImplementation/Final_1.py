import cv2
import numpy as np
from matplotlib import pyplot as plt
import math
from numpy import linalg 
from common import draw_str

# The drawline function draws straight strokes
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
	

# rearedge takes two vectors and detects if the vector falls in the theta rear view i.e. if it is a rear edge point or not.
def rearedge(x,y) :
	theta = np.pi
	vec1 = x[1]-x[0]
	vec2 = y[1]-y[0]
	z  = np.dot(vec1, vec2)
	z = z/(linalg.norm(vec1)*linalg.norm(vec2))
	if (z< math.cos((theta/2))) :
		return True
	else :
		return False


def CatmullRomSpline(P0, P1, P2, P3, nPoints=100):
	"""
	P0, P1, P2, and P3 should be (x,y) point pairs that define the Catmull-Rom spline.
	nPoints is the number of points to include in this curve segment.
	"""
	# Convert the points to np so that we can do array multiplication
	P0, P1, P2, P3 = map(np.array, [P0, P1, P2, P3])

	# Calculate t0 to t4
	alpha = 0.5
	def tj(ti, Pi, Pj):
		xi, yi = Pi
		xj, yj = Pj
		return ( ( (xj-xi)**2 + (yj-yi)**2 )**0.5 )**alpha + ti

	t0 = 0
	t1 = tj(t0, P0, P1)
	t2 = tj(t1, P1, P2)
	t3 = tj(t2, P2, P3)

	# Only calculate points between P1 and P2
	t = np.linspace(t1,t2,nPoints)

	# Reshape so that we can multiply by the points P0 to P3
	# and get a point for each value of t.
	t = t.reshape(len(t),1)

	A1 = (t1-t)/(t1-t0)*P0 + (t-t0)/(t1-t0)*P1
	A2 = (t2-t)/(t2-t1)*P1 + (t-t1)/(t2-t1)*P2
	A3 = (t3-t)/(t3-t2)*P2 + (t-t2)/(t3-t2)*P3

	B1 = (t2-t)/(t2-t0)*A1 + (t-t0)/(t2-t0)*A2
	B2 = (t3-t)/(t3-t1)*A2 + (t-t1)/(t3-t1)*A3

	C  = (t2-t)/(t2-t1)*B1 + (t-t1)/(t2-t1)*B2
	return C

def CatmullRomChain(P):
	"""
	Calculate Catmull Rom for a chain of points and return the combined curve.
	"""
	sz = len(P)

	# The curve C will contain an array of (x,y) points.
	C = []
	for i in range(sz-3):
		c = CatmullRomSpline(P[i], P[i+1], P[i+2], P[i+3])
		C.extend(c)

	return C


imname = 'multi_'
z = 0
plot_points = []
center_points = []
number_points = []
for r in range(1,4):
	print(r)
	name = imname+str(r)+'.jpg'
	img1 = cv2.imread(name)
	name = imname+str(r+1)+'.jpg'	
	img2 = cv2.imread(name)

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

	cv2.imshow('',img_1)
	cv2.waitKey(0)
	cv2.imshow('',img_2)
	cv2.waitKey(0)


	gray_1 = cv2.cvtColor(img_1,cv2.COLOR_BGR2GRAY)
	gray_2 = cv2.cvtColor(img_2,cv2.COLOR_BGR2GRAY)

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
	img2_c1 = img2.copy()
	img2_c = img2.copy()
	i=0
	k=0
	num_pts = np.zeros(len(centers2))
	h = 0
	for c1 ,c0 in zip(centers2,centers1):
		Di[0] = c0
		Di[1] = c1
		pt1 = pts_1[i]
		pt2 = pts_2[i]
		i = i+1
		m = 1
		for j in range(0,len(pt1)-1):
			Vi = np.array([c1,pt2[j]])
			z = rearedge(Di,Vi)
			if z:
				cv2.circle(img2_c1,(pt2[j,0],pt2[j,1]),3,(0,0,255),-1)
				back_edges.append(pt2[j])
				end_point.append(pt1[j])
				m = m + 1
				point = back_edges[k]
				k = k + 1
		num_pts[h] = m
		h = h + 1
	cv2.imshow('',img2_c1)
	cv2.waitKey(0)
	if r is 1:
		plot_points.append(end_point)
		plot_points.append(back_edges)
		center_points.append(centers1)
		center_points.append(centers2)
		z = z + 1
	else:
		plot_points[z] = end_point
		plot_points.append(back_edges)
		center_points[z] = centers1
		center_points.append(centers2)
		z = z + 1


final_img = img2.copy()
cpoints = np.asarray(center_points)
tracks = np.zeros((len(plot_points),len(plot_points[0]),2))
for i in range(0,len(plot_points)):
	t = np.asarray(plot_points[i])
	tracks[i,:len(tracks[0])-1] = t[:len(tracks[0])-1]

points = np.zeros((len(tracks)+2,2),np.int32)
for i in range(20,len(tracks[0]),30):
	points[1] = tracks[0,i]
	points[0] = points[1] - 1
	c = i
	z = 0
	tots = num_pts[0]
	while c > tots:
		z = z + 1
		tots = tots + num_pts[z]
	center = cpoints[:,z]

	for m in range(2,len(tracks) + 1):
		diff = center[m-2] - points[m-1]
		points[m] = center[m-1] + diff
			
	f = 100
	points[-1] = points[-2] + 1
	j = 0
	ptss = np.zeros((len(points)/2+1,2))
	for i in range(1,len(points)-1,2):
		ptss[j+1] = points[i]
		j = j + 1
		f = f + 50
	ptss[-1] = ptss[-2] + 1
	c = CatmullRomChain(ptss)
	x,y = zip(*c)
	x_ = np.asarray(x,np.int32)
	y_ = np.asarray(y,np.int32)
	# cv2.circle
	for i,j in zip(range(0,len(x)),range(0,len(y))):
  		cv2.circle(final_img,(x_[i],y_[j]),2,(220,220,220),-1)

cv2.imshow('',final_img)
cv2.waitKey(0)

cv2.destroyAllWindows()