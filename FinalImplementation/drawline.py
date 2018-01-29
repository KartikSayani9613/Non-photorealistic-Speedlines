import numpy as np  
import cv2
import math
from common import draw_str

def drawline (frame,start,end):
	x1, y1 = start
	x2, y2 = end
	dx = x2 - x1
	dy = y2 - y1

	#initial radius assumed
	in_rad = 3

	is_steep = abs(dy) > abs(dx)
 	
 	if is_steep:
    	x1, y1 = y1, x1
    	x2, y2 = y2, x2

    swapped = False
	if x1 > x2:
   		x1, x2 = x2, x1
    	y1, y2 = y2, y1
    	swapped = True
 	error = int(dx / 2.0)
	ystep = 1 if y1 < y2 else -1

	y = y1
	points = []

	for x in range(x1, x2 + 1):
    	coord = (y, x) if is_steep else (x, y)
    	points.append(coord)
    	error -= abs(dy)
    	if error < 0:
        	y += ystep
        	error += dx
 
   
	if swapped:
    	points.reverse()

	points= np.asarray(points)

	num = len(points)
	step= in_rad/num
	rad = in_rad
	for i in range (num):    
    	td=points[i]
    	x=td[0]
    	y=td[1]
    	cv2.circle(frame, (x,y) , int(rad), (0, 255, 0), -1)
    	rad = rad-step
    
	# cv2.imshow("image",frame)
	# cv2.waitKey(0)
	# cv2.destroyAllWindows()
	#top three lines are for drawing and viewing the image 
	return frame 
	#return the image with the strokes
	#if multiple strokes are needed then add the for loop in the program code. and in the function return the points
	