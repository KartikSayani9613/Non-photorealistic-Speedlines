import argparse
import cv2
import numpy as np 
from matplotlib import pyplot as plt
import math
from numpy import linalg 
from common import draw_str


# initialize the list of reference points and boolean indicating
# whether cropping is being performed or not
refPt = []
cropping = False

def click_and_crop(event, x, y, flags, param):
	# grab references to the global variables
	global refPt, cropping


	# if the left mouse button was clicked, record the starting
	# (x, y) coordinates and indicate that cropping is being
	# performed


	if event == cv2.EVENT_LBUTTONDOWN:
		refPt = [(x, y)]
		cropping = True

	# check to see if the left mouse button was released
	elif event == cv2.EVENT_LBUTTONUP:
		# record the ending (x, y) coordinates and indicate that
		# the cropping operation is finished
		refPt.append((x, y))
		cropping = False

		#Draw a rectangle
		cv2.rectangle(image, refPt[0],refPt[1],(0,255,0),2)


#construct the argument paser and parse the arguments

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required = True, help = "Path to the image")
args = vars(ap.parse_args())

#load image
image = cv2.imread(args["image"])
clone = image.copy()

cv2.namedWindow("image")
cv2.setMouseCallback("image",click_and_crop)

while True:
	cv2.imshow("image",image)
	key = cv2.waitKey(1) & 0xFF

	#if 'r' is pressed reset the cropping region
	if key == ord("r"):
		image = clone.copy()

	#if 'c' is pressed, break from loop
	elif key == ord('c'):
		break

if len(refPt) == 2:
	roi = clone[refPt[0][1]:refPt[1][1], refPt[0][0]:refPt[1][0]]
	cv2.imshow("ROI",roi)
	cv2.waitKey(0)

cv2.destroyAllWindows()