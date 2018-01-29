import cv2
import numpy as np


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
        # cv2.rectangle(image, refPt[0],refPt[1],(0,255,0),1)




cap = cv2.VideoCapture('eva-walle-1.mp4')

# take first frame of the video
ret,frame = cap.read()



#construct the argument paser and parse the arguments
image = frame.copy()
clone = image.copy()

cv2.namedWindow("image")
cv2.setMouseCallback("image",click_and_crop)

while True:
    cv2.imshow("image",image)
    key = cv2.waitKey(0) & 0xFF

    if key == ord('n'):
        ret, image = cap.read()
        clone = image.copy()

    #if 'r' is pressed reset the cropping region
    elif key == ord("r"):
        image = clone.copy()

    #if 'c' is pressed, break from loop
    elif key == ord('c'):
        break

    elif key == ord('x'):
        break

if len(refPt) == 2:
    roi = clone[refPt[0][1]:refPt[1][1], refPt[0][0]:refPt[1][0]]

cv2.imshow

r = refPt[0][1]
c = refPt[0][0]
h = refPt[1][1] - refPt[0][1]
w = refPt[1][0] - refPt[0][0]

track_window = (c,r,w,h)

hsv_roi =  cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
mask = cv2.inRange(hsv_roi, np.array((0., 60.,32.)), np.array((180.,255.,255.)))
roi_hist = cv2.calcHist([hsv_roi],[0],mask,[180],[0,180])
cv2.normalize(roi_hist,roi_hist,0,255,cv2.NORM_MINMAX)

# Setup the termination criteria, either 10 iteration or move by atleast 1 pt
term_crit = ( cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 3 )

object_data = []

while(1):
    ret ,frame = cap.read()
    if ret == True:
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        dst = cv2.calcBackProject([hsv],[0],roi_hist,[0,180],1)
        # x,y,w,h = track_window
        # img2 = cv2.rectangle(frame, (x,y), (x+w,y+h), 255,2)

        # apply meanshift to get the new location
        ret, track_window = cv2.meanShift(dst, track_window, term_crit)
        # Draw it on image
        x,y,w,h = track_window
        object_data.append([x,y,w,h])
        # cv2.imshow('',roin)
        # cv2.waitKey(0)
        img2 = frame.copy()
        img2 = cv2.rectangle(img2, (x,y), (x+w,y+h), 255,2)
        cv2.imshow('img2',img2)

        k = cv2.waitKey(0)
        if k == 27:
            break
        elif k == ord('n'):
            continue
    else:
        break

cv2.destroyAllWindows()
cap.release()

print(len(object_data))
img = frame.copy()

cv2.imshow('',img)
cv2.waitKey(0)