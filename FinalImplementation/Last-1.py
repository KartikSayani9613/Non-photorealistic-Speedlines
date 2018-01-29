import cv2
import argparse
import numpy as np
from matplotlib import pyplot as plt
import math
from numpy import linalg 
from common import draw_str

# This function draws the parallel straight speed-lines, input is a frame, start pixel and an end pixel. Returns the pixels that lie on the line connecting start and end pixels
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



# Generates Catmull Rom Splines between 4 points.
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


# Calls CatmullRomSplines with sets of 4 pixels from the bluprint of a track and gathers all the pixels of the track. 
# Returns the original set of bluprint pixels and the pixels that lie on the movement curve between them.
def CatmullRomChain(P):
    
    sz = len(P)

    # The curve C will contain an array of (x,y) points.
    C = []
    for i in range(sz-3):
        c = CatmullRomSpline(P[i], P[i+1], P[i+2], P[i+3])
        C.extend(c)

    return C


refPt = []
cropping = False


# Initialize the ROI on the first frame. This function uses the mouse to capture the object we intend to track.
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




ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video", required = True, help = "Path to the video")
args = vars(ap.parse_args())

cap = cv2.VideoCapture(args["video"])

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
x,y,w,h = track_window
# conversion to HSV, easy for manipulating RGB images
hsv_roi =  cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
# Create a mask of the object to be tracked
mask = cv2.inRange(hsv_roi, np.array((0., 60.,32.)), np.array((180.,255.,255.)))
# histogram (?)
roi_hist = cv2.calcHist([hsv_roi],[0],mask,[180],[0,180])
cv2.normalize(roi_hist,roi_hist,0,255,cv2.NORM_MINMAX)

# Setup the termination criteria, either 10 iteration or move by atleast 1 pt
term_crit = ( cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 3)

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
        x1,y1,w1,h1 = track_window
        if x1 is x and y1 is y:
            continue
        else:
            x = x1
            y = y1
        object_data.append([x,y])
        
        # cv2.imshow('',roin)
        # cv2.waitKey(0)
        img2 = frame.copy()
        img2 = cv2.circle(img2, (int(x),int(y)) , 3, (220, 220, 220), -1)
        img2 = cv2.rectangle(img2, (x,y), (x+w,y+h), 255,2)
        cv2.imshow('img2',img2)

        k = cv2.waitKey(0)
        if k == 27:
            break
        elif k == ord('n'):
            continue
    else:
        break


last_img = frame.copy()
cap.release()

l = len(object_data)
num_edgp = (w+h-1)*2



for i in range(0,l):
    [x,y] = object_data[i]
    m = 0
    edge_pts = np.zeros((num_edgp,2),np.int32)
    for j in range(0,h):
        edge_pts[m] = [x,y+j]
        m = m+1
    
    for j in range(0,w-1):
        edge_pts[m] = [x+j,y+h]
        m = m+1
    
    for j in range(0,h-1):
        edge_pts[m] = [x+w,y+h-j]
        m = m+1
    for j in range(0,w):
        edge_pts[m] = [x+w-j,y]
        m = m+1
    object_data[i] = edge_pts


Di = np.zeros((2,2),np.int32)
tracks = []

t = 0
for i in range(1,l):
    back_edges = []
    end_point = []
    pts1 = object_data[i-1]
    # print(pts)
    [x, y] = pts1[0]
    c0 = [int(x+w/2),int(y+h/2)]
    pts2 = object_data[i]
    [x, y] = pts2[0]
    c1 = [int(x+w/2),int(y+h/2)]
    Di[0] = c0
    Di[1] = c1
    for j in range(0,len(pts2)):
        Vi = np.array([c1,pts2[j]])
        z = rearedge(Di,Vi)
        if z:
            back_edges.append(pts2[j])
            end_point.append(pts1[j])
    back_edges = np.asarray(back_edges)
    end_point = np.asarray(end_point)
    if t is 0:
        tracks.append(end_point)
        tracks.append(back_edges)
        l = len(end_point)
        t = t+1
    else:
        tracks[t] = end_point[:l]
        tracks.append(back_edges[:l])
        t = t+1


paths = np.zeros((len(tracks),len(tracks[0]),2),np.int32)
l = len(tracks[0])
for i in range(0,len(tracks)):
    p = tracks[i]
    if l != len(p):
        l = len(p)
    paths[i,0:l] = p



points = np.zeros((len(tracks)+2,2),np.int32)
work_img = last_img.copy()
for i in range(11,len(tracks[0]),30):#Change the last parameter here, which is currently 30, to vary the density of speed-lines. Higher the value, lower the number of speed-lines.
    p = paths[:,i]
    points[1:-1] = p
    points[0] = points[1] - 1
    points[-1] = points[-2] + 1
    print(points)
    for o in range(0,int(len(points)/2)):
        points[o] = points[-o]
    print(points)
    c = CatmullRomChain(points)
    x,y = zip(*c)
    x_ = np.asarray(x,np.int32)
    y_ = np.asarray(y,np.int32)
    # cv2.circle
    tmp = work_img.copy()
    q = 0
    v = 0
    print(len(x))
    for i,j in zip(range(0,len(x)),range(0,len(y))):
        i = int(i + v)
        j = int(j + v)
        v = v*1.2
        if i > len(x):
            break
        cv2.circle(work_img,(x_[i],y_[j]),1,(150+q,50+q,150+q),0) #Make colour changes here
        q = q + 130/len(x)
    cv2.imshow('',work_img)
    key = cv2.waitKey(0)
    if key == ord('n'):
        work_img = tmp.copy()
    else:
        last_img = work_img.copy()


    


    
cv2.destroyAllWindows()
cv2.imshow('',last_img)
cv2.imwrite('abc.jpg',last_img)
cv2.waitKey(0)

