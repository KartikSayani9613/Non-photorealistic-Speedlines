Instructions on Running:

Run by command

/For Curved speed lines
python3 Last-1.py -v "PathToVideo"
/For parallel straight lines
python3 Last-2.py -v "PathToVideo"

The first frame of the video opens up.
Select an object by using the mousepad. Make sure selection is done from top towards bottom and left to right. Also, the selection should occur inside the body of the object, not surrounding it. 
To select the object press 'c'. To reset the selection, press 'r'. To select object over the next frame,press 'n'. Once the object is selected using 'c', press 'x' to proceed.

Now, keep pressing 'n' till the desired length of video is played frame by frame. Once the desired number of frames are covered, press 'esc'.
Now one by one, the speed lines appear. The speed-lines may go haywire because of Catmull Rom Splines. Once a speed-line has appeared, press 'y' if you wish to keep, if not, press 'n'. Once all the lines have been drawn, an output image 'abc.jpg' is written.



Implementation:

Once the ROI is selected by the user, We convert it to HSV format. It provides additional function to work on. Then, we construct a mask of the object. We use this mask and the histogram of ROI to track the object with meanShift. meanShift returns the positions of the object tracked. We record these positions. These positons are in form x,y,w,h in which x,y is the left most point and w,h are the eidth and height of a box surrounding the object. We use the points on the edges of this box as the edges of the object. We then determine the 'back_edges'. These backedges are collectively stored as the 'tracks' of the object. We use these tracks for catmull Rom Splines/ straight parallel lines.
Colour of the lines can be manipulated, but needs to be changed in the code. 'drawlines' is the function where this can be done.
Same is true for curved lines. A comment is added where the changes can be made.
The number of the lines can be changed	too. Change a parameter in the very last for loop. A comment has been added to find it.


A couple of additional codes have been added, just in case.