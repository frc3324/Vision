from freenect import sync_get_depth as get_depth, sync_get_video as get_video
import freenect
import cv
import cv2
import numpy as np

def doloop():
    global depth, rgb
    while True:
        # Get a fresh frame
        (depth,_), (rgb,_) = get_depth(), get_video(0, freenect.VIDEO_IR_8BIT)
	#(depth,_), (rgb,_) = get_depth(), get_video()
        
        # Build a two panel color image
        #d3 = np.dstack((depth,depth,depth)).astype(np.uint8)
        #da = np.hstack((d3,rgb))
        
        # Simple Downsample
        #cv.ShowImage('both', cv.fromarray(np.array(da[::2,::2,::-1])))
	#rgbarray = cv.fromarray(np.array(rgb))
	rgbarray = np.array(rgb)
	#gray = cv2.cvtColor(rgbarray, cv2.COLOR_BGR2GRAY)
	#blurred = cv.CloneMat(rgbarray)
	#sobeled = cv.CreateMat(rgbarray.rows, rgbarray.cols, cv.CV_32F)
	
	#cv.Sobel(rgbarray, sobeled, 1, 1)
	#sobeled = cv2.Sobel(blurred, cv.CV_32F, 1, 1)
	_, threshed = cv2.threshold(rgbarray, 250, 255, cv2.THRESH_BINARY)
	blurred = cv2.GaussianBlur(threshed, (255, 255), 0)
	
	#cv.Sobel(rgbarray2, sobeled, 0, 0, 1)
        cv.ShowImage('both', cv.fromarray(threshed))
        cv.WaitKey(5)
        
doloop()
