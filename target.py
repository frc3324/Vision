import freenect
import cv2
import numpy as np
import math

#Constants
width = 640
height = 480

#Define kinect IR camera calibration matrix
cam = np.array([
	[594.21480358642339,		0.0,			339.30546187516956],
	[0.0,				591.04092248505947,	242.73843891390746],
	[0.0,				0.0,			1.0]
])

#Define kinect IR camera distortion coefficients
dis = np.array([-0.26389095690190378, 0.99983033880181316, -0.00076323952014484080, 0.0050337278410637169, -1.3056496956879815])

#Define target points in real space
target = np.array([
	[-15.5,	-7.25,	0],
	[-15.5,	7.25,	0],
	[15.5,	7.25,	0],
	[15.5,	-7.25,	0]
])

#Function to get IR data from kinect
def get_ir():
	raw_data, _ = freenect.sync_get_video(0, freenect.VIDEO_IR_8BIT)
	return np.array(raw_data)

def threshold(img):
	return cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 9, -10)
	#return cv2.inRange(img, np.array([100]), np.array([255]))

def corners(pts):
	pts = np.array(pts)
	
	'''cx = pts[:,0].astype(np.double).sum() / len(pts)
	cy = pts[:,1].astype(np.double).sum() / len(pts)'''
	return cv2.cornerHarris(pts, 2, 3, 0.0)

def process(img):
	return corners(threshold(img))

def loop():
	cv2.namedWindow('target')
	
	while True:
		cv2.imshow('target', cv2.resize(process(get_ir()), (width, height)))
		if cv2.waitKey(10) == 27:
			break
	
	cv2.destroyAllWindows()

loop()
