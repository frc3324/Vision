import sys
sys.path.append('/users/sujit/libs/libfreenect/wrappers/python/')

import freenect
import cv2
import numpy as np
import math

use_webcam = False
if use_webcam:
	c = cv2.VideoCapture(0)
	c.open(0)

params = {
	"huel": 0,
	"hueh": 255,
	"satl": 0,
	"sath": 255,
	"luml": 40,
	"lumh": 255
}

"""params = {
	"huel": 0,
	"hueh": 255,
	"satl": 0,
	"sath": 255,
	"luml": 0,
	"lumh": 255
}"""

"""params = {
	"hueh": 19,
	"huel": 7,
	"luml": 103,
	"lumh": 255,
	"sath": 206,
	"satl": 71
}"""

"""params = {
	"hueh": 74,
	"huel": 52,
	"luml": 29,
	"lumh": 83,
	"sath": 255,
	"satl": 119
}"""

width = 640
height = 480
tolerance = 0.1
#imgpath = "wpitest.jpg"
imgpath = ""

comps = None

cam = np.array([
	[594.21480358642339,		0.0,			339.30546187516956],
	[0.0,				591.04092248505947,	242.73843891390746],
	[0.0,				0.0,			1.0]
])

dis = np.array([-0.26389095690190378, 0.99983033880181316, -0.00076323952014484080, 0.0050337278410637169, -1.3056496956879815])

target = np.array([
	[-15.5,	-7.25,	0],
	[-15.5,	7.25,	0],
	[15.5,	7.25,	0],
	[15.5,	-7.25,	0]
])

def getimg_webcam():
	_, img = c.read()
	return img

def getimg_irkinect():
	raw_data, _ = freenect.sync_get_video(0, freenect.VIDEO_IR_8BIT)
	#depth, _ = freenect.sync_get_depth()
	#print depth.scn
	return cv2.cvtColor(np.array(raw_data), cv2.COLOR_GRAY2BGR)

def getimg():
	if use_webcam:
		return getimg_webcam()
	else:
		return getimg_irkinect()

def donothing(img):
	return img

def hsvthreshold(img):
	hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
	lowerb = np.array([params['huel'], params['satl'], params['luml']],np.uint8)
	upperb = np.array([params['hueh'], params['sath'], params['lumh']],np.uint8)
	threshed = cv2.inRange(hsv, lowerb, upperb)
	return threshed

def brightnessthreshold(img):
	_, ret = cv2.threshold(img, 200, 255, cv2.THRESH_TOZERO)
	return ret

def onchange(prop):
	def ret(x):
		params[prop] = x
		cv2.setTrackbarPos(prop, 'processing', x)
	
	return ret

def blurred(img):
	return hsvthreshold(cv2.GaussianBlur(img, (11, 11), 5))

def contours(img):
	#img = blurred(img)
	#img = cv2.erode(hsvthreshold(img), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
	img = hsvthreshold(img)
	ret = cv2.merge([img, img, img])
	cons, hier = cv2.findContours(img, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
	targets = []

	#cv2.drawContours(ret, np.array(cons), -1, np.array([255.0, 255.0, 0.0]), -1)
	for i in range(len(cons)):
		if cv2.contourArea(cons[i]) >= 1000:
			#print len(cons[i])
			#print len(cv2.convexHull(cons[i]))
			#cons[i] = cv2.convexHull(cons[i])
			cv2.drawContours(ret, np.array(cons), i, np.array([255.0, 0.0, 0.0]), 1)
			pts = []
			for pt in cons[i]:
				pts.append(pt[0])
				#ret[pt[0][1]][pt[0][0]] = np.array([255.0, 0.0, 0.0])
			
			corn = corners(pts, ret)
			if len(corn) == 4:
				rvec, tvec = cv2.solvePnP(target,np.array(corn[:]),cam,dis)
				print np.linalg.norm(tvec)
				#target = pts[:,1].astype(np.double).sum() / len(pts)
				for x, y in corn:
					cv2.circle(ret, (int(x), int(y)), 5, np.array([0.0, 0.0, 255.0]), -1)
	
	return ret

def corners(pts, img):
	pts = np.array(pts)
	cx = pts[:,0].astype(np.double).sum() / len(pts)
	cy = pts[:,1].astype(np.double).sum() / len(pts)

	deltas = pts.astype(np.double) - [[cx, cy]] * len(pts)
	dists = np.sqrt(deltas[:,0] ** 2 + deltas[:,1] ** 2)
	angles = np.arctan2(deltas[:,0], deltas[:,1])
	stacked = np.column_stack([deltas, dists, angles])

	sorted = stacked[np.lexsort(stacked.transpose())]

	ret = []
	l = len(sorted)
	sides = []
	side = []
	for i in range(l):
		perfect = True
		#print "%f\t%f" % (sorted[i][3], sorted[i][2])
		side.append(sorted[i][:2] + [cx, cy])
		for k in range(1, 15):
			next = (i + k) % l
			prev = (i - k) % l
			if not (sorted[i][2] > sorted[next][2] and sorted[i][2] > sorted[prev][2]):
				perfect = False
				#print "%f, %f, %f" % (sorted[prev][2], sorted[i][2], sorted[next][2])
		if perfect:
			ret.append(sorted[i][:2] + [cx, cy])
			sides.append(side[:])
			side = []
		if len(ret) > 4:
			ret = []
			break
	if len(sides) > 0:
		sides[0].extend(side)
	
	if len(ret) == 4:
		for i in range(len(sides)):
			x1 = sides[i][0][0]
			y1 = sides[i][0][1]
			x2 = sides[i][len(sides[i]) - 1][0]
			y2 = sides[i][len(sides[i]) - 1][1]
			for n in range(len(sides[i])):
				x = sides[i][n][0]
				y = sides[i][n][1]
				
				a = (y2 - y1) / (x2 - x1)
				b = -1
				c = -a * x1 + y1
				
				dist = math.fabs(a * x + b * y + c)/math.sqrt(a**2 + b**2)
				
				if dist >= 100:
					ret = []
				img[sides[i][n][1]][sides[i][n][0]] = np.array([(i) * 100 % 255, (i + 1) * 100 % 255, (i+2) * 100 % 255])
	
	"""for quad in range(4):
		best = quad * l / 4
		for i in range(quad * l / 4 + 1, (quad + 1) * l / 4):
			if sorted[i][2] > sorted[best][2]:
				best = i
		ret.append(sorted[best][:2] + [cx, cy])"""
	
	return ret

def loop(processimg):
	if not use_webcam:
		ctx = freenect.init()
		dev = freenect.open_device(ctx, 0)
		freenect.set_tilt_degs(dev, 10)
		freenect.close_device(dev)
		
	
	cv2.namedWindow('processing')
	for k, v in params.iteritems():
		cv2.createTrackbar(k, 'processing', v, 255, onchange(k))
	
	runonce = True
	while runonce:
		#runonce = False
		if imgpath != "":
			img = cv2.imread(imgpath)
		else:
			img = getimg()

		cv2.imshow('processing', cv2.resize(processimg(img), (width, height)))
		char = cv2.waitKey(10)
		if char == 27:
			break
		elif char == ord('p'):
			for k, v in params.iteritems():
				print "%s: %d" % (k, v)
		#freenect.set_tilt_degs(dev, pid)
	cv2.destroyAllWindows()

loop(contours)
