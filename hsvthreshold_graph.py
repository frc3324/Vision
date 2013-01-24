import sys
sys.path.append('/users/sujit/libs/libfreenect/wrappers/python/')

import freenect
import math
import cv2
import numpy as np
import scipy.signal
import pylab

use_webcam = False
if use_webcam:
	c = cv2.VideoCapture(0)
	c.open(0)

plot_data = True
if plot_data:
	pylab.ion()
	graph, = pylab.plot([0], [0])
	graph2, = pylab.plot([0], [0])

"""params = {
	"huel": 0,
	"hueh": 255,
	"satl": 0,
	"sath": 255,
	"luml": 0,
	"lumh": 255
}"""

#webcam with salmon paper
"""params = {
	"hueh": 19,
	"huel": 7,
	"luml": 103,
	"lumh": 255,
	"sath": 206,
	"satl": 71
}"""

#jpg with salmon paper
params = {
	"hueh": 20,
	"huel": 0,
	"luml": 120,
	"lumh": 255,
	"sath": 255,
	"satl": 0
}

"""params = {
	"hueh": 74,
	"huel": 52,
	"luml": 10,
	"lumh": 83,
	"sath": 255,
	"satl": 119
}"""

width = 640
height = 480
tolerance = 0.1
#imgpath = "wpitest.jpg"
#imgpath = "salmon3.jpg"
#imgpath = "testout.png"
imgpath = ""

comps = None

def getimg_webcam():
	_, img = c.read()
	return img

def getimg_irkinect():
	raw_data, _ = freenect.sync_get_video(0, freenect.VIDEO_IR_8BIT)
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
	lowerb = np.array([params['huel'], params['satl'], params['luml']])
	upperb = np.array([params['hueh'], params['sath'], params['lumh']])
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
	return hsvthreshold(cv2.GaussianBlur(img, (21, 21), 1))
	#return hsvthreshold(cv2.medianBlur(img, 21))

def contours(_img):
	img = blurred(_img)
	#img = hsvthreshold(_img)
	ret = cv2.merge([img, img, img])
	cons, hier = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

	#cv2.drawContours(ret, np.array(cons), -1, np.array([255.0, 255.0, 0.0]))
	for i in range(len(cons)):
		if cv2.contourArea(cons[i]) >= 1000:
			#cv2.drawContours(ret, np.array(cons), i, np.array([0.0, 0.0, 255.0]))
			pts = []
			for pt in cons[i]:
				pts.append(pt[0])
				ret[pt[0][1]][pt[0][0]] = np.array([255.0, 0.0, 0.0])

			corn = corners(pts)
			if len(corn) == 5 or True:
				for x, y in corn:
					cv2.circle(ret, (int(x), int(y)), 10, np.array([0.0, 0.0, 255.0]), -1)
			"""corn = houghcorners(pts, img)
			if corn != None:
				for c in corn[0]:
					x1 = int(c[0])
					y1 = int(c[1])
					x2 = int(c[2])
					y2 = int(c[3])
					cv2.line(ret, (x1, y1), (x2, y2), np.array([0.0, 0.0, 255.0]), 3)"""
	
	return ret

def lderiv(pts):
	#pts = [f(x - k), f(x - h), f(x)]
	k = pts[2][0] - pts[0][0]
	h = pts[1][0] - pts[0][0]
	s = pts[1][1] - h ** 2 * pts[0][1] / k ** 2 + (h ** 2 / k ** 2 - 1) * pts[2][1]
	return s / (h ** 2 / k - h)

def rderiv(pts):
	#pts = [f(x), f(x + h), f(x + k)]
	k = pts[2][0] - pts[0][0]
	h = pts[1][0] - pts[0][0]
	s = pts[1][1] - h ** 2 * pts[2][1] / k ** 2 + (h ** 2 / k ** 2 - 1) * pts[0][1]
	return s / (h - h ** 2 / k)

def movingaverage(arr, window):
	radius = (window - 1) / 2
	weights = np.array([1.0] * window) / window
	arr2 = np.concatenate([arr[-radius:], arr, arr[:radius]])
	return np.convolve(arr2, weights, 'valid')

def houghcorners(pts, img):
	img2 = np.zeros_like(img)
	for pt in pts:
		img2[pt[1]][pt[0]] = 1
	segs = cv2.HoughLinesP(img2, 3, 0.001, len(pts) / 4 / 10, maxLineGap=len(pts))
	return segs

def corners(pts):
	pts = np.array(pts)
	cx = pts[:,0].astype(np.double).sum() / len(pts)
	cy = pts[:,1].astype(np.double).sum() / len(pts)

	deltas = pts.astype(np.double) - [[cx, cy]] * len(pts)
	#dists = movingaverage(np.sqrt(deltas[:,0] ** 2 + deltas[:,1] ** 2), 31)
	dists = np.sqrt(deltas[:,0] ** 2 + deltas[:,1] ** 2)
	angles = np.arctan2(deltas[:,0], deltas[:,1])
	stacked = np.column_stack([deltas, dists, angles])

	sorted = stacked[np.lexsort(stacked.transpose())]

	ret = [[cx, cy]]
	derivs = []
	derivs2 = []
	l = len(sorted)

	prev = np.roll(sorted, -1, axis=0)
	next = np.roll(sorted, 1, axis=0)
	"""ld = (sorted[:,2] - prev[:,2]) / (sorted[:,3] - prev[:,3])
	rd = (sorted[:,2] - next[:,2]) / (sorted[:,3] - next[:,3])
	r = sorted[:,2]
	theta = sorted[:,3]
	s = np.sin(theta)
	c = np.cos(theta)
	d1 = (next[:,2] - prev[:,2]) / (next[:,3] - prev[:,3])"""
	#derivs = np.abs(ld - rd)
	#derivs = scipy.signal.medfilt(np.arctan2(d1 * s + sorted[:,1], d1 * c - sorted[:,0]), 1)
	#xderiv = (next[:,1] - prev[:,1]) / (next[:,3] - prev[:,3])
	#yderiv = (next[:,0] - prev[:,0]) / (next[:,3] - prev[:,3])
	nnext = np.roll(sorted, -10, axis=0)
	dx = nnext[:,1] - sorted[:,1]
	dy = nnext[:,0] - sorted[:,0]
	window = len(pts) / 8
	if window % 2 == 0:
		window += 1
	slopes = np.arctan2(dy, dx)
	derivs = scipy.signal.medfilt(slopes, window)
	"""pderiv = np.roll(derivs, -1)
	nderiv = np.roll(derivs, 1)
	d2 = np.abs((nderiv - pderiv) / (next[:,3] - prev[:,3]))"""
	if plot_data:
		xdata = sorted[:,3]
		ydata = sorted[:,2]
		ydata2 = scipy.signal.medfilt(slopes, 21) / np.abs(slopes).max() * np.abs(ydata).max()
		graph.set_xdata(xdata)
		graph.set_ydata(ydata)
		graph.axes.set_xlim(xdata.min(), xdata.max())
		graph.axes.set_ylim(ydata.min(), ydata.max())
		graph2.set_xdata(xdata)
		graph2.set_ydata(ydata2)
		graph2.axes.set_xlim(xdata.min(), xdata.max())
		graph2.axes.set_ylim(ydata2.min(), ydata2.max())
		pylab.draw()

	for i in range(l):
		perfect = True
		"""prev = (i - 30) % l
		pprev = (i - 2) % l
		next = (i + 30) % l
		nnext = (i + 2) % l
		#ld = lderiv([sorted[pprev][3:1:-1], sorted[prev][3:1:-1], sorted[i][3:1:-1]])
		#rd = rderiv([sorted[i][3:1:-1], sorted[next][3:1:-1], sorted[nnext][3:1:-1]])
		ld = (sorted[i][2] - sorted[prev][2]) / (sorted[i][3] - sorted[prev][3])
		rd = (sorted[i][2] - sorted[next][2]) / (sorted[i][3] - sorted[next][3])"""
		#print "%f\t%f\t%f\t%f\t%f" % (sorted[i][3], sorted[i][2], slopes[i], derivs[i], d2[i])
		#print "%f\t%f\t%f" % (sorted[i][3], slopes[i], derivs[i])
		#print "%d\t%d" % (pts[i][0], pts[i][1])
		for k in range(1, 10):
			next = (i + k) % l
			prev = (i - k) % l
			#if not (d2[i] > d2[next] and d2[i] > d2[prev]):
			if not (sorted[i][2] > sorted[next][2] and sorted[i][2] > sorted[prev][2]):
				perfect = False
				break
				#print "%f, %f, %f" % (sorted[prev][2], sorted[i][2], sorted[next][2])"""
		if perfect:
			ret.append(sorted[i][:2] + [cx, cy])

	"""for quad in range(4):
		best = quad * l / 4
		for i in range(quad * l / 4 + 1, (quad + 1) * l / 4):
			if sorted[i][2] > sorted[best][2]:
				best = i
		ret.append(sorted[best][:2] + [cx, cy])"""
	
	return ret

def loop(processimg):
	if not use_webcam and imgpath == "":
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
	cv2.destroyAllWindows()

loop(contours)
