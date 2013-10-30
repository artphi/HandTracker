#	coding: utf-8

"""-----------------------------------------------------------------------------
HandTracker.py
Hand detection with OpenCV

Autors: 	Aude Piguet
			Olivier Francillon
			Raphael Santos
Due date:	18.11.2013
Release:

Notes:
Please refer to the french document "HDproject - Rapport"

Prerequists:
	* Python 2.7
	* OpenCV 2.4.6.1

Usage
On linux:
	* $ python HDproject.py [config [path]][debug [face, mask, all]]
	* To stop the program, please use the 'q' key, then on the terminal choose if you 
	  want to save or not the modifications
	* 

Files description
	* HDproject.py: Main Class
	* faceDetection.py: Face detection class using haar. Threaded in main class
	* .config: Config file
	* haar: folder containing some haar XML

/--------------------------------------------------------------/

This program is free software; you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation; either version 2 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License along
with this program; if not, write to the Free Software Foundation, Inc.,
51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.
/--------------------------------------------------------------/


-----------------------------------------------------------------------------""" tracking with OpenCV and Python
"""

import cv2
import sys
import pickle as pkl
import numpy as np
from numpy import sqrt, arccos, rad2deg
from faceDetection import FaceDetection as fd
import argparse as ap

class HandTracker(object):

	############################################
	# Initialization
	def __init__(self):

		# Parameters
		self.CAMERA_TIMEOUT = 500
		self.HAND_LIMIT_AREA = 10000
		self.IMAGE_WIDTH = 640

		# Initialisation
		self.im_background = None
		self.im_thread = np.zeros((60,60),np.uint8)
		self.im_face = np.zeros((60,60),np.uint8)

		# Parse arguments
		parser = ap.ArgumentParser(description='Hand tracking with OpenCV and Python')
		parser.add_argument('-d', '--debug',
			metavar='DEBUG_TYPE',
			choices=['mask', 'face'],
			help='enables debugging')
		parser.add_argument('-c', '--config',
			metavar='CONFIG_FILE',
			default='config.cfg',
			help='path to config file')
		args = parser.parse_args()
		self.debugType = args.debug
		self.configFile = args.config

		# Read config
		try:
			self.configParams = pkl.load(open(self.configFile, "r"))
		except:
			print "ERROR: Configuration file (", self.configFile, ") not found."
			sys.exit(1)

		self.th_Y_min = self.configParams['th_Y_min']
		self.th_Y_max = self.configParams['th_Y_max']
		self.th_CR_min = self.configParams['th_CR_min']
		self.th_CR_max = self.configParams['th_CR_max']
		self.th_CB_min = self.configParams['th_CB_min']
		self.th_CB_max = self.configParams['th_CB_max']
		self.blur = self.configParams['blur']

		# Create control window
		cv2.namedWindow('Control Panel')
		cv2.createTrackbar('Y_min', 'Control Panel', self.th_Y_min, 255, self.onChange_th_Y_min)
		cv2.createTrackbar('Y_max', 'Control Panel', self.th_Y_max, 255, self.onChange_th_Y_max)
		cv2.createTrackbar('CR_min', 'Control Panel', self.th_CR_min, 255, self.onChange_th_CR_min)
		cv2.createTrackbar('CR_max', 'Control Panel', self.th_CR_max, 255, self.onChange_th_CR_max)
		cv2.createTrackbar('CB_min', 'Control Panel', self.th_CB_min, 255, self.onChange_th_CB_min)
		cv2.createTrackbar('CB_max', 'Control Panel', self.th_CB_max, 255, self.onChange_th_CB_max)
		cv2.createTrackbar('blur', 'Control Panel', self.blur, 100, self.onChange_blur)
		cv2.resizeWindow('Control Panel', 800, 300)
		cv2.moveWindow('Control Panel', 50, 50)

		# Camera capture
		try:
			self.capture = cv2.VideoCapture(0)
		except Exception as detail:
			print "ERROR: Camera initialization failed (", detail, ")"
			sys.exit(1)
		cv2.waitKey(200)	# Wait for camera (Apple iSight bug?)
	
		# Initialize face detection thread
		self.fd_thread = fd(self)

		# Start app
		self.main()

	############################################
	# Main function
	def main(self):

		# Start face detection thread
		self.fd_thread.start()

		# Main loop
		run = True
		tries, maxTries = 0, self.CAMERA_TIMEOUT
		nbFingers = 0
		while(run):
			# Get the video frame
			try:
				ret, self.im_orig = self.capture.read()
			except Exception as detail:
				print "ERROR: Video capture exception (", detail, ")"
				sys.exit(1)
			
			# No return for >= maxTries
			if not ret and tries >= maxTries:
				print "ERROR: Video capture timeout"
				sys.exit(1)
			# No return for < maxTries
			elif not ret:
				tries += 1
				continue
			# With return
			else:
				tries = 0

			# Image flip & resize if width > IMAGE_WIDTH
			self.im_orig = cv2.flip(self.im_orig, 1)
			size = self.im_orig.shape
			if size[1] > self.IMAGE_WIDTH:
				new_width = self.IMAGE_WIDTH
				new_height = size[0] / (size[1] / new_width)
				self.im_orig = cv2.resize(self.im_orig, (new_width, new_height))
				self.im_face = cv2.resize(self.im_face, (new_width, new_height))

			# Copy for face detetction
			try:
				self.im_thread = self.im_orig.copy()
			except Exception as detail:
				print "ERROR: image copy error (", detail, ")"

			# if Background is not set, set it
			if self.im_background == None:
				self.im_background = np.copy(self.im_orig)

			self.im_output = np.copy(self.im_orig)

			# Background removal
			#self.skin = self.backgroundRemoval(self.im_orig, self.im_background)

			# Skin extraction
			self.skin = self.skinExtraction(self.im_orig)

			# Face removal
			try:
				self.skin = cv2.add(self.skin, self.im_face)
			except:
				pass
			# bitwise inversion
			self.skin = cv2.bitwise_not(self.skin)

			# Contours detection
			contours = cv2.findContours(self.skin, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)[0]

			for cnt in contours:
				if cv2.contourArea(cnt) > self.HAND_LIMIT_AREA:

					self.drawHandCenter(cv2.moments(cnt))
					hull = cv2.convexHull(cnt, returnPoints = False)
					nbFingers = self.drawFingers(cnt, hull)
				#cv2.drawContours(self.im_output, [cnt], -1, (0, 255, 0), -1)

			cv2.putText(self.im_output, "fingers = " + `nbFingers`, (10, 20), cv2.FONT_HERSHEY_PLAIN, 1, [255, 255, 255])
			cv2.imshow('output', np.concatenate((self.im_orig, self.im_output), 1))
			#cv2.moveWindow('output', 50, 500)

			keyPressed = cv2.waitKey(3)
			# q > quit
			if (keyPressed == 113 or keyPressed == ord('q')):
				run = False
			# space > new background
			elif (keyPressed == 32):
				self.im_background = np.copy(self.im_orig)

		# Stop face capture thread
		self.fd_thread.stop()
		# Destroy windows
		cv2.destroyAllWindows()
		# Release video capture
		self.capture.release()
		#self.save()


	############################################
	# YCrCb Conversion
	def yCrCbConversion(self, image):
		try:
			return cv2.cvtColor(image, cv2.COLOR_BGR2YCR_CB)
		except Exception as detail:
			print "ERROR: YCrCb conversion (", detail, ")"


	############################################
	# Channels splitting
	def channelsSplit(self,image):
		return cv2.split(image)


	############################################
	# Thresholding for min and max values
	def thresholding(self, image, min, max):
		val, mask = cv2.threshold(image, max, 255, cv2.THRESH_BINARY)
		val, mask_inv = cv2.threshold(image, min, 255, cv2.THRESH_BINARY_INV)
		return cv2.add(mask, mask_inv)


	############################################
	# Background removal
	def backgroundRemoval(self, image, background):
		try:
			# Blur image and background
			im_blur = cv2.blur(image, (self.blur, self.blur))
			im_bg_blur = cv2.blur(background, (self.blur, self.blur))

			# Convert to YCrCb
			im_YCrCb = cv2.cvtColor(im_blur, cv2.COLOR_BGR2YCR_CB)
			im_bg_YCrCb = cv2.cvtColor(im_bg_blur, cv2.COLOR_BGR2YCR_CB)

			# Substract background from image
			diff = cv2.absdiff(im_YCrCb, im_bg_YCrCb)

			# Split Y Cr Cb channels
			channels = cv2.split(diff)

			# Threshold on each channel
			y_img = self.thresholding(channels[0], self.th_Y_min, self.th_Y_max)
			cr_img = self.thresholding(channels[1], self.th_CR_min, self.th_CR_max)
			cb_img = self.thresholding(channels[2], self.th_CB_min, self.th_CB_max)

			# Define kernel for morophology edit
			kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (17, 17))

			# Dilate & erode each layer
			y_img = cv2.erode(cv2.dilate(y_img, kernel), kernel)
			cr_img = cv2.erode(cv2.dilate(cr_img, kernel), kernel)
			cb_img = cv2.erode(cv2.dilate(cb_img, kernel), kernel)

			# Sum channels together
			sum = cv2.add(y_img, cr_img, cb_img)

			# Define new kernel for morphology edit
			#kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (17, 17))

			# Dilate & erode	
			sum = cv2.erode(cv2.dilate(sum, kernel), kernel)

			# Bitwise not
			sum = cv2.bitwise_not(sum)

			#total = cv2.merge([y_img, cr_img, cb_img])
			#fg = cv2.add(self.im_orig, sum)
			if self.debugType == 'mask':
				cv2.imshow('debug', sum)
		except Exception as detail:
			print "ERROR: Background removal (", detail, ")"
			sys.exit(1)
		return sum


	############################################
	# Skin Extraction
	def skinExtraction(self, image):
		try:
			# Convert to YCrCb
			im_YCrCb = cv2.cvtColor(image, cv2.COLOR_BGR2YCR_CB)

			# Split channels
			channels = cv2.split(im_YCrCb)

			# Threshold
			y_img = self.thresholding(channels[0], self.th_Y_min, self.th_Y_max)
			cr_img = self.thresholding(channels[1], self.th_CR_min, self.th_CR_max)
			cb_img = self.thresholding(channels[2], self.th_CB_min, self.th_CB_max)

			# Assemble
			skin = cv2.add(cr_img, cb_img)

			# Blur
			#skin = cv2.blur(skin, (self.blur, self.blur))

			if self.debugType == 'mask':
				cv2.imshow('debug', skin)
				#cv2.moveWindow('debug', 900, 50)
		except Exception as detail:
			print "ERROR: Skin extraction (", detail, ")"
			sys.exit(1)
		return skin


	############################################
	# Angle calculation
	def angleCalc(self, cent, rect1, rect2):
		v1 = (rect1[0] - cent[0], rect1[1] - cent[1])
		v2 = (rect2[0] - cent[0], rect2[1] - cent[1])
		dist = lambda a:sqrt(a[0] ** 2 + a[1] ** 2)
		angle = arccos((sum(map(lambda a, b:a*b, v1, v2))) / (dist(v1) * dist(v2)))
		angle = abs(rad2deg(angle))
		return angle


	############################################
	# Draw hand center
	def drawHandCenter(self, moments):
		try:
			centroid_x = int(moments['m10']/moments['m00'])
			centroid_y = int(moments['m01']/moments['m00'])
			cv2.circle(self.im_output, (centroid_x, centroid_y), 15, (0,0,255), 1)
			cv2.circle(self.im_output, (centroid_x, centroid_y), 10, (0,0,255), 1)
			cv2.circle(self.im_output, (centroid_x, centroid_y), 5, (0,0,255), -1)
		except Exception as detail:
			print "ERROR: Hand center drawing (", detail, ")"


	############################################
	# Draw Fingers
	def drawFingers(self, cnt, hull):
		angles = []
		defects = cv2.convexityDefects(cnt, hull)
		if defects != None:
			try:
				for i in range(defects.shape[0]):
					s,e,f,d = defects[i,0]
					if d > 5000:
						start, end, far = tuple(cnt[s][0]), tuple(cnt[e][0]), tuple(cnt[f][0])
						angleDefect = self.angleCalc(far, start, end)
						if angleDefect < 90:
							angles.append(angleDefect)
							cv2.circle(self.im_output, far, 5, [0,0,255], -1)
							cv2.line(self.im_output, start, far, [0,255,0], 2)
							cv2.line(self.im_output, far, end, [0,255,0], 2)
			except Exception as detail:
				print "ERROR: Fingers drawing: (", detail, ")"
		return len(angles)+1


	############################################
	# Trackbars onChange functions
	def onChange_th_Y_min(self, value):
		self.th_Y_min = value
	
	def onChange_th_Y_max(self, value):
		self.th_Y_max = value

	def onChange_th_CR_min(self, value):
		self.th_CR_min = value

	def onChange_th_CR_max(self, value):
		self.th_CR_max = value

	def onChange_th_CB_min(self, value):
		self.th_CB_min = value

	def onChange_th_CB_max(self, value):
		self.th_CB_max = value

	def onChange_blur(self, value):
		self.blur = value


	############################################
	# Trackbars onChange functions
	def save(self):
		self.configParams["th_Y_min"] = self.th_Y_min
		self.configParams["th_Y_max"] = self.th_Y_max
		self.configParams["th_CR_min"] = self.th_CR_min
		self.configParams["th_CR_max"] = self.th_CR_max
		self.configParams["th_CB_min"] = self.th_CB_min
		self.configParams["th_CB_max"] = self.th_CB_max
		self.configParams["blur"] = self.blur
		pkl.dump(self.configParams, open(self.configFile, "w"))


if __name__ == '__main__':
	app = HandTracker()