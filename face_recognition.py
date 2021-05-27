from imutils.video import VideoStream
from imutils.video import FPS
import numpy as np
from numpy import random
import argparse
import imutils
import pickle
import time
import cv2
import os

args = {
	'confidence': 0.4,  # filter out weak detections
	'IPcam': False,  # True to stream with IP cam, False to use the Webcam
	'IP': 'https://192.168.0.165:8080/video',  # IP for IPCam if True
	'detector': 'models',  # path to OpenCV's deep learning face detector
	'detector_proto': 'deploy.prototxt',  # face detector based on a res net
	'detector_model': 'res10_300x300_ssd_iter_140000.caffemodel',  # face detector
	'embedding_model': 'openface_nn4.small2.v1.t7',  # path to OpenCV's deep learning face embedding model
	'recog_pickle': 'models/trained/recognizer.pickle',  # load the face recognition model
	'le_pickle': 'models/trained/le.pickle'  # load label encoder
}

# deploy.prototxt file defines the model architecture, and the res10_300x300_ssd_iter_140000_fp16.caffemodel
# contains the weights for the actual layers. In order to perform a forward pass for the whole network ...
protoPath = os.path.sep.join([args['detector'], args['detector_proto']])  # face detector based on a res net
modelPath = os.path.sep.join([args['detector'], args['detector_model']])  # face detector
detector = cv2.dnn.readNetFromCaffe(protoPath, modelPath)  # load serialized face detector from disk
print("[INFO] loading face detector:", args['detector_model'])
face_rec = os.path.sep.join([args["detector"], args["embedding_model"]])
embedder = cv2.dnn.readNetFromTorch(face_rec)  # load serialized face embedding model from disk
recognizer = pickle.loads(open(args['recog_pickle'], "rb").read())  # load the face recognition model
le = pickle.loads(open(args['le_pickle'], "rb").read())  # load label encoder
print("[INFO] loading face recognizer:", args["embedding_model"])

print("[INFO] starting video stream ...")
if args['IPcam'] is True:  # -- stream from the phone
	vs = cv2.VideoCapture(args['IP'])  # initialize the video stream
	time.sleep(1.0)  # camera sensor to warm up
else:  # -- Computer Camera
	vs = VideoStream(src=0).start()
	time.sleep(1.0)

fps = FPS().start()  # start the FPS throughput estimator
color = [random.randint(0, 255) for _ in range(3)]
while True:  # loop over frames from the video file stream
	if args['IPcam'] is True:  # -- IP cam from the phone
		ret, frame = vs.read()  # tuple: ret, frame = () -- instead of only frame =
	else:  # -- Computer Camera
		frame = vs.read()  # grab the frame from the threaded video stream

	frame = imutils.resize(frame, width=600)  # resize to 600px keeping aspect ratio then grab the image dimensions
	frame = cv2.normalize(frame, None, 10, 230, cv2.NORM_MINMAX)  # normalize to get a better contrast
	(h, w) = frame.shape[:2]

	# construct a blob from the image
	imageBlob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0, (300, 300), (104.0, 187.0, 123.), swapRB=False, crop=False)
	detector.setInput(imageBlob)  # OpenCV's deep learning-based face detector to localize faces in the input image
	detections = detector.forward()

	for i in range(0, detections.shape[2]):  # loop over the detections
		confidence = detections[0, 0, i, 2]  # extract the confidence associated with the prediction

		if confidence > args['confidence']:  # filter out weak detections
			box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])  # x, y coordinates of the bounding box for the face
			(startX, startY, endX, endY) = box.astype("int")

			face = frame[startY:endY, startX:endX]  # extract the face ROI
			(fH, fW) = face.shape[:2]

			if fW < 20 or fH < 20:  # ensure the face width and height are sufficiently large
				continue

			# construct blob for the face ROI, then
			# pass the blob through face embedding model to obtain the 128-d quantification of the face
			faceBlob = cv2.dnn.blobFromImage(face, 1.0 / 255, (96, 96), (0, 0, 0), swapRB=True, crop=False)
			embedder.setInput(faceBlob)
			vec = embedder.forward()

			preds = recognizer.predict_proba(vec)[0]  # perform classification to recognize the face with sklearn
			j = np.argmax(preds)
			proba = preds[j]
			name = le.classes_[j]

			text = "{}: {:.2f}%".format(name, proba * 100)  # draw the face's bounding box along with the probability
			y = startY - 10 if startY - 10 > 10 else startY + 10
			cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)
			cv2.putText(frame, text, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)

	fps.update()  # update the FPS counter

	cv2.imshow("rbf_1.1_1.11", frame)  # show the output frame
	key = cv2.waitKey(1) & 0xFF

	if key == ord("q"):  # if the `q` key was pressed, break from the loop
		break

fps.stop()  # stop the timer and display FPS information
print("[INFO] elasped time: {:.2f} in approx. FPS: {:.2f}".format(fps.elapsed(), fps.fps()))

cv2.destroyAllWindows()
vs.stop()