from imutils import paths
import numpy as np
import imutils
import pickle
import cv2
import os
import sklearn
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn import svm
from sklearn.svm import SVC
from tqdm import tqdm
from time import time
from utils.dlib_faceAlign import dlib_aligner, toshow
from imutils.face_utils import FaceAligner, rect_to_bb
import dlib

# Dictionary with parameters
args = {
	'detector': 'models',  # path to OpenCV's deep learning face detector
	'detector_proto': 'deploy.prototxt',  # face detector based on a resnet
	'detector_model': 'res10_300x300_ssd_iter_140000.caffemodel',  # face detector
	'embedding_model': 'openface_nn4.small2.v1.t7',  # path to OpenCV's deep learning face embedding model
	'dataset': 'dataset',  # path to input directory of faces + images
	'emb_names': 'models/trained/emb_names.pickle',
	'embeddings': 'models/trained/embeddings.pickle',  # path to output serialized db of facial embeddings
	'recognizer': 'models/trained/recognizer.pickle',  # path to output model trained to recognize faces
	'le': 'models/trained/le.pickle',  # path to output label encoder
	'confidence': 0.5,  # minimum probability to filter weak detections
	'shape_predictor': 'shape_predictor_68_face_landmarks.dat',  # facial landmark predictor
	'normalized_size': 300  # normalized size for resizing
}

protoPath = os.path.sep.join([args['detector'], args['detector_proto']])  # caffemodel face detector
modelPath = os.path.sep.join([args['detector'], args['detector_model']])  # face detector
detector = cv2.dnn.readNetFromCaffe(protoPath, modelPath)  # load serialized face detector from disk
print("[INFO] loading face detector:", args['detector_model'])
predictor_model = os.path.sep.join([args['detector'], args['shape_predictor']])
predictor = dlib.shape_predictor(predictor_model)  # to align the face
print("[INFO] loading shape predictor:", args['shape_predictor'])
face_rec = os.path.sep.join([args["detector"], args["embedding_model"]])
embedder = cv2.dnn.readNetFromTorch(face_rec)  # load serialized face embedding model from disk
print("[INFO] loading face recognizer:", args["embedding_model"])
cv2.waitKey(20)
imagePaths = list(paths.list_images(args["dataset"]))  # grab the paths to the input images in dataset
knownEmbeddings = []  # lists of extracted facial embeddings
knownNames = []  # list of corresponding people names
total = 0  # initialize the total number of faces to process

pbar = tqdm(total=len(imagePaths))
for (i, imagePath) in enumerate(imagePaths):  # loop over the image paths
	name = imagePath.split(os.path.sep)[-2]  # extract the person name from the image path
	# print("[INFO] processing image {}/{}".format(i + 1, len(imagePaths)))
	# print('[INFO] processing image:', imagePath.split('\\')[-1])  # name
	image = cv2.imread(imagePath)
	image = imutils.resize(image, width=800)  # resize to 600px with aspect ratio
	image = cv2.normalize(image, None, 20, 230, cv2.NORM_MINMAX)  # normalize to get a better contrast
	(h, w) = image.shape[:2]  # image dimensions

	# construct a blob from the image
	imageBlob = cv2.dnn.blobFromImage(
		cv2.resize(image, (args['normalized_size'], args['normalized_size'])), 1.0,
		(args['normalized_size'], args['normalized_size']), (104.0, 187.0, 123.0), swapRB=False, crop=False
	)

	# apply OpenCV's deep learning-based face detector to localize faces in the input image
	detector.setInput(imageBlob)
	detections = detector.forward()

	if len(detections) > 0:  # ensure at least one face was found
		# assumption that each image has only ONE face, so find the bounding box with the largest probability
		i = np.argmax(detections[0, 0, :, 2])
		confidence = detections[0, 0, i, 2]

		# ensure that the detection with the largest probability also
		# means our minimum probability test (thus helping filter out weak detections)
		if confidence > args["confidence"]:
			box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])  # (x, y)-coordinates of bounding box for the face
			(startX, startY, endX, endY) = box.astype("int")  # coordinates of the face detection <<bb>>

			face = image[startY:endY, startX:endX]  # extract the face ROI and grab the ROI dimensions
			(fH, fW) = face.shape[:2]

			face_align, face_draw, face_circle = dlib_aligner(image, predictor, box.astype('int'), args['normalized_size'])
			toshow(name, face_align, face_draw, face_circle, face, total, save_emb=True)

			if fW < 20 or fH < 20:  # ensure the face width and height are sufficiently large
				continue
			# <face_align> goes now to the faceblob line 88 instead of <face>
			# blob for the face ROI, then pass it through the face embedding model to obtain the 128-d quantifications
			faceBlob = cv2.dnn.blobFromImage(face_align, 1.0 / 255, (96, 96), (0, 0, 0), swapRB=True, crop=False)
			embedder.setInput(faceBlob)
			vec = embedder.forward()

			knownNames.append(name)  # add the name of the person + corresponding face
			knownEmbeddings.append(vec.flatten())  # embedding to their respective lists
			total += 1
	pbar.update(1)
pbar.close()

data = {  # all elements with same len() embedding[0] + names[0] = array with the 128-d + name
	"embeddings": knownEmbeddings,
	"names": knownNames
}

f = open(args["embeddings"], "wb")
f.write(pickle.dumps(data['embeddings']))  # write the actual face recognition model to disk
f.close()

f = open(args['emb_names'], 'wb')
f.write(pickle.dumps(data['names']))
f.close()

cv2.waitKey(20)
# train the model used to accept the 128-d embeddings of the face and then produce the actual face recognition
print("[INFO] serializing {} encodings".format(total))
total_classes = set(data['names'])  # to get only the unique classes
print("[INFO] loading face embeddings & encoding labels:", total_classes)
le = LabelEncoder()
labels = le.fit_transform(data["names"])  # encode the labels

t0 = time()
recognizer = SVC(C=1.1, kernel="rbf", gamma=1.11, probability=True)

recognizer.fit(data["embeddings"], labels)
print('[INFO] model trained in %0.4fs' % (time() - t0))

f = open(args["recognizer"], "wb")
f.write(pickle.dumps(recognizer))  # write the actual face recognition model to disk
f.close()

f = open(args["le"], "wb")
f.write(pickle.dumps(le))  # write the label encoder to disk
f.close()