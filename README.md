# Extracting and clustering dlib face embeddings

**dlib models:** https://github.com/davisking/dlib-models
- dlib_face_recognition_resnet_model_v1.dat
- shape_predictor_68_face_landmarks.dat

**caffemodel** https://github.com/gopinath-balu/computer_vision/tree/master/CAFFE_DNN
- res10_300x300_ssd_iter_140000.caffemodel
- deploy.prototxt.txt

**openface model** https://cmusatyalab.github.io/openface/models-and-accuracies/
- openface_nn4.small2.v1.t7

**git**
```
$ git clone https://github.com/chacoff/FaceDLIBOpenFAce
```

**dataset**

Choose the correct address to your dataset with faces. it is expected that the dataset folder contains folders each with the name of the person you want to extract the landmarks and is also expected that every photo has only 1 face. The code will take care of creating a dataframe with the landmarks and the respective face's name. Please see the example in the repository.

```
dataset = os.path.join('c:/', 'Coding', '5_FaceSVM', 'dataset')
```

**results**

<image src='https://github.com/chacoff/FaceDLIBOpenFAce/blob/main/models/dataset_git.png' width='420'>
