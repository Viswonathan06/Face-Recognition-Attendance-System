# Face-Recognition-Attendance-System
  ---
## Incentive
With rising concerns about personal health and safety with the onset of COVID-19, a lot of changes have occured in our daily life, including a shift towards contactless technology. This application is a python programmed OpenCV-based facial recognition app made to be used in schools and universities, and with minor tweeks, can be used online as well.

## How it works?
As computer vision problems take days to compute on huge datasets, a reliable facial recognition model would have taken a large time as well with the current bottlenecked system in possession. Hence, a pre-trained model by Adam Geitgey using the method of Histogram of Oriented Gradients — or just HOG for short. He uses the face landmark estimation method and a pretrained model to get the encodings, which is later classified using a SVM classifier. 

To learn more about the method, go to:

https://medium.com/@ageitgey/machine-learning-is-fun-part-4-modern-face-recognition-with-deep-learning-c3cffc121d78

## Quirks of the application
The application currently detects the faces that are saved in the directory provided, and marks their attendance in a csv file with time and date of detection. 

##Future expansions

* A GUI can be made to make the process of adding individuals to the dataset
* A face mask detection method can also be used to detect if the individuals are following protocol. 

---

### Libraries used
numpy as np
cv2 as cv
dlib
cmake
face_recognition
os
datetime 
