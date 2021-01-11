import numpy as np
import cv2 as cv
import dlib
import cmake
import face_recognition
import os


path = "Face Recog pics"
images = []
classNames = []
myList = os.listdir(path)
print(myList)

for clas in myList:
    curImg = cv.imread(f'{path}/{clas}')
    images.append(curImg)
    classNames.append(os.path.splitext(clas)[0])
print(classNames)
print(len(images))
def encodeImages( images ):
    encodings = []
    for img in images:
        img = cv.resize(img, (500, 500))
        img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        faceEncoded = face_recognition.face_encodings(img)[0]
        encodings.append(faceEncoded)
        # print(face_recognition.face_encodings(img))
    return encodings

encodeExisting = encodeImages( images )
print(len(encodeExisting))
print("Completed encoding pictures")


camera = cv.VideoCapture(0)
while True:
    success, img = camera.read()
    img = cv.flip(img, 1)
    imgResized = cv.resize(img, (0,0), None, 0.25, 0.25)
    imgResized = cv.cvtColor(imgResized, cv.COLOR_BGR2RGB)
    
    curr_faceLocation = face_recognition.face_locations(imgResized)
    curr_facesEncoded = face_recognition.face_encodings(imgResized, curr_faceLocation)

    for encodeFace,faceLocation in zip(curr_facesEncoded, curr_faceLocation):
        matches = face_recognition.compare_faces( encodeExisting, encodeFace)
        faceDist = face_recognition.face_distance(  encodeExisting, encodeFace )
        print(faceDist)
        matchIndex = np.argmin( faceDist )

        if matches[matchIndex]:
            name = classNames[matchIndex].upper()
            print(name)
            cv.rectangle(img, (faceLocation[3], faceLocation[0]),
                         (faceLocation[1], faceLocation[2]), (255, 255, 0))
            cv.putText(img, f'{name} ', (50,50), cv.FONT_HERSHEY_TRIPLEX, 1, (0, 255, 255), 2)

    cv.imshow("Frame", img)
    if cv.waitKey(1) & 0xFF == ord('q'):
        break



# ElonImg = cv.imread("Face Recog pics/Elon.jpg")
# ElonImg = cv.resize( ElonImg, (500, 500))
# ElonImg = cv.cvtColor(ElonImg, cv.COLOR_BGR2RGB)
#
# faceLocation = face_recognition.face_locations(ElonImg)[0]
# faceEncoded = face_recognition.face_encodings(ElonImg)[0]
# cv.rectangle(ElonImg, (faceLocation[3], faceLocation[0]), (faceLocation[1], faceLocation[2]),  (255, 255, 0))
#
#
#
# ElonTest = cv.imread("Face Recog pics/elon test.jpg")
# ElonTest = cv.resize(ElonTest, (500, 500))
# ElonTest= cv.cvtColor(ElonTest, cv.COLOR_BGR2RGB)
#
# test_faceLocation = face_recognition.face_locations(ElonTest)[0]
# test_faceEncoded = face_recognition.face_encodings(ElonTest)[0]
# cv.rectangle(ElonTest, (test_faceLocation[3], test_faceLocation[0]), (test_faceLocation[1], test_faceLocation[2]), (255, 255, 0) )
#
#
#
# results = face_recognition.compare_faces([faceEncoded], test_faceEncoded)
# faceDistance = face_recognition.face_distance([faceEncoded],test_faceEncoded)
# print(results, faceDistance)
# cv.putText(ElonTest, f'{results} {round(faceDistance[0],2)}', (50,50), cv.FONT_HERSHEY_TRIPLEX, 1, (0, 255, 255), 2)
# cv.imshow("Elon", ElonImg)
# cv.imshow("Elontest", ElonTest)


