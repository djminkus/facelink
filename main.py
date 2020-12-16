import cv2
import numpy as np
import os
import glob
from PIL import Image

# Creating the dataset and labeled the Training and Test data
# Defining the directory for the Training and Test datasets
TRAININGSET_DIRECTORY = 'TrainingDataset'
TESTSET_DIRECTORY = 'TestDataset'


assert (os.path.exists(TRAININGSET_DIRECTORY))
Training_Set_Names = glob.glob(os.path.join(TRAININGSET_DIRECTORY, "*.jpg"))
assert (len(Training_Set_Names) > 0)
assert (os.path.exists(TESTSET_DIRECTORY))
Test_Set_Names = glob.glob(os.path.join(TESTSET_DIRECTORY, "*.jpg"))
assert (len(Test_Set_Names) > 0)

Training_Set_Names.sort()
Test_Set_Names.sort()
font = cv2.FONT_HERSHEY_SIMPLEX


def getFacesAndLabels(paths, detector):
    faceSamples = []
    ids = []
    for imagePath in paths:
        PIL_img = Image.open(imagePath).convert('L') # convert it to grayscale
        img_numpy = np.array(PIL_img,'uint8')
        id = int(os.path.split(imagePath)[-1].split(".")[1])

        faces = detector.detectMultiScale(img_numpy)
        for (x, y, w, h) in faces:
            faceSamples.append(img_numpy[y:y+h,x:x+w])
            ids.append(id)
    return faceSamples, ids


def main():

    detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    # recognizer.read('trainer/trainer.yml')

    print("\n Training faces ...")
    faces, ids = getFacesAndLabels(Training_Set_Names, detector)
    recognizer.train(faces, np.array(ids))
    # Save the model into trainer/Trained Model.yml
    recognizer.write('trainer/Trained_Model.yml')
    # Print the number of faces trained and end program
    print("\n Faces trained.")


if __name__ == '__main__':
    main()
