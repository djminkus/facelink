import cv2
import numpy as np
import sys
import os
import glob
from PIL import Image

# Creating the dataset and labeled the training and test data
TESTSET_DIRECTORY = 'TestDataset'

assert (os.path.exists(TESTSET_DIRECTORY))
Test_Set_Names = glob.glob(os.path.join(TESTSET_DIRECTORY, "*.jpg"))
assert (len(Test_Set_Names) > 0)

Test_Set_Names.sort()
font = cv2.FONT_HERSHEY_SIMPLEX
names = ['David', 'Hakan', 'Unknown']


def getImages(paths):
    # imagePaths = [os.path.join(path, f) for f in os.listdir(path)]

    imgs = []
    grays = []
    for imagePath in paths:
        PIL_img = Image.open(imagePath)
        PIL_img_gray = PIL_img.convert('L')
        img_numpy = np.array(cv2.imread(imagePath))
        img_gray_numpy = np.array(PIL_img_gray, 'uint8')
        imgs.append(img_numpy)
        grays.append(img_gray_numpy)
    return imgs, grays


def main():

    face_detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    face_recognizer = cv2.face.LBPHFaceRecognizer_create()
    face_recognizer.read('trainer/Trained_Model.yml')

    imgs, grays = getImages(Test_Set_Names)
    counter = 0

    # Creating a directory for the results
    directory = "FACELINK RESULTS"

    # TODO PLEASE ENTER YOUR DIRECTORY!!
    parent_dir = "C:/Users/ayaz_a/Desktop/Computer Vision/Face_Recognition_Project"
    path = os.path.join(parent_dir, directory)
    os.mkdir(path)

    for img, gray in zip(imgs, grays):

        # Scale Factor is 1.2 which is determined by trial and error
        # Min Neighbours is chosen 5
        faces = face_detector.detectMultiScale(gray, 1.2, 5)

        for (x, y, w, h) in faces:

            id, similarity = face_recognizer.predict(gray[y:y + h, x:x + w])
            if similarity < 100:
                id = names[id-1]
                confidence = "  {0}%".format(round(100 - similarity))
            else:
                id = "unknown"
                confidence = "  {0}%".format(round(100 - similarity))

            # According the similarity results face box will be change
            if similarity < 20:
                color = (0, 255, 0)    # green box
            elif similarity < 40:
                color = (0, 255, 255)  # yellow box
            else:
                color = (0, 0, 255)    # red box

            cv2.rectangle(img, (x, y), (x + w, y + h), color , 3)
            cv2.putText(img, str(id), (x + 5, y - 5), font, 1, color, 2)
            cv2.putText(img, str(confidence), (x + 5, y + h - 5), font, 1, (255, 255, 0), 1)

        cv2.imwrite(f'FACELINK RESULTS/{id}_{counter}.jpg', img)
        counter += 1
    print('Test Dataset results are READY!!')


if __name__ == '__main__':
    main()
