import cv2
import numpy as np
import sys
import os
import glob
from PIL import Image

# Name of folder holding test images
TESTSET_DIRECTORY = 'TestDataset'

assert (os.path.exists(TESTSET_DIRECTORY))
Test_Set_Names = glob.glob(os.path.join(TESTSET_DIRECTORY, "*.jpg"))
Test_Set_Names.extend(glob.glob(os.path.join(TESTSET_DIRECTORY, '*.JPG')))
assert (len(Test_Set_Names) > 0)
Test_Set_Names.sort()

font = cv2.FONT_HERSHEY_SIMPLEX
names = ['David', 'Hakan', 'Unknown']
bios = [
    '''seeking Master's in Electrical Engineering at Colorado School of Mines''',
    'seeking a Ph.D. in __ at Colorado School of Mines'
]
unknown_bio = 'this person is not a FaceLink user.'


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
    # face_detector_side = cv2.CascadeClassifier('')
    face_recognizer = cv2.face.LBPHFaceRecognizer_create()
    face_recognizer.read('trainer/Trained_Model_w_DNN_hakan.yml')
    # face_recognizer.read('trainer/trainer.yml')

    imgs, grays = getImages(Test_Set_Names)
    counter = 0

    # Creating a directory for the results
    directory = "FACELINK RESULTS"

    # TODO PLEASE ENTER YOUR DIRECTORY!!
    #parent_dir = "/Users/dg/coding/PycharmProjects/facelink/bioPanes"
    parent_dir = os.path.dirname(__file__)

    path = os.path.join(parent_dir, directory)
    if not os.path.isdir(path):
        os.mkdir(path)

    for img, gray in zip(imgs, grays):

        # Scale Factor is 1.2 which is determined by trial and error
        # Min Neighbours is chosen 5
        faces = face_detector.detectMultiScale(gray, 1.2, 5)

        # TODO: Modularize this section
        for (x, y, w, h) in faces:
            # Get predicted label and "inverse similarity"
            id_number, inv_conf = face_recognizer.predict(gray[y : y + h, x : x + w])
            if inv_conf < 100:  # FaceLink user found; display name and bio
                user_name = names[id_number-1]
                bio = bios[id_number-1]
            else:  # No facelink user found
                user_name = "unknown"
                bio = unknown_bio
                
            confidence = "  {0}%".format(round(100 - inv_conf))

            # Color-code the face box based on confidence/similarity level
            if inv_conf < 20:          # HIGH confidence
                color = (0, 255, 0)    # green box
            elif inv_conf < 40:        # MEDIUM conf
                color = (0, 255, 255)  # yellow box
            else:                      # LOW conf
                color = (0, 0, 255)    # red box

            # Draw box around face:
            cv2.rectangle(img, (x, y), (x + w, y + h), color, 3)
            # Label box with username:
            cv2.putText(img, str(user_name), (x + 5, y - 5), font, 1, color, 2)
            # Include confidence value:
            cv2.putText(img, str(confidence), (x + 5, y + h - 5), font, 1, (255, 255, 0), 1)
            # Include bio:
            cv2.putText(img, bio, (x + 5, y + h + 20), font, .75, (255, 255, 0), 1)
            #                                                ^ fontScale
            # End faces loop
        cv2.imwrite(f'FACELINK RESULTS/{user_name}_{counter}.jpg', img)
        counter += 1
        # End images loop
    print('Test Dataset results are READY!!')


if __name__ == '__main__':
    main()
