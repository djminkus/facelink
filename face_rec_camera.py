import cv2
import numpy as np
import sys
import os
import glob
from PIL import Image

font = cv2.FONT_HERSHEY_SIMPLEX
names = ['David', 'Hakan', 'Unknown']


def main():

    counter = 0

    # Initialize and start realtime video capture
    cam = cv2.VideoCapture(0)
    cam.set(3, 640)  # set video widht
    cam.set(4, 480)  # set video height

    # Define min window size to be recognized as a face
    minW = 0.1 * cam.get(3)
    minH = 0.1 * cam.get(4)

    face_detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    face_recognizer = cv2.face.LBPHFaceRecognizer_create()
    face_recognizer.read('trainer/Trained_Model.yml')

    while True:

        ret, img = cam.read()
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        faces = face_detector.detectMultiScale(
            gray,
            scaleFactor=1.2,
            minNeighbors=5,
            minSize=(int(minW), int(minH)),
        )

        for img, gray in faces:

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

            cv2.imshow('camera', img)
            k = cv2.waitKey(10) & 0xff  # Press 'ESC' for exiting video
            if k == 27:
                break

    # Do a bit of cleanup
    print("\n [INFO] Exiting Program and cleanup stuff")
    cam.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
