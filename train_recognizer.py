import cv2
import numpy as np
import os
import glob
from PIL import Image


# Define paths for DNN
base_dir = os.path.dirname(__file__)
prototxt_path = os.path.join(base_dir + '/model_data/deploy.prototxt')
caffemodel_path = os.path.join(base_dir + '/model_data/weights.caffemodel')
# Read the model with DNN algorithm
model = cv2.dnn.readNetFromCaffe(prototxt_path, caffemodel_path)

# Creating the dataset and labeled the Training and Test data
# Defining the directory for the Training and Test datasets
TRAININGSET_DIRECTORY = 'TrainingDataset'


assert (os.path.exists(TRAININGSET_DIRECTORY))

Training_Set_Names = glob.glob(os.path.join(TRAININGSET_DIRECTORY, "*.jpg"))
Training_Set_Names.extend(glob.glob(os.path.join(TRAININGSET_DIRECTORY, '*.JPG')))
print("files found: " + str(len(Training_Set_Names)))
#Training_Set_Names = os.listdir(base_dir+'/TrainingDataset/')
assert (len(Training_Set_Names) > 0)

Training_Set_Names.sort()
font = cv2.FONT_HERSHEY_SIMPLEX


def getFacesAndLabels(paths):
    faceSamples = []
    ids = []
    count = 0
    for imagePath in paths:
        # PIL_img = Image.open(imagePath).convert('L') # convert it to grayscale
        # img_numpy = np.array(PIL_img,'uint8')

        img_numpy = cv2.imread(imagePath)
        gray = cv2.cvtColor(img_numpy, cv2.COLOR_BGR2GRAY)
        id = int(os.path.split(imagePath)[-1].split(".")[1])

        # DNN
        (h, w) = gray.shape[:2]
        blob = cv2.dnn.blobFromImage(cv2.resize(img_numpy, (300, 300)), 1.0,  # ***
                                     (300, 300), (104.0, 177.0, 123.0))
        model.setInput(blob)
        detections = model.forward()

        # faces = detector.detectMultiScale(img_numpy)
        # for (x, y, w, h) in faces:
        #     faceSamples.append(img_numpy[y:y+h,x:x+w])
        #     ids.append(id)

        # Identify each face
        for i in range(0, detections.shape[2]):
            detections[0, 0, i, 3:7]
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")

            confidence = detections[0, 0, i, 2]

            # haar filtering? Check for face
            # Check for face in 0-index
            #   if not found, assume face in 1-index

            # If confidence > 0.5, append it to the face samples
            if confidence > 0.5 and i < 1:
                count += 1
                gray_resized = cv2.resize(gray, (300, 300))
                face = gray_resized[startY:endY, startX:endX]  # ***
                faceSamples.append(face)
                ids.append(id)
                cv2.imwrite(base_dir + '/dnn_extracted_faces/' + str(i) + '_' + imagePath, face)

    print("faces found: " + str(count))
    return faceSamples, ids


def main():
    # detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    # recognizer.read('trainer/trainer.yml')

    print('\n detecting faces ...')
    faces, ids = getFacesAndLabels(Training_Set_Names)
    print("\n Training faces ...")
    recognizer.train(faces, np.array(ids))
    # Save the model into trainer/Trained Model.yml
    recognizer.write('trainer/Trained_Model_w_DNN.yml')
    # Print the number of faces trained and end program
    print("\n Faces trained.")


if __name__ == '__main__':
    main()
