# coding:utf-8
from kivy.app import App
from kivy.uix.image import Image
from kivy.clock import Clock
from kivy.graphics.texture import Texture
import cv2

font = cv2.FONT_HERSHEY_SIMPLEX
names = ['David', 'Hakan', 'Unknown']
bios = [
    '''seeking Master's in Electrical Engineering at Colorado School of Mines''',
    'seeking a Ph.D. in __ at Colorado School of Mines',
    'this person is not a FaceLink user.'
]

# ---- everything before cam loop:
counter = 0

# Initialize and start realtime video capture
cam = cv2.VideoCapture(0)
cam.set(3, 640)  # set video width
cam.set(4, 480)  # set video height

# Define min window size to be recognized as a face
minW = 0.1 * cam.get(3)
minH = 0.1 * cam.get(4)

# face_detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
# face_recognizer = cv2.face.LBPHFaceRecognizer_create()
# face_recognizer.read('trainer/trained_model.yml')
#

class KivyCamera(Image):
    def __init__(self, capture, fps, **kwargs):
        super(KivyCamera, self).__init__(**kwargs)
        self.capture = capture  #deined as cv2.VideoCapture(0) in CamApp build func
        Clock.schedule_interval(self.update, 1.0 / fps)

        self.face_detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

        self.face_recognizer = cv2.face.LBPHFaceRecognizer_create()
        self.face_recognizer.read('trainer/trained_model.yml')

    def update(self, dt):
        _, frame = self.capture.read()  # Frame is image
        if _:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # print(gray)

            faces = self.face_detector.detectMultiScale(
                gray,
                scaleFactor=1.2,
                minNeighbors=5,
                minSize=(int(minW), int(minH)),
            )

            # print(faces)

            for (x, y, w, h) in faces:

                id, inv_conf = self.face_recognizer.predict(gray[y:y + h, x:x + w])
                if inv_conf < 100:
                    id = names[id - 1]
                    confidence = "  {0}%".format(round(100 - inv_conf))
                else:
                    id = "unknown"
                    confidence = "  {0}%".format(round(100 - inv_conf))

                # According the similarity results face box will be change
                if inv_conf < 20:
                    color = (0, 255, 0)  # green box
                elif inv_conf < 40:
                    color = (0, 255, 255)  # yellow box
                else:
                    color = (0, 0, 255)  # red box

                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 3)
                cv2.putText(frame, str(id), (x + 5, y - 5), font, 1, color, 2)
                cv2.putText(frame, str(confidence), (x + 5, y + h - 5), font, 1, (255, 255, 0), 1)

            # convert it to texture for display in app:
            buf1 = cv2.flip(frame, 0)
            buf = buf1.tostring()
            image_texture = Texture.create(
                size=(frame.shape[1], frame.shape[0]), colorfmt='bgr')
            image_texture.blit_buffer(buf, colorfmt='bgr', bufferfmt='ubyte')
            # display image from the texture
            self.texture = image_texture


class CamApp(App):
    def build(self):
        self.capture = cv2.VideoCapture(0)
        self.my_camera = KivyCamera(capture=self.capture, fps=30)
        return self.my_camera

    def on_stop(self):
        # Without this, app will not exit even if the window is closed
        self.capture.release()


if __name__ == '__main__':
    CamApp().run()
