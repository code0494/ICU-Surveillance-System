import cv2
import sys
import imutils
from face_det import Face 
from datetime import datetime
from PyQt5.QtCore import QTimer
from obj_det import obj_detection
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QPushButton, QLabel, QHBoxLayout

class ICUcamApp(QWidget):
    def __init__(self):
        super().__init__()
        
        self.init_ui()
        self.face = Face()
        self.counter = 29

    def init_ui(self):
        self.setWindowTitle('ICU Survillence')

        layout = QVBoxLayout()

        video_layout = QHBoxLayout()
        self.video_label1 = QLabel()
        video_layout.addWidget(self.video_label1)
        self.video_label2 = QLabel()
        video_layout.addWidget(self.video_label2)

        layout.addLayout(video_layout)
        
        label_layout = QHBoxLayout()
        self.face_label1 = QLabel()
        self.face_label2 = QLabel()
        label_layout.addWidget(self.face_label1)
        label_layout.addWidget(self.face_label2)
        
        layout.addLayout(label_layout)

        self.start_button = QPushButton('Start')
        self.start_button.clicked.connect(self.start_webcam)
        layout.addWidget(self.start_button)

        self.exit_button = QPushButton('Exit')
        self.exit_button.clicked.connect(self.exit_app)
        layout.addWidget(self.exit_button)

        self.setLayout(layout)

        self.webcam1 = cv2.VideoCapture(0)
        self.webcam2 = cv2.VideoCapture('icu1.mp4')
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        
        self.face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')


    def start_webcam(self):
        if not self.timer.isActive():
            self.timer.start(30)

    def update_frame(self):
        ret1, frame1 = self.webcam1.read()
        ret2, frame2 = self.webcam2.read()
        

        if ret1:
            rgb_frame1, faces = self.face.face_det(frame1)
            h1, w1, ch1 = rgb_frame1.shape
            bytes_per_line1 = ch1 * w1
            qimg1 = QImage(rgb_frame1.data, w1, h1, bytes_per_line1, QImage.Format_RGB888)
            pixmap1 = QPixmap.fromImage(qimg1)
            self.video_label1.setPixmap(pixmap1)
            name = ''
            if len(faces)>0:
                name =  self.face.rec_face(frame1)
                self.face_label1.setText('face detected ' + name)
                k = datetime.now()
                l = str(k.year) + str(k.month) + str(k.day) + str(k.hour) + str(k.minute) + str(k.second)
                #print(name, self.counter)
                if name != 'Unknown' and self.counter >= 30:
                    cv2.imwrite('log\\' + name + '_' +  l + '.jpg', frame1)
                    self.counter = 0
                else:
                    self.counter += 1
                
            else:
                self.face_label1.setText('no face detected ')

        if ret2:
            rgb_frame2, text = obj_detection(frame2)
            h2, w2, ch2 = rgb_frame2.shape
            bytes_per_line2 = ch2 * w2
            qimg2 = QImage(rgb_frame2.data, w2, h2, bytes_per_line2, QImage.Format_RGB888)
            pixmap2 = QPixmap.fromImage(qimg2)
            self.video_label2.setPixmap(pixmap2)
            
            if text.lower().find("1 person") == 0:
                self.face_label2.setStyleSheet('color: {}'.format('green'))
                self.face_label2.setText(text)
            else:
                self.face_label2.setStyleSheet('color: {}'.format('red'))
                self.face_label2.setText(text)
            
            
            
    def exit_app(self):
        self.timer.stop()
        self.webcam1.release()
        self.webcam2.release()
        QApplication.quit()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    webcam_app = ICUcamApp()
    webcam_app.show()
    sys.exit(app.exec_())