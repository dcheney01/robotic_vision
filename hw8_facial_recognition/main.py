import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QLabel, QPushButton, QVBoxLayout, QWidget, QMessageBox, QTextEdit
from PyQt5.QtCore import QTimer
import cv2 as cv
import numpy as np
from PyQt5.QtGui import QImage, QPixmap
import os
from deepface import DeepFace
import time

class FacialRecognitionApp(QMainWindow):
    def __init__(self):

        super().__init__()

        self.gui_setup()

        self.database_path = "hw8_facial_recognition/database"
        self.camera = cv.VideoCapture(0)
        self.num_users = len(os.listdir(self.database_path))
        self.curr_frame = None

    def gui_setup(self):
        self.setWindowTitle("Facial Recognition App")
        self.setGeometry(100, 100, 660, 500)

        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)

        self.layout = QVBoxLayout()
        self.central_widget.setLayout(self.layout)

        self.camera_label = QLabel()
        self.layout.addWidget(self.camera_label)

        self.authenticate_button = QPushButton("Authenticate")
        self.enroll_button = QPushButton("Enroll")
        self.quit_button = QPushButton("Quit")

        self.authenticate_button.clicked.connect(self.authenticate)
        self.enroll_button.clicked.connect(self.enroll)
        self.quit_button.clicked.connect(self.quit)

        self.layout.addWidget(self.authenticate_button)
        self.layout.addWidget(self.enroll_button)
        self.layout.addWidget(self.quit_button)

        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(10)

        self.user_info = QTextEdit()
        self.layout.addWidget(self.user_info)

    def update_frame(self):
        ret, frame = self.camera.read()
        if ret:
            frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
            height, width, channel = frame.shape
            bytes_per_line = 3 * width
            q_image = QImage(frame.data, width, height, bytes_per_line, QImage.Format_RGB888)
            self.camera_label.setPixmap(QPixmap.fromImage(q_image))
            self.curr_frame = frame

    def authenticate(self):
        print("Authenticating")
        if self.num_users == 0:
            self.user_info.setText("No users in the database. Enroll first.")
            return
        
        self.user_info.setText("Authenticating")
        dfs = DeepFace.find(self.curr_frame, self.database_path, silent=True)[0]
        if not dfs.empty:
            user_id = None    
            for i, instance in dfs.iterrows():
                if user_id is None:
                    user_id = instance["identity"].split("/")[-2]
                elif user_id != instance["identity"].split("/")[-2]:
                    self.user_info.setText("Incorrect Detection")
                    return

            self.user_info.setText(f"Welcome user {user_id}")
        else:
            self.user_info.setText("User not found!")

    def enroll(self):
        print("Enrolling a new user")
        user_id = self.num_users - 1
        user_path = os.path.join(self.database_path, f"{user_id}")
        os.makedirs(user_path, exist_ok=True)

        imgs_captured = 0
        while imgs_captured < 10:
            ret, frame = self.camera.read()
            if ret:
                img_path = os.path.join(user_path, f"{imgs_captured}.jpg")
                cv.imwrite(img_path, frame)
                imgs_captured += 1
                time.sleep(0.2)

        self.user_info.setText("User enrolled")

        self.num_users += 1

            
    def quit(self):
        print("Quitting")
        self.user_info.setText("Quitting")
        self.close()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    camera_app = FacialRecognitionApp()
    camera_app.show()
    sys.exit(app.exec_())