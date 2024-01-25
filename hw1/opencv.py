import threading
import numpy as np
import cv2 as cv
from PIL import Image, ImageTk
from tkinter import Tk, Frame, Button, BOTH, Label, Scale, Radiobutton  # Graphical User Inetrface Stuff
from tkinter import font as tkFont
import tkinter as tk

camera = cv.VideoCapture(4)
width = int(camera.get(cv.CAP_PROP_FRAME_WIDTH))
height = int(camera.get(cv.CAP_PROP_FRAME_HEIGHT))
videoout = cv.VideoWriter('hw1/test.avi', 0, 10, (width, height))  # Video format

# Button Definitions
ORIGINAL = 0
BINARY = 1
EDGE = 2
LINE = 3
ABSDIFF = 4
RGB = 5
HSV = 6
CORNERS = 7
CONTOURS = 8


def cvMat2tkImg(arr):  # Convert OpenCV image Mat to image for display
    rgb = cv.cvtColor(arr, cv.COLOR_BGR2RGB)
    img = Image.fromarray(rgb)
    return ImageTk.PhotoImage(img)


class App(Frame):
    def __init__(self, winname='OpenCV'):  # GUI Design

        self.root = Tk()
        self.stopflag = True
        self.buffer = np.zeros((height, width, 3), dtype=np.uint8)

        global helv18
        helv18 = tkFont.Font(family='Helvetica', size=18, weight='bold')
        # print("Width",windowWidth,"Height",windowHeight)
        self.root.wm_title(winname)
        positionRight = int(self.root.winfo_screenwidth() / 2 - width / 2)
        positionDown = int(self.root.winfo_screenheight() / 2 - height / 2)
        # Positions the window in the center of the page.
        self.root.geometry("+{}+{}".format(positionRight, positionDown))
        self.root.wm_protocol("WM_DELETE_WINDOW", self.exitApp)
        Frame.__init__(self, self.root)
        self.pack(fill=BOTH, expand=1)
        # capture and display the first frame
        ret0, frame = camera.read()
        image = cvMat2tkImg(frame)
        self.panel = Label(image=image)
        self.panel.image = image
        self.panel.pack(side="top")
        # buttons
        global btnQuit
        btnQuit = Button(text="Quit", command=self.quit)
        btnQuit['font'] = helv18
        btnQuit.pack(side='right', pady=2)
        global btnStart
        btnStart = Button(text="Start", command=self.startstop)
        btnStart['font'] = helv18
        btnStart.pack(side='right', pady=2)
        # sliders
        global Slider1, Slider2
        Slider2 = Scale(self.root, from_=0, to=255, length=255, orient='horizontal')
        Slider2.pack(side='right')
        Slider2.set(192)
        Slider1 = Scale(self.root, from_=0, to=255, length=255, orient='horizontal')
        Slider1.pack(side='right')
        Slider1.set(64)
        # radio buttons
        global mode
        mode = tk.IntVar()
        mode.set(ORIGINAL)
        Radiobutton(self.root, text="Original", variable=mode, value=ORIGINAL).pack(side='left', pady=4)
        Radiobutton(self.root, text="Binary", variable=mode, value=BINARY).pack(side='left', pady=4)
        Radiobutton(self.root, text="Edge", variable=mode, value=EDGE).pack(side='left', pady=4)
        Radiobutton(self.root, text="Line", variable=mode, value=LINE).pack(side='left', pady=4)
        Radiobutton(self.root, text="Corners", variable=mode, value=CORNERS).pack(side='left', pady=4)
        Radiobutton(self.root, text="Abs Diff", variable=mode, value=ABSDIFF).pack(side='left', pady=4)
        Radiobutton(self.root, text="Contours", variable=mode, value=CONTOURS).pack(side='left', pady=4)
        Radiobutton(self.root, text="RGB", variable=mode, value=RGB).pack(side='left', pady=4)
        Radiobutton(self.root, text="HSV", variable=mode, value=HSV).pack(side='left', pady=4)
        # threading
        self.stopevent = threading.Event()
        self.thread = threading.Thread(target=self.capture, args=())
        self.thread.start()

    def capture(self):
        while not self.stopevent.is_set():
            if not self.stopflag:
                ret0, frame = camera.read()
                if mode.get() == BINARY:
                    lThreshold = Slider1.get()
                    hThreshold = Slider2.get()
                    frame_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
                    ret, lower = cv.threshold(frame_gray, lThreshold, 255, cv.THRESH_BINARY)
                    ret, upper = cv.threshold(frame_gray, hThreshold, 255, cv.THRESH_BINARY_INV)
                    frame = cv.bitwise_and(lower, upper)
                elif mode.get() == EDGE:
                    lThreshold = Slider1.get()
                    hThreshold = Slider2.get()
                    frame = cv.Canny(frame, lThreshold*10, hThreshold*10)
                elif mode.get() == CORNERS:
                    lThreshold = Slider1.get()
                    hThreshold = Slider2.get()
                    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
                    init_corners = cv.goodFeaturesToTrack(gray, 100, 0.01, hThreshold)
                    corners = cv.cornerSubPix(gray, init_corners, (lThreshold+1, lThreshold+1), (-1, -1), (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001))
                    for i in range(corners.shape[0]):
                        cv.circle(frame, (int(corners[i,0,0]), int(corners[i,0,1])), 3, (0, 0, 255), cv.FILLED)
                elif mode.get() == LINE:
                    lThreshold = Slider1.get()
                    hThreshold = Slider2.get()
                    dst = cv.Canny(frame, 35*10, 70*10)
                    lines = cv.HoughLines(dst, 1, np.pi / 180, hThreshold)
                    if lines is not None:
                        for i in range(0, len(lines)):
                            rho = lines[i][0][0]
                            theta = lines[i][0][1]
                            a = np.cos(theta)
                            b = np.sin(theta)
                            x0 = a * rho
                            y0 = b * rho
                            pt1 = (int(x0 + 1000 * (-b)), int(y0 + 1000 * (a)))
                            pt2 = (int(x0 - 1000 * (-b)), int(y0 - 1000 * (a)))
                            cv.line(frame, pt1, pt2, (0, 0, 255), 3, cv.LINE_AA)
                elif mode.get() == ABSDIFF:
                    lThreshold = Slider1.get()
                    hThreshold = Slider2.get()

                    temp = frame.copy()
                    frame = cv.absdiff(frame, self.buffer)
                    self.buffer = temp
                elif mode.get() == CONTOURS:
                    lThreshold = Slider1.get()
                    hThreshold = Slider2.get()

                    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
                    edges = cv.Canny(gray, 27*10, 21*10) #150 , 171
                    # kernel = np.ones((5, 5), np.uint8)
                    # dilated = cv.dilate(edges, kernel, iterations=1)
                    contours, hierarchy = cv.findContours(edges, cv.RETR_CCOMP, cv.CHAIN_APPROX_TC89_KCOS) # CHAIN_APPROX_TC89_KCOS
                    # cv.drawContours(frame, contours, -1, (0, 255, 0), 3)
                    for i, cnt in enumerate(contours):
                        if cv.contourArea(cnt) > 1000:
                            x, y, w, h = cv.boundingRect(cnt)
                            cv.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                elif mode.get() == RGB:
                    lThreshold = Slider1.get()
                    hThreshold = Slider2.get()
                elif mode.get() == HSV:
                    lThreshold = Slider1.get()
                    hThreshold = Slider2.get()
                    # Add your code here

                image = cvMat2tkImg(frame)
                self.panel.configure(image=image)
                self.panel.image = image
                videoout.write(frame)

    def startstop(self):  # toggle flag to start and stop
        if btnStart.config('text')[-1] == 'Start':
            btnStart.config(text='Stop')
        else:
            btnStart.config(text='Start')
        self.stopflag = not self.stopflag

    def run(self):  # run main loop
        self.root.mainloop()

    def quit(self):  # exit loop
        self.stopflag = True
        t = threading.Timer(1.0, self.stop)  # start a timer (non-blocking) to give main thread time to stop
        t.start()

    def exitApp(self):  # exit loop
        self.stopflag = True
        t = threading.Timer(1.0, self.stop)  # start a timer (non-blocking) to give main thread time to stop
        t.start()

    def stop(self):
        self.stopevent.set()
        self.root.quit()


app = App()
app.run()
# release the camera
camera.release()