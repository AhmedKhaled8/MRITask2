from PyQt5 import QtWidgets, QtGui, QtWidgets
import sys
import os
import numpy as np
import cv2 as cv
from GUI import Ui_MainWindow
from functions import fourier
import matplotlib.pyplot as plt



class ApplicationWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super(ApplicationWindow, self).__init__()
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        
    ##############  UI control lists  ##############
        self.imageViews = [self.ui.graphicsView , self.ui.graphicsView_2 ] 
        self.legend = [0, 0, 0] # guard for legend overaddition

        for image in self.imageViews : ## hide unwanted options
            image.ui.histogram.hide()
            image.ui.roiBtn.hide()
            image.ui.menuBtn.hide()
            image.ui.roiPlot.hide()

    ############# initialize fourier #############
        self.originalImage0 = cv.cvtColor(cv.imread('images/Brain2.png'),cv.COLOR_BGR2GRAY)
        #plot original
        self.imageViews[0].show()
        self.imageViews[0].setImage(self.originalImage0.T)

        #plot fourier
        self.imageViews[1].show()
        self.imageViews[1].setImage(fourier(self.originalImage0).T)
    ####################################
        #self.axes1 = self.ui.graphicsView1.figure.add_subplot(111, title="$M_x/M_o$ vs time")
        #self.axes2 = self.ui.graphicsView2.figure.add_subplot(111)
        #self.axes3 = self.ui.graphicsView3.figure.add_subplot(111)
        
    ##############  UI Signals  ##############
        self.ui.pushButton.clicked.connect(lambda: self.plotEffect())
        self.ui.pushButton_2.clicked.connect(lambda : self.plotFourier())
        
    ##########################################
    

    

    def plotFourier(self) :
        
        """[summary] : plots : original image, it's fourier transform
        """     
        #Read the image file 
        fname = QtGui.QFileDialog.getOpenFileName( self, 'choose the image', os.getenv('HOME') ,"Images (*.png *.xpm *.jpg)" )
        self.path = fname[0] 
        #Assertion for path errors
        if self.path =="" :
            return
        #convert image to grey Scale
        self.originalImage = cv.cvtColor(cv.imread(self.path),cv.COLOR_BGR2GRAY)
        #plot original
        self.imageViews[0].show()
        self.imageViews[0].setImage(self.originalImage.T)

        #plot fourier
        self.imageViews[1].show()
        self.imageViews[1].setImage(fourier(self.originalImage).T)
        return None
    
    


    def plotEffect(self):
        print("Drawing curves")
            
        B = 1.5
        BPositive = 2.5
        BNegative = 0.5
        gyroRatio = 42
        w = gyroRatio * B
        wPositive = gyroRatio * BPositive
        wNegative = gyroRatio * BNegative
        T1 = 490/1000
        T2 = 43/1000
        t = np.arange(start=0, stop=10, step=0.0001)

        omega = 2*np.pi*w*t
        omegaPositive = 2*np.pi*wPositive*t + np.pi/8
        omegaNegative = 2*np.pi*wNegative*t - np.pi/8


        Mx = np.exp(-1*t/T2)*np.sin(omega)
        MxPositive = np.exp(-1*t/T2)*np.sin(omegaPositive)
        MxNegative = np.exp(-1*t/T2)*np.sin(omegaNegative)


        My = np.exp(-1*t/T2)*np.cos(omega)
        MyPositive = np.exp(-1*t/T2)*np.cos(omegaPositive)
        MyNegative = np.exp(-1*t/T2)*np.cos(omegaNegative)


        Mxy = np.sqrt(Mx**2 + My**2)
        MxyPositive = np.sqrt(MxPositive**2 + MyPositive**2)
        MxyNegative = np.sqrt(MxNegative**2 + MyNegative**2)
        
        self.axes1 = self.ui.graphicsView1.figure.add_subplot(111, title="$M_x/M_o$ vs time",xlabel="time",ylabel="$M_x/M_o$")
        self.axes1.plot(t[:1000], Mx[:1000], 'r', label="No Noise")
        self.axes1.plot(t[:1000], MxPositive[:1000], 'b', label="Positive Noise")
        self.axes1.plot(t[:1000], MxNegative[:1000], 'y', label="Negative Noise")
        if self.legend[0] == 0:
            self.axes1.legend()
            self.legend[0] += 1


        self.axes2 = self.ui.graphicsView2.figure.add_subplot(111,title="$M_y/M_o$ vs time",xlabel="time",ylabel="$M_y/M_o$")
        self.axes2.plot(t[:1000], My[:1000], 'r', label="No Noise")
        self.axes2.plot(t[:1000], MyPositive[:1000], 'b', label="Positive Noise")
        self.axes2.plot(t[:1000], MyNegative[:1000], 'y', label="Negative Noise")
        if self.legend[1] == 0:
            self.axes2.legend()
            self.legend[1] = 1

        self.axes3 = self.ui.graphicsView3.figure.add_subplot(111,title="$M_{xy}$ in X-Y Plane",xlabel="$M_x/M_o$",ylabel="$M_y/M_o$")
        self.axes3.plot(Mx, My, 'r', label="No Noise")
        self.axes3.plot(MxPositive, MyPositive, 'b', label="Positive Noise")
        self.axes3.plot(MxNegative, MyNegative, 'y', label="Negative Noise")
        if self.legend[2] == 0:
            self.axes3.legend()
            self.legend[2] = 1

        self.axes1.figure.canvas.draw()
        self.axes2.figure.canvas.draw()
        self.axes3.figure.canvas.draw()







def main():
    #os.chdir(os.path.dirname(os.path.abspath(__file__))) # to load the directory folder
    app = QtWidgets.QApplication(sys.argv)
    application = ApplicationWindow()
    application.show()
    app.exec_()


if __name__ == "__main__":
    main()
