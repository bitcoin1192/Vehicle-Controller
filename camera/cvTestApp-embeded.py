from math import ceil, floor
import time
import numpy as np
import cv2
import tensorflow as tf
import os

t1 = 255/3
t2 = 255/2

class ImageProcessor:
    def __init__(self, CVCaptureDevice) -> None:
        #Set camera source and it's resolution
        self.vid = CVCaptureDevice
        self.vid.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.vid.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    def __getImage(self):
        ret, frame = self.vid.read()
        if type(frame) == None or ret != True:
            #Spew no image captured error
            pass
        else:
            #Flip in y axis
            frame = cv2.flip(frame,1)
            return frame

    def ImagePreProcessing(self):
        #Convert camera input to opencv gray output
        gray = cv2.cvtColor(self.__getImage(), cv2.COLOR_BGR2GRAY)
        return gray
        #Normalize gray frame
        #norm = cv2.normalize(gray,norm,0,255,cv2.NORM_MINMAX)
        #image_norm = cv2.normalize(gray, None, alpha=0,beta=200, norm_type=cv2.NORM_MINMAX)
        #image_norm = cv2.equalizeHist(gray)
    

class FaceDetector:
    def __init__(self, ImageProcessor, minPixelSize, outputSize, setOuterBorder) -> None:
        #Set opencv clasifier file
        self.ImgProcess = ImageProcessor
        self.minPixelSize = minPixelSize
        self.fS_scale = setOuterBorder
        self.resizeToPixel = outputSize
        self.face_cascade = cv2.CascadeClassifier('/home/lumin0x1/Documents/kode-skripsi/raspberrypi-app/camera/haar_alt')
        self.tempImage = 0
    
    def getFace(self):
        grayImage = self.ImgProcess.ImagePreProcessing()
        faceImage = []
        facePosition = self.face_cascade.detectMultiScale(grayImage, scaleFactor=1.1, minNeighbors=6)
        for faceCoordinate in facePosition:
            x,y,width,height = faceCoordinate
            if width < self.minPixelSize:
                faceImage = 0
                print("Rejected: Width is less than 128")
            else:
                start, end = self.calculatePixelLocation(x,y,width,height)
                if start[0] < 0 or start[1] < 0 or end[0] < 0 or end[1] < 0:
                    pass
                else:
                    resizeImage = cv2.resize(grayImage[start[1]:end[1],start[0]:end[0]],(128,128),cv2.INTER_AREA)
                    faceImage.append(resizeImage)
        self.tempImage = faceImage
        return faceImage
    
    def getCanny(self):
        cannyImage = []
        if self.tempImage != 0:
            for image in self.tempImage:
                cannyImage.append(cv2.Canny(image,t1,t2))
        return cannyImage
    
    def calculatePixelLocation(self, x,y,w,h):
        center_x = x+w//2
        center_y = y+h//2
        new_start = (int(center_x-(w/2*self.fS_scale)),int(center_y-(h/2*self.fS_scale)))
        new_end = (int(center_x+(w/2*self.fS_scale)),int(center_y+(h/2*self.fS_scale)))
        return new_start,new_end


class HelmetDetector:
    def __init__(self, FaceDetector) -> None:        
        #Set tflite model and it's input and output details
        self.FaceDetection = FaceDetector
        self.interpreter = tf.lite.Interpreter("/home/lumin0x1/Documents/kode-skripsi/raspberrypi-app/camera/model-relu-3.tflite")
        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
    
    def debug(self, resizeImage, cannyImage):
        cv2.imshow("debug-1", resizeImage)
        cv2.imshow("debug-2", cannyImage)
        cv2.waitKey(10)

    def _thresholdFFT(self,TwoDInput):
        fftx = np.fft.fft2(TwoDInput)
        fftshift = np.fft.fftshift(fftx).flatten('F')
        detectOnRectangle = np.abs(fftshift[0:ceil(fftx.size/2)].real)
        cutoffThr = np.amax(detectOnRectangle)/2
        detectOnRectangle[detectOnRectangle<cutoffThr] = 0
        #return 1D FFT value
        return detectOnRectangle

    def _dataTypeConversion(self, OneDInput):
        #Transpose input array needed for prediction
        inputframe = np.reshape(OneDInput,(1,8192,))
        #Check if the inputframe type is quantized, then rescale inputframe data to uint8
        if self.input_details[0]['dtype'] == np.uint8:
            input_scale, input_zero_point = self.input_details[0]["quantization"]
            inputframe = inputframe / input_scale + input_zero_point
        #Return unsigned int8 one dimensional array
        return np.array(inputframe, dtype=np.uint8)

    def _predictionPreProcess(self,inputFrame):
        #FFT with thresholding process
        FlattenFFT = self._thresholdFFT(inputFrame)
        #Data Type Conversion to convert 2D array to tf array needed by model
        return self._dataTypeConversion(FlattenFFT)

    def runPrediction(self):
        faceImage = self.FaceDetection.getFace()
        cannyImage = self.FaceDetection.getCanny()
        for face in cannyImage:
            self.debug(faceImage[0],face)
            input_data = self._predictionPreProcess(face)
            self.interpreter.set_tensor(self.input_details[0]['index'], input_data)
            # Run model calculation
            self.interpreter.invoke()
            # The function `get_tensor()` returns a copy of the tensor data.
            # Use `tensor()` in order to get a pointer to the tensor.
            output_data = self.interpreter.get_tensor(self.output_details[0]['index'])
            # As per tensorflow documentation for model with quantize output 
            output_data = output_data/255
            return output_data
    
    def helmetUsed(self, sample):
        tally = np.zeros(2)
        i = 0
        while(i < sample):
            output_data = self.runPrediction()
            if output_data is not None:
                if output_data[0][0] > 0.95:
                    plus = np.array([1,0])
                    tally = np.add(tally,plus)
                    txtPrintImg = ["Helmet Detected",(0,255,0)]
                    i += 1
                elif output_data[0][1] > 0.95:
                    plus = np.array([0,1])
                    tally = np.add(tally,plus)
                    txtPrintImg = ["Helmet not Detected",(0,0,255)]
                    i += 1
        if tally[0]<tally[1]:
            return False
        else:
            return True
td = ImageProcessor(cv2.VideoCapture(0))
fd = FaceDetector(td,128,128,1.7)
hd = HelmetDetector(fd)
while(True):
    result = hd.helmetUsed(26)
    print(result)
cv2.destroyAllWindows()