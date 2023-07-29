from math import ceil, floor
import time
import numpy as np
import cv2
#import tensorflow as tf
import tflite_runtime.interpreter as tflite

t1 = 255/3
t2 = 255/2
#cvHaarPath = "/home/lumin0x1/Documents/kode-skripsi/raspberrypi-app/camera/haar_alt"
#tflitePath = "/home/lumin0x1/Documents/kode-skripsi/raspberrypi-app/camera/model-relu-3-test (5).tflite"
#matrixCalibPath = "/home/lumin0x1/Documents/kode-skripsi/raspberrypi-app/camera/mtx.correction.npy"
#distortionCalibPath = "/home/lumin0x1/Documents/kode-skripsi/raspberrypi-app/camera/dist.correction.npy"
cvHaarPath = "/home/pi/final-skripsi/camera/haar_alt"
tflitePath = "/home/pi/final-skripsi/camera/model-resnet50-relu-1.tflite"
matrixCalibPath = "/home/pi/final-skripsi/camera/mtx.correction-2.npy"
distortionCalibPath = "/home/pi/final-skripsi/camera/dist.correction-2.npy"

#Global Variable
debugFlagDetector = True
frameTotal = 0

class ImageProcessor:
    def __init__(self, CVCaptureDevice) -> None:
        #Set camera source and it's resolution
        self.vid = CVCaptureDevice
        self.vid.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.vid.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        self.vid.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.75)
        self.mtxconst = np.load(matrixCalibPath)
        self.distconst = np.load(distortionCalibPath)
        self.newcameramtx, self.roi = cv2.getOptimalNewCameraMatrix(self.mtxconst, self.distconst, (1280,720), 1, (1280,720))
        self.toc = 0
        self.tic = 0

    def __getImage(self):
        retry = 0
        while(retry < 5):
            ret, frame = self.vid.read()
            #self.debug()
            if ret == True:
                x,y,w,h = self.roi
                dst = cv2.undistort(frame, self.mtxconst, self.distconst, None, self.newcameramtx)
                #dst = dst[y:y+h,x:x+w]
                frame = cv2.flip(dst,0)
#                print("Reading camera")
                #frame = dst
                return frame
            retry += 1
            print("Retrying to fetch image %d" % retry)
        self.vid.release()
        raise Exception("Failed to fetch image from camera")
        
    def ImagePreProcessing(self):
        #Convert camera input to opencv gray output
        global debugFlagDetector
        global frameTotal
        inputImage = self.__getImage()
        frameTotal += 1
        norm = np.zeros((128,128))
        gray = cv2.cvtColor(inputImage, cv2.COLOR_RGB2GRAY)
        norm = cv2.normalize(gray,norm,0,255,cv2.NORM_MINMAX)
        if debugFlagDetector:
            self.debug()
        return (gray,inputImage)
        #Normalize gray frame
        #norm = cv2.normalize(gray,norm,0,255,cv2.NORM_MINMAX)
        #image_norm = cv2.normalize(gray, None, alpha=0,beta=200, norm_type=cv2.NORM_MINMAX)
        #image_norm = cv2.equalizeHist(gray)
    
    def debug(self):
        self.toc = self.tic
        self.tic = time.time()
        #print("debug is {}".format(debugFlagDetector))
        print("I'm fetching image at {} fps".format(1000/((self.tic-self.toc)*1000)))
    

class FaceDetector:
    def __init__(self, ImageProcessor, minPixelSize, outputSize, setOuterBorder) -> None:
        #Set opencv clasifier file
        self.ImgProcess = ImageProcessor
        self.minPixelSize = minPixelSize
        self.fS_scale = setOuterBorder
        self.resizeToPixel = outputSize
        self.face_cascade = cv2.CascadeClassifier(cvHaarPath)
        self.tempImage = 0
    
    def getFace(self):
        global debugFlagDetector
        imageTupple = self.ImgProcess.ImagePreProcessing()
        grayImage = imageTupple[0]
        colorImage = imageTupple[1]
        #self.debug(grayImage)
        faceImage = []
        facePosition = self.face_cascade.detectMultiScale(grayImage, scaleFactor=1.3, minNeighbors=3)
        if len(facePosition) == 0 and debugFlagDetector:
            self.debug(colorImage,"NOFACE")
        for faceCoordinate in facePosition:
            x,y,width,height = faceCoordinate
            if width < self.minPixelSize:
                faceImage.clear()
                if debugFlagDetector:
                    self.debug(colorImage,"SMALL")
                print("Rejected: Width is less than 128")
            else:
                start, end = self.calculatePixelLocation(x,y,width,height)
                if start[0] < 0 or start[1] < 0 or end[0] > 1280 or end[1] > 720:
                    if debugFlagDetector:
                        self.debug(colorImage,"OOB")
                    pass
                else:
                    #print("Face Detected, Continue to Helmet Detection")
                    resizeImage = cv2.resize(colorImage[start[1]:end[1],start[0]:end[0]],(128,128),cv2.INTER_AREA)
                    faceImage.append(resizeImage)
        self.tempImage = faceImage
        return faceImage
    
    def calculatePixelLocation(self, x,y,w,h):
        center_x = x+w//2
        center_y = -15+y+h//2
        new_start = (int(center_x-(w/2*self.fS_scale)),int(center_y-(h/2*self.fS_scale)))
        new_end = (int(center_x+(w/2*self.fS_scale)),int(center_y+(h/2*self.fS_scale)))
        return new_start,new_end

    def debug(self,img,msg):
        global frameTotal
        cv2.imwrite("/home/pi/result-debug/facedetector-{}-{}.jpg".format(frameTotal,msg),img)
        print("Debug: Writing frame {} to debug folder".format(frameTotal))  
        #frameTotal += 1


class HelmetDetector:
    def __init__(self, FaceDetector, ConfidenceThreshold, SampleNeeded) -> None:        
        #Set tflite model and it's input and output details
        self.sampleNeeded = SampleNeeded
        self.confidence = ConfidenceThreshold
        self.FaceDetection = FaceDetector
        self.currentHelmetStatus = False
        self.stopFlag = False
        self.counter = 0
        self.tallyCounter = np.zeros(2)
        self.interpreter = tflite.Interpreter(tflitePath)
        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
    
    def debug(self, resizeImage, detectResult):
        global frameTotal
        cv2.imwrite("/home/pi/result-debug/helmetdetector-{}-{}.jpg".format(frameTotal,detectResult),resizeImage)
        print("Debug: Writing helmet detection frame {} to debug folder".format(detectResult))


    def _dataTypeConversion(self, TwoDInput):
        #Reshape for model input, resnet50 require rgb image. In total 3 channel is needed
        inputframe = np.reshape(TwoDInput,(1,128,128,3))
        #Resnet50 is already equipped with image scaler, no need for scaling here.
        return np.array(inputframe, dtype=np.float32)

    def _predictionPreProcess(self,inputFrame):
        #Data Type Conversion to convert 2D array to tf array needed by model
        return self._dataTypeConversion(inputFrame)

    def runPrediction(self):
        faceImage = self.FaceDetection.getFace()
        for face in faceImage:
            #self.debug(faceImage[0],face)
            input_data = self._predictionPreProcess(face)
            self.interpreter.set_tensor(self.input_details[0]['index'], input_data)
            # Run model calculation
            self.interpreter.invoke()
            # The function `get_tensor()` returns a copy of the tensor data.
            # Use `tensor()` in order to get a pointer to the tensor.
            output_data = self.interpreter.get_tensor(self.output_details[0]['index'])
            # As per tensorflow documentation for model with quantize output 
            output_data = output_data
            return (output_data, face)
        return(None,None)

    def detectHelmet(self):
        global debugFlagDetector
        output_data,face = self.runPrediction()
        if output_data is not None:
            if output_data[0][0] > output_data[0][1] and output_data[0][0] > self.confidence:
                plus = np.array([1,0])
                self.tallyCounter = np.add(self.tallyCounter,plus)
                txtPrintImg = ["Helmet Detected",(0,255,0)]
                if debugFlagDetector:
                    self.debug(face,"helmet-{}".format(output_data[0][0]))
                print(txtPrintImg[0]) 
            elif output_data[0][1] > output_data[0][0] and output_data[0][1] > self.confidence:
                plus = np.array([0,1])
                self.tallyCounter = np.add(self.tallyCounter,plus)
                txtPrintImg = ["Helmet not Detected",(0,0,255)]
                if debugFlagDetector:
                    self.debug(face,"nohelmet-{}".format(output_data[0][1]))
                print(txtPrintImg[0])
            else:
                if debugFlagDetector:
                    self.debug(face,"rejected-{}-{}".format(output_data[0][0],output_data[0][1]))
            self.counter += 1

    """This function is called from helmetAnnouncer.py to get the current helmet status.
       It will return True if helmet is detected, False if helmet is not detected. 
       If sampleNeeded is less that specified, it will return last known status.
       It has intial state of False"""        
    def tallyResult(self):
        if self.stopFlag == False:
            if self.counter >= self.sampleNeeded:
                self._makeDecision()
                self.resetCounter()
            else:
                self.detectHelmet()
        return self.currentHelmetStatus

    def _makeDecision(self):
        if self.tallyCounter[0] < self.tallyCounter[1]:
            self.currentHelmetStatus = False
        else:
            self.currentHelmetStatus = True
        print(self.tallyCounter)

    def resetCounter(self):
        self.counter = 0
        self.tallyCounter = np.zeros(2)

    def stopFlags(self, input):
        self.stopFlag = input
        if(self.stopFlag == True):
            self.currentHelmetStatus = False

