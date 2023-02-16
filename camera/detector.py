from math import ceil, floor
import time
import numpy as np
import cv2
import tensorflow as tf

t1 = 255/3
t2 = 255/2
#cvHaarPath = "/home/lumin0x1/Documents/kode-skripsi/raspberrypi-app/camera/haar_alt"
#tflitePath = "/home/lumin0x1/Documents/kode-skripsi/raspberrypi-app/camera/model-relu-3.tflite"
#matrixCalibPath = "/home/lumin0x1/Documents/kode-skripsi/raspberrypi-app/camera/mtx.correction.npy"
#distortionCalibPath = "/home/lumin0x1/Documents/kode-skripsi/raspberrypi-app/camera/dist.correction.npy"
cvHaarPath = "/home/pi/final-skripsi/camera/haar_alt"
tflitePath = "/home/pi/final-skripsi/camera/model-relu-3-test.tflite"
matrixCalibPath = "/home/pi/final-skripsi/camera/mtx.correction.npy"
distortionCalibPath = "/home/pi/final-skripsi/camera/dist.correction.npy"

debug = False

class ImageProcessor:
    def __init__(self, CVCaptureDevice) -> None:
        #Set camera source and it's resolution
        self.vid = CVCaptureDevice
        self.vid.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.vid.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        self.mtxconst = np.load(matrixCalibPath)
        self.distconst = np.load(distortionCalibPath)
        self.newcameramtx, self.roi = cv2.getOptimalNewCameraMatrix(self.mtxconst, self.distconst, (1280,720), 1, (1280,720))
        self.debugFlag = True
        self.toc = 0
        self.tic = 0

    def __getImage(self):
        retry = 0
        while(retry < 5):
            ret, frame = self.vid.read()
            time.sleep(0.2)
            #self.debug()
            if type(frame) != None or ret == True:
                x,y,w,h = self.roi
                dst = cv2.undistort(frame, self.mtxconst, self.distconst, None, self.newcameramtx)
                dst = dst[y:y+h,x:x+w]
                frame = cv2.flip(dst,0)
                #frame = dst
                return frame
            retry += 1
            print("Retrying to fetch image %d" % retry)                
        raise None
    def ImagePreProcessing(self):
        #Convert camera input to opencv gray output
        gray = cv2.cvtColor(self.__getImage(), cv2.COLOR_RGB2GRAY)
        return gray
        #Normalize gray frame
        #norm = cv2.normalize(gray,norm,0,255,cv2.NORM_MINMAX)
        #image_norm = cv2.normalize(gray, None, alpha=0,beta=200, norm_type=cv2.NORM_MINMAX)
        #image_norm = cv2.equalizeHist(gray)
    
    def debug(self):
        if self.debugFlag is True:
            self.toc = self.tic
            self.tic = time.time()
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
        self.debugFlag = debug
        self.frameTotal = 1
    
    def getFace(self):
        grayImage = self.ImgProcess.ImagePreProcessing()
        faceImage = []
        facePosition = self.face_cascade.detectMultiScale(grayImage, scaleFactor=1.3, minNeighbors=1)
        for faceCoordinate in facePosition:
            x,y,width,height = faceCoordinate
            if width < self.minPixelSize:
                faceImage.clear()
                print("Rejected: Width is less than 128")
            else:
                start, end = self.calculatePixelLocation(x,y,width,height)
                if start[0] < 0 or start[1] < 0 or end[0] < 0 or end[1] < 0:
                    pass
                else:
                    print("Detecting face")
                    resizeImage = cv2.resize(grayImage[start[1]:end[1],start[0]:end[0]],(128,128),cv2.INTER_AREA)
                    faceImage.append(resizeImage)
        self.tempImage = faceImage
        return faceImage
    
    def getCanny(self):
        cannyImage = []
        if len(self.tempImage) != 0:
            for image in self.tempImage:
                temp = cv2.Canny(image,t1,t2)
                cannyImage.append(cv2.Canny(image,t1,t2))
                self.debug(temp)
        return cannyImage
    
    def calculatePixelLocation(self, x,y,w,h):
        center_x = x+w//2
        center_y = y+h//2
        new_start = (int(center_x-(w/2*self.fS_scale)),int(center_y-(h/2*self.fS_scale)))
        new_end = (int(center_x+(w/2*self.fS_scale)),int(center_y+(h/2*self.fS_scale)))
        return new_start,new_end

    def debug(self,imgArray):
        cv2.imwrite("result-debug/image-{}.jpg".format(self.frameTotal),imgArray)
        print("Debug: Writing frame {} to debug folder".format(self.frameTotal))  
        self.frameTotal += 1


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
        self.interpreter = tf.lite.Interpreter(tflitePath)
        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
    
    def debug(self, resizeImage, cannyImage):
        #cv2.imshow("debug-1", resizeImage)
        #cv2.imshow("debug-2", cannyImage)
        #cv2.waitKey(10)
        pass

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
        inputframe = np.reshape(OneDInput,(1,128,128,1))
        #Check if the inputframe type is quantized, then rescale inputframe data to uint8
        #if self.input_details[0]['dtype'] == np.uint8:
        #    input_scale, input_zero_point = self.input_details[0]["quantization"]
        #    inputframe = inputframe / input_scale + input_zero_point
        #Return unsigned int8 one dimensional array
        return np.array(inputframe, dtype=np.float32)

    def _predictionPreProcess(self,inputFrame):
        #FFT with thresholding process
        FlattenFFT = self._thresholdFFT(inputFrame)
        #Data Type Conversion to convert 2D array to tf array needed by model
        return self._dataTypeConversion(FlattenFFT)

    def blackCenteredFrame():
        pass

    def runPrediction(self):
        faceImage = self.FaceDetection.getFace()
        cannyImage = self.FaceDetection.getCanny()
        for face in faceImage:
            self.debug(faceImage[0],face)
            input_data = self._predictionPreProcess(face)
            self.interpreter.set_tensor(self.input_details[0]['index'], input_data)
            # Run model calculation
            self.interpreter.invoke()
            # The function `get_tensor()` returns a copy of the tensor data.
            # Use `tensor()` in order to get a pointer to the tensor.
            output_data = self.interpreter.get_tensor(self.output_details[0]['index'])
            # As per tensorflow documentation for model with quantize output 
            output_data = output_data
            return output_data
    
    def detectHelmet(self):
        output_data = self.runPrediction()
        if output_data is not None:
            if output_data[0][0] > self.confidence:
                plus = np.array([1,0])
                self.tallyCounter = np.add(self.tallyCounter,plus)
                txtPrintImg = ["Helmet Detected",(0,255,0)]
                print(txtPrintImg[0]) 
            elif output_data[0][1] > self.confidence:
                plus = np.array([0,1])
                self.tallyCounter = np.add(self.tallyCounter,plus)
                txtPrintImg = ["Helmet not Detected",(0,0,255)]
                print(txtPrintImg[0])
            self.counter += 1

        
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

