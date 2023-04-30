<<<<<<< HEAD
from math import ceil, floor

import time
from cv2 import COLOR_RGB2BGR
import numpy as np
import cv2
#import tensorflow as tf
import tflite_runtime.interpreter as tflite
import os

matrixCalibPath = "/home/pi/final-skripsi/camera/mtx.correction-2.npy"
distortionCalibPath = "/home/pi/final-skripsi/camera/dist.correction-2.npy"

t1 = 255/3
t2 = 255/2

mtxconst = np.load(matrixCalibPath)
distconst = np.load(distortionCalibPath)
newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtxconst, distconst, (1280,720), 1, (1280,720))
vid = cv2.VideoCapture(0)
#face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
face_cascade = cv2.CascadeClassifier('haar_alt')
#face_cascade = cv2.CascadeClassifier('haar_alt2')
#face_cascade = cv2.CascadeClassifier('haar_alt_three')
vid.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
vid.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
#vid.set(cv2.CAP_PROP_FRAME_WIDTH, 1024)
#vid.set(cv2.CAP_PROP_FRAME_HEIGHT, 768)
#train_model = tf.keras.models.load_model('trained_model/my_model')
interpreter = tflite.Interpreter("model-resnet50-relu-1.tflite")
#interpreter = tflite.Interpreter("model-relu-1-test.tflite")
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
#input_type = interpreter.get_input_details()[0]['dtype']
#print('inputframe: ', input_type)
#output_type = interpreter.get_output_details()[0]['dtype']
#print('output: ', output_type)

def testModel(intest, sampleNumber):
    testSubject = intest
    resultTally = {testSubject[0]:None,testSubject[1]:None}
    txtPrintImg = ["No frame",(255,255,255)]
    for subject in testSubject:
        i = 0
        h_idx = 0
        tally = np.zeros(2)
        input("\nPress enter to test Subject with {}".format(subject))
        #Clear old image buffer by reading it.
        while(i <= 4):
            vid.read()
            i += 1
        i = 0
        while(i<sampleNumber):
            norm = np.zeros((128,128))
            detection = False
            ret, frame = vid.read()
            if type(frame) == None or ret != True:
                pass
            else:
                frame = cv2.undistort(frame, mtxconst, distconst, None, newcameramtx)
                frame = cv2.flip(frame,0)
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                norm = cv2.normalize(gray,norm,0,255,cv2.NORM_MINMAX)
                face = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=4)
                fS_scale = 1.7
#                print("Getting sample: {}".format(i))
#                cv2.imwrite("/home/pi/test/{}.jpg".format(i),gray)

                fS_scaledown = 0.4
                #image_norm = cv2.normalize(gray, None, alpha=0,beta=200, norm_type=cv2.NORM_MINMAX)
                #image_norm = cv2.equalizeHist(gray)
                image_norm = frame
                
                for (x,y,w,h) in face:
                    if w < 128:
                        print("Rejected: Width is less than 128")
                        break
                    else:
                        tic = time.perf_counter()
                        detection = True
                        #img = cv2.rectangle(gray,(x,y),(x+w,y+h),(255,0,0),2)
                        #img = cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,244),2)
                        img = frame
                        center_x = x+w//2
                        center_y = -15+y+h//2
                        new_start = (int(center_x-(w/2*fS_scale)),int(center_y-(h/2*fS_scale)))
                        new_end = (int(center_x+(w/2*fS_scale)),int(center_y+(h/2*fS_scale)))
                        if(new_start[0]< 0 or new_start[1] < 0 or new_end[0] > frame.shape[1] or new_end[1] > frame.shape[0]):
                            pass
                        else:
                            #maskSize = np.ones((720,1280),dtype=np.uint8)
                            #maskSize[blackOutMaskStart[1]:blackOutMaskEnd[1],blackOutMaskStart[0]:blackOutMaskEnd[0]] = 0
                            #maskSize[y:y+h,x:x+w] = 0
                            #resizeMask = cv2.resize(maskSize[new_start[1]:new_end[1],new_start[0]:new_end[0]],(128,128),cv2.INTER_AREA)
                            #image_norm[y:y+h-50,x:x+w-50] = 0
                            resizeFace = cv2.resize(image_norm[new_start[1]:new_end[1],new_start[0]:new_end[0]],(128,128),cv2.INTER_AREA)
                            #blur = cv2.blur(resizeFace,(1,1))
                            blur = resizeFace
                            #visualCanny = cv2.Canny(image_norm[new_start[1]:new_end[1],new_start[0]:new_end[0]],t1,t2)
                            #visualCanny = image_norm[new_start[1]:new_end[1],new_start[0]:new_end[0]]
                            #visualCanny = visualCanny*maskSize[new_start[1]:new_end[1],new_start[0]:new_end[0]]
                            #detectOnRectangle = cv2.Canny(blur,t1,t2)
                            #cv2.imshow("maskres",detectOnRectangle)
                            #detectOnRectangle = detectOnRectangle*resizeMask
                            #cv2.imshow("realDetection", blur)
                            #cv2.imshow("maskcheck",resizeMask*255)

                            #fftx = np.fft.fft2(detectOnRectangle)
                            #fftshift = np.fft.fftshift(fftx).flatten('F')
                            #detectOnRectangle = np.abs(fftshift[0:ceil(fftx.size/2)].real)
                            #cutoffThr = np.amax(detectOnRectangle)/2
                            #detectOnRectangle[detectOnRectangle<cutoffThr] = 0

                            #if(len(img.shape) < 3):
                            #    anyhow = np.ndarray((visualCanny.shape[0],visualCanny.shape[1]))
                            #    anyhow[0:visualCanny.shape[0],0:visualCanny.shape[1]] = visualCanny
                            #else:
                            #    anyhow = np.ndarray((visualCanny.shape[0],visualCanny.shape[1],len(img.shape)))
                            #    anyhow[0:visualCanny.shape[0],0:visualCanny.shape[1],0] = visualCanny
                            #    anyhow[0:visualCanny.shape[0],0:visualCanny.shape[1],1] = visualCanny
                            #    anyhow[0:visualCanny.shape[0],0:visualCanny.shape[1],2] = visualCanny
                            #img[new_start[1]:new_end[1],new_start[0]:new_end[0]] = anyhow
                            #train_model.summary()
                            #print(detectOnRectangle.shape)
                            confidence = -100
                            inputframe = np.reshape(blur,(1,128,128,3))
                            # Test model on random inputframe data.
                            # Check if the inputframe type is quantized, then rescale inputframe data to uint8
                            #if input_details[0]['dtype'] == np.uint8:
                            #    input_scale, input_zero_point = input_details[0]["quantization"]
                            #    inputframe = inputframe / input_scale + input_zero_point
                            input_data = np.array(inputframe, dtype=np.float32)#/255.0
                            
                            interpreter.set_tensor(input_details[0]['index'], input_data)
                            
                            interpreter.invoke()
                            toc = time.perf_counter()
                            elapsed = toc-tic
                            print("Time taken: {} \n".format(elapsed))
                            # The function `get_tensor()` returns a copy of the tensor data.
                            # Use `tensor()` in order to get a pointer to the tensor.
                            output_data = interpreter.get_tensor(output_details[0]['index'])
                            output_data = output_data
        
                            print(confidence)
                            if output_data[0][0] > output_data[0][1] and output_data[0][0] > 0.65:
                                plus = np.array([1,0])
                                tally = np.add(tally,plus)
                                txtPrintImg = ["Helmet Detected",(0,255,0)]
                                i += 1
                            elif output_data[0][1] > output_data[0][0] and output_data[0][1] > 0.65:
                                plus = np.array([0,1])
                                tally = np.add(tally,plus)
                                txtPrintImg = ["Helmet not Detected",(0,0,255)]
                                i += 1
                            cv2.putText(img,txtPrintImg[0],(40,120),cv2.FONT_HERSHEY_SIMPLEX, 2, txtPrintImg[1],thickness=10)
                            print("Writing img {}-{}".format(i,subject))
                            cv2.imwrite("/home/pi/test/{}-{}.jpg".format(i,subject),img)
                            print("Output is {}".format(output_data[0]))

                # Display the resulting frame
                '''
                if(detection):
                    cv2.putText(img,txtPrintImg[0],(40,120),cv2.FONT_HERSHEY_SIMPLEX, 2, txtPrintImg[1],thickness=10)
                    print("Writing img {}-detected".format(i))
                    cv2.imwrite("/home/pi/test/{}-{}.jpg".format(i,subject),img)
                    cv2.imwrite("/home/pi/test/{}.jpg".format(i),img)
                '''
        resultTally[subject] = tally
    vid.release()
    tp = resultTally['helm'][0]
    fp = resultTally['helm'][1]
    tn = resultTally['no-helmet'][1]
    fn = resultTally['no-helmet'][0]
    print("Positive: {}, Negative: {}".format(intest[0],intest[1]))
    print("TP: {}, FP: {}, TN: {}, FN: {}".format(tp,fp,tn,fn))
    vid.release()
    #cv2.destroyAllWindows()
    input("\nPress enter to end test {}".format(subject))

testModel(['helm','no-helmet'],10)
=======
from math import ceil, floor
import time
from cv2 import COLOR_RGB2BGR
import numpy as np
import cv2
#import tensorflow as tf
import tflite_runtime.interpreter as tflite
import os

matrixCalibPath = "/home/pi/final-skripsi/camera/mtx.correction-2.npy"
distortionCalibPath = "/home/pi/final-skripsi/camera/dist.correction-2.npy"

t1 = 255/3
t2 = 255/2

mtxconst = np.load(matrixCalibPath)
distconst = np.load(distortionCalibPath)
newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtxconst, distconst, (1280,720), 1, (1280,720))
vid = cv2.VideoCapture(0)
#face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
face_cascade = cv2.CascadeClassifier('haar_alt')
#face_cascade = cv2.CascadeClassifier('haar_alt2')
#face_cascade = cv2.CascadeClassifier('haar_alt_three')
vid.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
vid.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
#vid.set(cv2.CAP_PROP_FRAME_WIDTH, 1024)
#vid.set(cv2.CAP_PROP_FRAME_HEIGHT, 768)
#train_model = tf.keras.models.load_model('trained_model/my_model')
interpreter = tflite.Interpreter("model-resnet50-relu-1.tflite")
#interpreter = tflite.Interpreter("model-relu-1-test.tflite")
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
#input_type = interpreter.get_input_details()[0]['dtype']
#print('inputframe: ', input_type)
#output_type = interpreter.get_output_details()[0]['dtype']
#print('output: ', output_type)

def testModel(intest, sampleNumber):
    testSubject = intest
    resultTally = {testSubject[0]:None,testSubject[1]:None}
    txtPrintImg = ["No frame",(255,255,255)]
    for subject in testSubject:
        i = 0
        h_idx = 0
        tally = np.zeros(2)
        input("\nPress enter to test Subject with {}".format(subject))
        while(i<sampleNumber):
            norm = np.zeros((128,128))
            detection = False
            ret, frame = vid.read()
            if type(frame) == None or ret != True:
                pass
            else:
                frame = cv2.undistort(frame, mtxconst, distconst, None, newcameramtx)
                frame = cv2.flip(frame,0)
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                norm = cv2.normalize(gray,norm,0,255,cv2.NORM_MINMAX)
                face = face_cascade.detectMultiScale(norm, scaleFactor=1.3, minNeighbors=3)
                fS_scale = 1.7

                fS_scaledown = 0.4
                #image_norm = cv2.normalize(gray, None, alpha=0,beta=200, norm_type=cv2.NORM_MINMAX)
                #image_norm = cv2.equalizeHist(gray)
                image_norm = frame
                
                for (x,y,w,h) in face:
                    if w < 128:
                        print("Rejected: Width is less than 128")
                        break
                    else:
                        tic = time.perf_counter()
                        detection = True
                        #img = cv2.rectangle(gray,(x,y),(x+w,y+h),(255,0,0),2)
                        #img = cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,244),2)
                        img = frame
                        center_x = x+w//2
                        center_y = -15+y+h//2
                        new_start = (int(center_x-(w/2*fS_scale)),int(center_y-(h/2*fS_scale)))
                        new_end = (int(center_x+(w/2*fS_scale)),int(center_y+(h/2*fS_scale)))
                        if(new_start[0]< 0 or new_start[1] < 0 or new_end[0] > 1179 or new_end[1] > 719):
                            pass
                        else:
                            #maskSize = np.ones((720,1280),dtype=np.uint8)
                            #maskSize[blackOutMaskStart[1]:blackOutMaskEnd[1],blackOutMaskStart[0]:blackOutMaskEnd[0]] = 0
                            #maskSize[y:y+h,x:x+w] = 0
                            #resizeMask = cv2.resize(maskSize[new_start[1]:new_end[1],new_start[0]:new_end[0]],(128,128),cv2.INTER_AREA)
                            #image_norm[y:y+h-50,x:x+w-50] = 0
                            resizeFace = cv2.resize(image_norm[new_start[1]:new_end[1],new_start[0]:new_end[0]],(128,128),cv2.INTER_AREA)
                            #blur = cv2.blur(resizeFace,(1,1))
                            blur = resizeFace
                            #visualCanny = cv2.Canny(image_norm[new_start[1]:new_end[1],new_start[0]:new_end[0]],t1,t2)
                            #visualCanny = image_norm[new_start[1]:new_end[1],new_start[0]:new_end[0]]
                            #visualCanny = visualCanny*maskSize[new_start[1]:new_end[1],new_start[0]:new_end[0]]
                            #detectOnRectangle = cv2.Canny(blur,t1,t2)
                            #cv2.imshow("maskres",detectOnRectangle)
                            #detectOnRectangle = detectOnRectangle*resizeMask
                            #cv2.imshow("realDetection", blur)
                            #cv2.imshow("maskcheck",resizeMask*255)

                            #fftx = np.fft.fft2(detectOnRectangle)
                            #fftshift = np.fft.fftshift(fftx).flatten('F')
                            #detectOnRectangle = np.abs(fftshift[0:ceil(fftx.size/2)].real)
                            #cutoffThr = np.amax(detectOnRectangle)/2
                            #detectOnRectangle[detectOnRectangle<cutoffThr] = 0

                            #if(len(img.shape) < 3):
                            #    anyhow = np.ndarray((visualCanny.shape[0],visualCanny.shape[1]))
                            #    anyhow[0:visualCanny.shape[0],0:visualCanny.shape[1]] = visualCanny
                            #else:
                            #    anyhow = np.ndarray((visualCanny.shape[0],visualCanny.shape[1],len(img.shape)))
                            #    anyhow[0:visualCanny.shape[0],0:visualCanny.shape[1],0] = visualCanny
                            #    anyhow[0:visualCanny.shape[0],0:visualCanny.shape[1],1] = visualCanny
                            #    anyhow[0:visualCanny.shape[0],0:visualCanny.shape[1],2] = visualCanny
                            #img[new_start[1]:new_end[1],new_start[0]:new_end[0]] = anyhow
                            #train_model.summary()
                            #print(detectOnRectangle.shape)
                            confidence = -100
                            inputframe = np.reshape(blur,(1,128,128,3))
                            # Test model on random inputframe data.
                            # Check if the inputframe type is quantized, then rescale inputframe data to uint8
                            #if input_details[0]['dtype'] == np.uint8:
                            #    input_scale, input_zero_point = input_details[0]["quantization"]
                            #    inputframe = inputframe / input_scale + input_zero_point
                            input_data = np.array(inputframe, dtype=np.float32)#/255.0
                            
                            interpreter.set_tensor(input_details[0]['index'], input_data)
                            
                            interpreter.invoke()
                            toc = time.perf_counter()
                            elapsed = toc-tic
                            print("Time taken: {} \n".format(elapsed))
                            # The function `get_tensor()` returns a copy of the tensor data.
                            # Use `tensor()` in order to get a pointer to the tensor.
                            output_data = interpreter.get_tensor(output_details[0]['index'])
                            output_data = output_data
        
                            print(confidence)
                            if output_data[0][0] > output_data[0][1] and output_data[0][0] > 0.65:
                                plus = np.array([1,0])
                                tally = np.add(tally,plus)
                                txtPrintImg = ["Helmet Detected",(0,255,0)]
                            elif output_data[0][1] > output_data[0][0] and output_data[0][1] > 0.65:
                                plus = np.array([0,1])
                                tally = np.add(tally,plus)
                                txtPrintImg = ["Helmet not Detected",(0,0,255)]
                            i += 1

                # Display the resulting frame
                if(detection):
                    cv2.putText(img,txtPrintImg[0],(40,120),cv2.FONT_HERSHEY_SIMPLEX, 2, txtPrintImg[1],thickness=10)
                    cv2.imwrite("/home/pi/test/{}.jpg".format(i),img)
        
        resultTally[subject] = tally
    vid.release()
    tp = resultTally['helm'][0]
    fp = resultTally['helm'][1]
    tn = resultTally['no-helmet'][1]
    fn = resultTally['no-helmet'][0]
    print("Positive: {}, Negative: {}".format(intest[0],intest[1]))
    print("TP: {}, FP: {}, TN: {}, FN: {}".format(tp,fp,tn,fn))
    vid.release()
    #cv2.destroyAllWindows()
    input("\nPress enter to end test {}".format(subject))

testModel(['helm','no-helmet'],10)
>>>>>>> e3542486d9fbad9d93dd5a6231b7e63aa421cdea
