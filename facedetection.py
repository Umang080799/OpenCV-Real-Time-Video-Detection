import cv2, numpy,os,argparse

DEFAULT_OUTPUT_PATH = 'OpenCV Results/'
DEFAULT_CASCADE_INPUT_PATH = 'haarcascade_frontalface_default.xml'
car_cascade = cv2.CascadeClassifier('cars.xml')
fullbody_cascade = cv2.CascadeClassifier('haarcascade_fullbody.xml')
car_plate_cascade = cv2.CascadeClassifier('haarcascade_licence_plate_rus_16stages.xml')

class VideoCapture:
    
    def __init__(self):
        self.count = 0
        self.argsObj = Parse()
        self.faceCascade = cv2.CascadeClassifier(self.argsObj.input_path)
        self.videoSource = cv2.VideoCapture(0)
        
    def CaptureFrames(self):
        while True:
            
            ##Create a unique number for each frame
            frameNumber = '%08d' % (self.count) 
            
            ##Capturing Frame by Frame
            ret, frame = self.videoSource.read()
            
            ##Setting the screen color to grey, so that the haar cascade can easily detect the edges and the faces
            screenColor = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
            
            ##Customize how the cascade detects your face
            faces = self.faceCascade.detectMultiScale(
               screenColor,
               scaleFactor = 1.1,
               minNeighbors = 5,
               minSize = (30,30),
               flags = cv2.CASCADE_SCALE_IMAGE)
            ##Customize how the cascade detects your left eye
            car = car_cascade.detectMultiScale(screenColor,
               scaleFactor = 1.1,
               minNeighbors = 2,
               minSize = (20,20),
               flags = cv2.CASCADE_SCALE_IMAGE)

            ##Customize how the cascade detects your smile
            fullbody = fullbody_cascade.detectMultiScale(screenColor,
               scaleFactor = 1.1,
               minNeighbors = 3,
               minSize = (5,5),
               flags = cv2.CASCADE_SCALE_IMAGE) 

            ##Customize how the cascade detects the car plate
            car_plate = car_plate_cascade.detectMultiScale(screenColor,
               scaleFactor = 1.1,
               minNeighbors = 1,
               minSize = (1,1),
               flags = cv2.CASCADE_SCALE_IMAGE)

       

            
            ##Displaying the resulting frame
            cv2.imshow('Detecting Faces!!' , screenColor)
            
            ##If no faces are detected then the length of faces will be zero
            if len(faces) == 0:
                pass
            
            elif len(faces) > 0 :
           
                print 'Umang\'s app is detecting faces'
            
                ##Graphing the face and drawing the rectangle around it:
                for (x,y,w,h) in faces:
                    cv2.rectangle(frame, (x,y) , (x+w, y+h),(0,255,0), 4)
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    cv2.putText(frame,'Face',(x,y), font, 1, (0,0,0)) 

                ##Graphing the car plate and drawing the rectangle around it:
                for (x,y,w,h) in car_plate:
                    cv2.rectangle(frame, (x,y) , (x+w, y+h),(0,255,0), 4)
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    cv2.putText(frame,'Car Plate',(x,y), font, 1, (0,0,0)) 

                ##Graphing the car and drawing the rectangle around it:
                for (x,y,w,h) in car:
			cv2.rectangle(frame, (x,y) , (x+w, y+h),(255,20,147), 4)
			font = cv2.FONT_HERSHEY_SIMPLEX
			cv2.putText(frame,'Car',(x,y), font, 0.5, (0,0,0)) 

                ##Graphing the fullbody:
                for (x,y,w,h) in fullbody:
			cv2.rectangle(frame, (x,y) , (x+w, y+h),(255,20,147), 4)
			font = cv2.FONT_HERSHEY_SIMPLEX
		##	cv2.putText(frame,'Human Body',(x,y), font, 0.5, (0,0,0)) 


                               
                cv2.imwrite(DEFAULT_OUTPUT_PATH + frameNumber + '.png',frame)
                
            
            ##Incremeting count for each of the frames
            self.count += 1
            
            ##In every 1 millisecond, new frame is captured. As soon as the escape key is pressed,
            ##we exit out of the loop and no more frames are captured and the while loop ends
            if cv2.waitKey(1) == 27:
                break;
        
        ##When the recording is done, close the webcam connection and close all the windows.  
        self.videoSource.release() ##close the webcam
        cv2.waitKey(500)
        cv2.destroyAllWindows()
        cv2.waitKey(500)
        

def Parse():
    parser =  argparse.ArgumentParser(description='Cascade path for Face Detection')
    parser.add_argument('-i','--input_path', type = str , default = DEFAULT_CASCADE_INPUT_PATH,help = 'CASCADE input Path')
    parser.add_argument('-o','--output_path',type = str, default = DEFAULT_OUTPUT_PATH, help = 'Output path for pics taken')
    args = parser.parse_args()
    return args     

def ClearImageFolder():
    if not (os.path.exists(DEFAULT_OUTPUT_PATH)):
        os.makedirs(DEFAULT_OUTPUT_PATH)
        
    else:
        for files in os.listdir(DEFAULT_OUTPUT_PATH):
            filePath = os.path.join(DEFAULT_OUTPUT_PATH, files)
            if os.path.isfile(filePath):
                os.unlink(filePath)
            else:
                continue
        
        
def main():
    ##To clear the image slate 
    ClearImageFolder()
    
    ##Instantiate the class object
    facedetection = VideoCapture()
    
    ##Call CaptureFrames from the class to begin the class detection
    facedetection.CaptureFrames()
    

if __name__ == '__main__':
    main()
