import cv2
import os
import time

#to store all the data in a directory called data
direct = 'C:/Users/JYOTI/Downloads/DEEP_LEARNING/sign_language/data/train'

#to check if this directory data is already made, if not create new
if not os.path.exists(direct):
    os.makedirs(direct)

classes = 3
dataset_size = 70

#for capturing video, 0 index for default camera, higher index for other camera


#now we make a loop such that for every class or label it should collect 100 images
cap = cv2.VideoCapture(0)
for j in range(classes):
    print(j)
    #basically to check if directory of that class is present in our main directory
    #join is basically making a directory in base directory
    if not os.path.exists(os.path.join(direct, str(j))):
        os.makedirs(os.path.join(direct, str(j)))
    
    #print("collecting data for class {}".format(j))
    cap = cv2.VideoCapture(0)
    #ROI
    ret, frame = cap.read()
    #(x1, y1): Represents the top-left corner of the rectangle.
    #(x2, y2): Represents the bottom-right corner of the rectangle.
    x1 = 10
    y1 = 10
    x2 = int(0.5 * frame.shape[1])
    y2 = int(0.5 * frame.shape[0])
    #frame.shape[0]: The height of the frame (the number of rows, or the number of pixels vertically).
    #frame.shape[1]: The width of the frame (the number of columns, or the number of pixels horizontally).
    #frame.shape[2]: The number of channels in the image (typically 3 for RGB images).

    cv2.rectangle(frame, (x1-1, y1-1), (x2+1, y2+1), (0,255,0), 2)

    while cap.isOpened():
        ret, frame = cap.read()
        cv2.rectangle(frame, (x1-1, y1-1), (x2+1, y2+1), (0,255,0), 2)
        cv2.putText(frame, 'get ready!', (100, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 0, 0), 3, cv2.LINE_AA)
        cv2.imshow('frame', frame)

        
        if cv2.waitKey(10)==ord('q'):
            break
    
    for i in range(0,dataset_size):
        #to capture image, ret is boolean, and frame stores the image
        ret, frame = cap.read()
        cv2.rectangle(frame, (x1-1, y1-1), (x2+1, y2+1), (0,255,0), 2)
        #initialise roi
        roi = frame[y1:y2, x1:x2]
        roi = cv2.resize(roi, (64,64))

        cv2.imshow('frame',frame)

        roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        _,roi = cv2.threshold(roi, 120,255, cv2.THRESH_BINARY)
        cv2.imshow('ROI', roi)
        cv2.resizeWindow('ROI', 300, 300) 
        cv2.waitKey(25)
        cv2.imwrite(os.path.join(direct, str(j), '{}.jpg'.format(i)), roi)
        
     
cap.release()
cv2.destroyAllWindows()

#to store all the data in a directory called data
direct = 'C:/Users/JYOTI/Downloads/DEEP_LEARNING/sign_language/data/test'

#to check if this directory data is already made, if not create new
if not os.path.exists(direct):
    os.makedirs(direct)

classes = 3
dataset_size = 30

cap = cv2.VideoCapture(0)
for j in range(classes):
    print(j)
    #basically to check if directory of that class is present in our main directory
    #join is basically making a directory in base directory
    if not os.path.exists(os.path.join(direct, str(j))):
        os.makedirs(os.path.join(direct, str(j)))
    
    #print("collecting data for class {}".format(j))
    cap = cv2.VideoCapture(0)
    #ROI
    ret, frame = cap.read()
    #(x1, y1): Represents the top-left corner of the rectangle.
    #(x2, y2): Represents the bottom-right corner of the rectangle.
    x1 = 10
    y1 = 10
    x2 = int(0.5 * frame.shape[1])
    y2 = int(0.5 * frame.shape[0])
    #frame.shape[0]: The height of the frame (the number of rows, or the number of pixels vertically).
    #frame.shape[1]: The width of the frame (the number of columns, or the number of pixels horizontally).
    #frame.shape[2]: The number of channels in the image (typically 3 for RGB images).

    cv2.rectangle(frame, (x1-1, y1-1), (x2+1, y2+1), (0,255,0), 2)

    while cap.isOpened():
        ret, frame = cap.read()
        cv2.rectangle(frame, (x1-1, y1-1), (x2+1, y2+1), (0,255,0), 2)
        cv2.putText(frame, 'get ready!', (100, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 0, 0), 3, cv2.LINE_AA)
        cv2.imshow('frame', frame)

        
        if cv2.waitKey(10)==ord('q'):
            break
    
    for i in range(0,dataset_size):
        #to capture image, ret is boolean, and frame stores the image
        ret, frame = cap.read()
        cv2.rectangle(frame, (x1-1, y1-1), (x2+1, y2+1), (0,255,0), 2)
        #initialise roi
        roi = frame[y1:y2, x1:x2]
        roi = cv2.resize(roi, (64,64))

        cv2.imshow('frame',frame)

        roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        _,roi = cv2.threshold(roi, 120,255, cv2.THRESH_BINARY)
        cv2.imshow('ROI', roi)
        cv2.resizeWindow('ROI', 300, 300) 
        cv2.waitKey(25)
        cv2.imwrite(os.path.join(direct, str(j), '{}.jpg'.format(i)), roi)
        
     
cap.release()
cv2.destroyAllWindows()