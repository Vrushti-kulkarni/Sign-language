import cv2
import os
import numpy
import operator

from tensorflow.keras.models import model_from_json

with open('model_json',"r") as json_file:
    loaded_model_json = json_file.read()
model = model_from_json(loaded_model_json)
model.load_weights("model_json.weights.h5")


cap = cv2.VideoCapture(0)

labels = {0:'A', 1:'B', 2:'C'}

while True:  
    ret, frame = cap.read()  
    x1 = 10
    y1 = 10
    x2 = int(0.5 * frame.shape[1])
    y2 = int(0.5 * frame.shape[0])
    cv2.rectangle(frame, (x1-1, y1-1), (x2+1, y2+1), (0,255,0), 2)
    #initialise roi
    roi = frame[y1:y2, x1:x2]
    roi = cv2.resize(roi, (64,64))

    cv2.imshow('frame',frame)

    roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    _, test_image = cv2.threshold(roi, 120,255, cv2.THRESH_BINARY)    

    result = model.predict(test_image.reshape(1,64,64,1))
    cv2.imshow('test',test_image)

    predictions = {'A':result[0][0],'B' : result[0][1], 'L': result[0][2]}

    prediction = sorted(predictions.items(), key=operator.itemgetter(1), reverse = True)
    cv2.putText(frame, prediction[0][0],  (10, 120), cv2.FONT_HERSHEY_PLAIN, 2, (255,0,0), 3)
    cv2.putText(frame, prediction[0][0],  (10, 120), cv2.FONT_HERSHEY_PLAIN, 2, (255,0,0), 2)
    cv2.imshow("Frame", frame)

    if cv2.waitKey(10) == ord('e'):
        break

cap.release()
cv2.destroyAllWindows()