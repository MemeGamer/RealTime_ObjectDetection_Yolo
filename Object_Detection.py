import cv2
import numpy as np
from gtts import gTTS
import os
from playsound import playsound
import time

language = 'en'
net = cv2.dnn.readNet('yolov3.weights','yolov3.cfg')
classes = []
with open('coco.names','r') as f:
    classes = f.read().splitlines()

cap = cv2.VideoCapture(0)
#address = "https://192.168.1.7:8080/shot.jpg"
#cap.open(address)
#img = cv2.imread('240Nw.jpg')

while True:
    _, img =cap.read()
    height, width,_= img.shape

    blob = cv2.dnn.blobFromImage(img,1/255,(416,416),(0,0,0),swapRB=True, crop=False)
    net.setInput(blob)
    output_layers_names = net.getUnconnectedOutLayersNames()
    layerOutputs = net.forward(output_layers_names)

    boxes = []
    confidences = []
    class_ids = []

    for output in layerOutputs:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:
                center_x = int(detection[0]*width)
                center_y = int(detection[1]*height)
                w = int(detection[2]*width)
                h = int(detection[3]*height)

                x = int(center_x - w/2)
                y = int(center_y - h/2)

                boxes.append([x,y,w,h])
                confidences.append((float(confidence)))
                class_ids.append(class_id)


    indexes = cv2.dnn.NMSBoxes(boxes,confidences,0.5,0.4)

    font = cv2.FONT_HERSHEY_PLAIN
    colors = np.random.uniform(0,255,size=(len(boxes),3))

    if len(indexes) > 0:
        for i in indexes.flatten():
            x,y,w,h = boxes[i]
            label = str(classes[class_ids[i]])
            print(label)
            output = gTTS(text=label, lang=language, slow=False)
            output.save("output.mp3")
            playsound('output.mp3')
            time.sleep(1)
            os.remove("output.mp3")
            confidence = str(round(confidences[i],2))
            color = colors[i]
            cv2.rectangle(img,(x,y),(x+w, y+h), color, 2)
            cv2.putText(img, label +" "+ confidence,(x,y+20), font,2,(255,255,255),2)
    cv2.imshow('Image',img)
    key = cv2.waitKey(1)
    if key==27:
        break
cap.release()
cv2.destroyAllWindows()

