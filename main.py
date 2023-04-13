import face_recognition as fr #RGB
import cv2 #BGR
import numpy as np
import csv
from datetime import datetime


vid_cap=cv2.VideoCapture(0)
arush_image=fr.load_image_file("ArushPic.png")
#reads and loads the image file
vihaan_image=fr.load_image_file("VihaanPic.png")
arush_image_encoding = fr.face_encodings(arush_image)[0]
#reads and returns the RGB values of every pixels in one dimension
vihaan_image_encoding = fr.face_encodings(vihaan_image)[0]

known_face_encodings=[arush_image_encoding, vihaan_image_encoding]
known_face_names=["Arush", "Vihaan"]
#stores the values of the name

face_locations=[]
face_encoding=[]
face_names=[]
process_this_frame=True
present_list= []
while True:
    ret,frame=vid_cap.read()
    small_frame=cv2.resize(frame,(0,0), fx=0.25, fy=0.25)
    #resizes the frame to reduce the resolution
    rgb_small_frame=cv2.cvtColor(small_frame,cv2.COLOR_BGR2RGB)
    #converts bgr to rgb to ensure image is correct
    if process_this_frame:
        face_locations=fr.face_locations(rgb_small_frame)
        #methods passes the changing the color
        face_encodings=fr.face_encodings(rgb_small_frame, face_locations)
        #wherever the face location is in the frame, stores the code for the pixel
        face_names=[]
        for face_encoding in face_encodings:
            matches=fr.compare_faces(known_face_encodings, face_encoding)
            #returns boolean value if it matches
            
            for i in range(len(matches)-1):
                if matches[i]== True:
                    present_list.append(known_face_names[i])                    
        
            name= "Unknown"

            face_distances= fr.face_distance(known_face_encodings,face_encoding)
            #calculates distances
            best_match=np.argmin(face_distances)
            #stores closes to match, not exact, but similar
            if matches[best_match]: #the index of the best match:
                name=known_face_names[best_match]
                #stores the name of the best match
                face_names.append(name)
    process_this_frame=not process_this_frame
    
    for (top,right,bottom,left),name in zip(face_locations,face_names):
        top*=4
        right*=4
        bottom*=4
        left*=4
        #scales the image back

        cv2.rectangle(frame,(left,top),(right,bottom),(0,0,255),2)
        #draws a triangle using the top left and bottom right point
        cv2.rectangle(frame,(left, bottom-35),(right,bottom),(0,0,255),-1)
        #draws a filled box
        font=cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame,name,(left+6,bottom-6),font,1.0,(255,255,255),1)
        #puts the text on the frame
            
    cv2.imshow("faces",frame)
    if cv2.waitKey(1) & 0xFF==ord("q"):
        break
    
singleList= list(set(present_list))
#prints 1 occurence of the name to show if present


with open(r'present.csv', 'r+', newline='') as file:
  
    # creating the csv writer
    file_write = csv.writer(file)

    # storing current date and time
    current_date_time = datetime.now()
  
    # Iterating over all the data in the rows 
    # variable
    for val in singleList:
        # Inserting the date and time at 0th 
        # index
        val.insert(0, current_date_time)
        file_write.writerow(val)


vid_cap.release()
cv2.destroyAllWindows()



        
