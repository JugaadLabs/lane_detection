import numpy as np
import cv2
import pickle
import pdb

def drawBbox(frame,boxes):

    for box in boxes:
        # height,width,length = pred_box['dim']
        # x,y,z = pred_box['loc']
        x1,y1,x2,y2 = box['bbox']
        # height = x2-x1
        # width = y2-y1
        cv2.rectangle(frame,(x1,y1),(x2,y2),(255,00,0),3)




# load bboxes
with open('/home/lance/Desktop/jl/CenterTrack/src/tusimple1_detection.pkl', 'rb') as fp:
    pred_data = pickle.load(fp)



video_name = '/home/lance/Desktop/jl/sit_aware/test_videos/tusimple1.mp4'
cap = cv2.VideoCapture(video_name)

fourcc = cv2.VideoWriter_fourcc(*'XVID')
video_folder = '/home/lance/Desktop/jl/sit_aware/test_videos_output/'
out = cv2.VideoWriter(video_folder + 'tusimple1.avi',fourcc, 4, (1280,720))


cnt = 1

while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()

    if ret:
        # Display the resulting frame
        # cv2.imshow('frame',frame)
        pred_boxes = pred_data[cnt]
        # drawBbox(frame,pred_boxes)
        for box in pred_boxes:
            # height,width,length = pred_box['dim']
            # x,y,z = pred_box['loc']
            x1,y1,x2,y2 = box['bbox']
            # pdb.set_trace()
            # height = x2-x1
            # width = y2-y1
            cv2.rectangle(frame,(x1,y1),(x2,y2),(0,0,255),3)

            # cv2.rectangle(frame,(x2,y2),(x1,y1),(0,255,0),3)
            # cv2.rectangle(frame,(0,10),(100,100),(255,00,0),3)
        cv2.imshow('frame',frame)
        out.write(frame)
    else:
        break
    
    if cv2.waitKey(250) & 0xFF == ord('q'):
        break

    cnt+=1

# When everything done, release the capture
cap.release()
out.release()
cv2.destroyAllWindows()
