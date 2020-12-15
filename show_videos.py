import numpy as np
import cv2
import pickle
import pdb

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
        # draw 2D bbox
        pred_boxes = pred_data[cnt]
        for box in pred_boxes:
            x1,y1,x2,y2 = box['bbox']
            cv2.rectangle(frame,(x1,y1),(x2,y2),(0,0,255),3)
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
