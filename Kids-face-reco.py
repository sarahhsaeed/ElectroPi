# imports
import cv2
from deepface import DeepFace
from retinaface import RetinaFace



def Kids_Recognition(video_path):
    cap = cv2.VideoCapture(video_path)
    """
    video_path: path -> the video u wanna process
    """
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Detect faces using RetinaFace
        faces = RetinaFace.detect_faces(frame) # git the faces from the fram 
        
        # get face coordinates and give it to deepface to find kid name from the DataBase
        for key, face in faces.items():

            facial_area = face["facial_area"]
            x, y, width, height = facial_area[0], facial_area[1], facial_area[2]-facial_area[0], facial_area[3]-facial_area[1]


            # Extract the image data from the bounding box coordinates
            face_img = frame[y:y+height, x:x+width]

            find_label = DeepFace.find(
                img_path = face_img, 
                db_path = "C:/Users/Sarah saeed/Desktop/gurndian angel vscode env/DB and vid/DB and vid/DataB",
                enforce_detection = False,
                )

            label = list(find_label[0]['identity'].values[0])
            name =  str(label[-7]) # git the kid name from the DataBase

            cv2.rectangle(frame, (x, y), (x + width, y + height), (0, 255, 0), 2)

            # but the kid name on the rectangle
            cv2.putText(frame, name, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2) 

        cv2.imshow("Kids Recognition", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()



video_path = "C:/Users/Sarah saeed\Desktop/gurndian angel vscode env/DB and vid/DB and vid/try.mp4"
Kids_Recognition(video_path)




"""

# load yolov8s model
model = YOLO('yolov8s.pt')

# load video
video_path = './Data4_for_yolo.mp4'
cap = cv2.VideoCapture(video_path)


width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) + 0.5)
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) + 0.5)
size = (width, height)
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('your_video.avi', fourcc, 20.0, size)

frame_width = int(cap.get(3)) 
frame_height = int(cap.get(4)) 
   
size = (frame_width, frame_height) 
   
# Below VideoWriter object will create 
# a frame of above defined The output  
# is stored in 'filename.avi' file.

result = cv2.VideoWriter('filename.mp4',  
                         cv2.VideoWriter_fourcc(*'MJPG'), 
                         10, size) 


ret = True
# read frames
while ret:
    ret, frame = cap.read()



    if ret:

        # detect objects
        # track objects
        results = model.track(frame, persist=True)

        # plot results
        frame_ = results[0].plot()

        # visualize
        cv2.imshow('Detected kids', frame_)
        if cv2.waitKey(30) & 0xFF == ord('q'):
            break
        
    else:
        break
        
"""
