import cv2
from ultralytics import YOLO

# Load the YOLOv8 model
model = YOLO('YOLOs100normal.pt')

#s100 normal 80
#s100+50bt 75
#s150 best 60
#s50  best 70
#n300 best 70
#s50+100bt 65
#s50+200bt 70
#s50+special20 85 +-5



# Open the video file
video_path = "Mission2.mp4"
cap = cv2.VideoCapture(video_path)

fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

output = cv2.VideoWriter('predict1.mp4', cv2.VideoWriter_fourcc(*'mp4v'), fps*2, (width, height))

# Loop through the video frames
while cap.isOpened():
    # Read a frame from the video
    success, frame = cap.read()

    if success:
        # Run YOLOv8 inference on the frame
        results = model(frame, stream=True,imgsz=640,device="cuda:0")  # generator of Results objects

        # Visualize the results on the frame
        annotated_frame = next(results).plot()
        output.write(annotated_frame)
        # Display the annotated frame
        cv2.imshow("YOLOv8 Inference", annotated_frame)
        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    else:
        # Break the loop if the end of the video is reached
        break

# Release the video capture object and close the display window
cap.release()
output.release()
cv2.destroyAllWindows()