import cv2

# our video file
video = cv2.VideoCapture('Cars and Pedestrians_Trim.mp4')

# our pre-trained car and  pedestrian classifiers
car_tracker_file = 'car_detector.xml'
pedestrian_tracker_file = 'haarcascade_fullbody.xml'

# create car and pedestrian classifiers
car_tacker = cv2.CascadeClassifier(car_tracker_file)
pedestrian_tacker = cv2.CascadeClassifier(pedestrian_tracker_file)

# run until the car stops
while True:
    # read the current frame
    (read_successful, frame) = video.read()

    if read_successful:
        # if read successfully converted to grayscale
        grayscaled_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    else:
        break

    # detect cars AND pedestrians
    cars = car_tacker.detectMultiScale(grayscaled_frame)
    pedestrians = pedestrian_tacker.detectMultiScale(grayscaled_frame)
    # print(cars)

    # draw rectangle around the cars in red color
    for (x, y, w, h) in cars:
        cv2.rectangle(frame, (x + 1, y + 2), (x + w, y + h), (255, 0, 0), 2)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)

    # draw rectangle around the pedestrians in yellow color
    for (x, y, w, h) in pedestrians:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 255), 2)

    # display the image (frame) with the faces spotted
    cv2.imshow("AI Cars and Pedestrians Detector", frame)

    # don't auto close, wait for the any key to press
    key = cv2.waitKey(2)

    # if that key = q or Q, so then close the video
    if key == 81 or key == 113:
        break

# Release the VideoCapture object
video.release()

print("Done!")
