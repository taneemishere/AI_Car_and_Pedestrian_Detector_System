# AI Car and Pedestrian Detector System 
An AI based system that do detects the cars and the pedestrians from a captured video. 
## Requirements
Apart from installing [Python](http://www.python.org/), you'll also need to pip and install the [OpenCV](https://docs.opencv.org/2.4/modules/objdetect/doc/cascade_classification.html?highlight=detectmultiscale), and this is pretty much it.

## The Trainig Data
The trainging datasets are downloaded from the internet you can find the car's dataset also known as [car_detector data set](https://github.com/taneemishere/AI_Car_and_Pedestrian_Detector_System/blob/master/car_detector.xml), and the pedestrian's dataset known as [haarcascade_fullbody](https://github.com/taneemishere/AI_Car_and_Pedestrian_Detector_System/blob/master/haarcascade_fullbody.xml).

## The Code Flow
-	First and foremost import the opencv liberary that's gonna do a lot of work for us in this detection system.
-	Save the captured video file
-	Save the car and pedestrian pre-trained classifiers
-	Create the classifiers for the both by the use of opencv cascade classifiers
-	Now utill the car (video) didn't stops the process will continue
	- And the process is, to read the video frames
	- If it reads successfully convert that frame to the gray-scale or say black and white, because it can be process faster than in the colored form
	- Now detect the cars and pedestrian from the frames, which it gives us the an array of 4D arrays, of each and every single car and pedestrian.
	- Now for every car and pedestrian draw a rectangle over it
	- Show the frames by ```cv2.imshow()``` method
	- And at the last release the VideoCapture object we've created
