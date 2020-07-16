import cv2
import sys
import os
import numpy as np
from collections import deque
import pandas as pd

(major_ver, minor_ver, subminor_ver) = (cv2.__version__).split('.')
print(cv2.__version__)
url = 'https://www.youtube.com/watch?v=I5DJ-05ZUKo'


if __name__ == '__main__':

    # Set up tracker.
    # Instead of MIL, you can also use

    # net = cv2.dnn.readNet('goturn.prototxt', 'goturn.caffemodel')
    # inp0 = numpy.random.standard_normal([1, 3, 227, 227]).astype(numpy.float32)
    # inp1 = numpy.random.standard_normal([1, 3, 227, 227]).astype(numpy.float32)
    # net.setInput(inp0, "data1")
    # net.setInput(inp1, "data2")
    # net.forward()
    # tracker = cv2.TrackerGOTURN_create()
    #tracker = cv2.TrackerCSRT_create()
    #tracker = cv2.TrackerTLD_create()
   # tracker = cv2.TrackerMedianFlow_create()
   # tracker = cv2.TrackerMIL_create()
    tracker = cv2.TrackerMOSSE_create()
    pts = deque(maxlen=32)
    direction_change_array = []
    current_direction = ""
    (dX, dY) = (0, 0)
    direction = ""
    counter = 0
    # Read video
    video = cv2.VideoCapture('c2.mp4')

    # Exit if video not opened.
    if not video.isOpened():
        print("Could not open video")
        sys.exit()

    # Read first frame.
    ok, frame = video.read()
    if not ok:
        print('Cannot read video file')
        sys.exit()

    # Define an initial bounding box


    # Uncomment the line below to select a different bounding box
    bbox = cv2.selectROI(frame, False)

    # Initialize tracker with first frame and bounding box
    ok = tracker.init(frame, bbox)

    while True:
        # Read a new frame
        ok, frame = video.read()
        if not ok:
            break

        # Start timer
        timer = cv2.getTickCount()

        # Update tracker
        ok, bbox = tracker.update(frame)

        # Calculate Frames per second (FPS)
        fps = cv2.getTickFrequency() / (cv2.getTickCount() - timer)

        # Draw bounding box
        if ok:
            # Tracking success
            p1 = (int(bbox[0]), int(bbox[1]))
            p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
            cv2.rectangle(frame, p1, p2, (255,0,0), 2, 1)
            pts.appendleft((bbox[0], bbox[3]))

            for i in np.arange(1, len(pts)):
                # if either of the tracked points are None, ignore
                # them
                print('here')
                if pts[i - 1] is None or pts[i] is None:
                    continue

                if counter >= 10 and i == 1 and pts[-10] is not None:
                    # compute the difference between the x and y
                    # coordinates and re-initialize the direction
                    # text variables
                    dX = pts[-10][0] - pts[i][0]
                    dY = pts[-10][1] - pts[i][1]
                    print(dX, dY)
                    (dirX, dirY) = ("", "")

                    # ensure there is significant movement in the
                    # x-direction
                    if np.abs(dX) > 20:
                        dirX = "East" if np.sign(dX) == 1 else "West"

                    # ensure there is significant movement in the
                    # y-direction
                    if np.abs(dY) > 20:
                        dirY = "North" if np.sign(dY) == 1 else "South"

                    # handle when both directions are non-empty
                    if dirX != "" and dirY != "":
                        direction = "{}-{}".format(dirY, dirX)

                    # otherwise, only one direction is non-empty
                    else:
                        direction = dirX if dirX != "" else dirY

                    if direction not in [current_direction]:
                        current_direction = direction
                        direction_change_array.append({
                            "direction": current_direction,
                            "timestamp": (video.get(cv2.CAP_PROP_POS_MSEC) / 1000)
                        })

            cv2.putText(frame, direction, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                0.65, (0, 0, 255), 3)
            cv2.putText(frame, "dx: {}, dy: {}".format(dX, dY),
                (10, frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX,
                0.35, (0, 0, 255), 1)
        else :
            # Tracking failure
            cv2.putText(frame, "Tracking failure detected", (100,80), cv2.FONT_HERSHEY_SIMPLEX, 0.75,(0,0,255),2)

        # Display tracker type on frame
      #  cv2.putText(frame, tracker_type + " Tracker", (100,20), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50,170,50),2);

        # Display FPS on frame
        cv2.putText(frame, "FPS : " + str(int(fps)), (100,50), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50,170,50), 2);

        # Display result
        cv2.imshow("Tracking", frame)

        # Exit if ESC pressed
        k = cv2.waitKey(1) & 0xff
        if k == 27 : break
        counter += 1

    df = pd.DataFrame(direction_change_array)
    df = df[df.direction != ""]
    f1 = open('output.json', 'w')
    f1.write(df.to_json(orient='records'))
    f1.close()

    video.release()

# close all windows
cv2.destroyAllWindows()