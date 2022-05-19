import cv2 # opencv library
import time
import numpy as np
from os.path import isfile, join
import matplotlib.pyplot as plt
import urllib.request
from VideoGet import VideoGet
from threading import Thread
from multiprocessing import Process
from queue import Queue

cap = cv2.VideoCapture()
#Vehicle count for North, East, South, West junctions
vehicle_count = [0,0,0,0]
consecutive_frame = False
consecutive_x = 0

# List for divided images from previous frame (0:3)
prev_images = []

# kernel for image dilation
kernel = np.ones((4,4),np.uint8)
count_i = 0

# font style
font = cv2.FONT_HERSHEY_SIMPLEX

def threadVideoGet(source, q):
    """
    Dedicated thread for grabbing video frames with VideoGet object.
    Main thread shows video frames.
    """
    global count_i
    video_getter = VideoGet(source).start()
    initial = True

    while True:

        processed_frames = []
        if (cv2.waitKey(1) == ord("q")) or video_getter.stopped:
            video_getter.stop()
            print("Video getter stopped either by user or an error")
            break

        frame = video_getter.frame
        height, width, ch = frame.shape
        roi_height = int(height / 2)
        roi_width = int(width / 2)
        
        images = []

        for x in range(0, 2):
            for y in range(0, 2):
                tmp_image=frame[x*roi_height:(x+1)*roi_height, y*roi_width:(y+1)*roi_width]
                images.append(tmp_image)
        
        if (initial == False):
            for i in range(len(images)):
                processed_frames.append(traffic_detection(images[i], prev_images[i], i))
            
            for x in range(0, 2):
                for y in range(0, 2):
                    cv2.imshow(str(x*2+y+1), processed_frames[x*2+y])
                    cv2.moveWindow(str(x*2+y+1), 100+(y*roi_width), 50+(x*roi_height))
            #cv2.imshow("Video", frame)
        prev_images = images
        initial = False

def traffic_detection(frame, lastFrame, camera_index):
    global vehicle_count, consecutive_frame, consecutive_x
    cntr_found = False

    # frame differencing
    grayA = cv2.cvtColor(lastFrame, cv2.COLOR_BGR2GRAY)
    grayB = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    diff_image = cv2.absdiff(grayB, grayA)
    
    # image thresholding
    ret, thresh = cv2.threshold(diff_image, 30, 255, cv2.THRESH_BINARY)
    
    # image dilation
    dilated = cv2.dilate(thresh,kernel,iterations = 1)
    
    # find contours
    contours, hierarchy = cv2.findContours(dilated.copy(), cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)
    
    # shortlist contours appearing in the detection zone
    valid_cntrs = []
    for cntr in contours:
        x,y,w,h = cv2.boundingRect(cntr)
        if (x <= 200) & (y >= 84) & (y <= 85) & (cv2.contourArea(cntr) >= 25):
            if (y >= 100) & (cv2.contourArea(cntr) < 40):
                break
            if (consecutive_frame == True) & (cv2.contourArea(cntr) >= consecutive_x*0.95) & (cv2.contourArea(cntr) <= consecutive_x*1.05):
                consecutive_frame = False
                consecutive_x = 0
                break
            valid_cntrs.append(cntr)
            vehicle_count[camera_index] += 1
            cntr_found = True
            consecutive_x = x
            print("Vehicle Registered" + str(vehicle_count[camera_index]))
            q.put(vehicle_count)

    if cntr_found == True:
        consecutive_frame = True
    else:
        consecutive_frame = False
    # add contours to original frames
    img = frame.copy()
    cv2.drawContours(img, valid_cntrs, -1, (127,200,0), 2)
    cv2.putText(img, "vehicles: " + str(vehicle_count[camera_index]) , (55, 15), font, 0.6, (0, 180, 0), 2)
    cv2.line(img, (0, 90),(360,90),(100, 255, 255))
    return(img)
    #cv2.imwrite(pathIn+str(i)+'.png',img)  

def thingspeak_write(name, q):
    vehicle_count = q.get()
    baseURL = 'http://api.thingspeak.com/update?api_key=W&'
    while True:
        b=urllib.request.urlopen(baseURL + "field1="+str(vehicle_count[0])+"&field2="+str(vehicle_count[1])+"&field3="+str(vehicle_count[2])+"&field4="+str(vehicle_count[3]))      
        print(vehicle_count)

if __name__ == "__main__":
    q = Queue()
    t1 = Thread(target = thingspeak_write, args=("Thread-1", q))
    t2 = Thread(target = threadVideoGet, args=(0, q))
    t1.start()
    t2.start()
    t1.join()
    t2.join()
    cap.release()
    cv2.destroyAllWindows()

"""
Modification notes to keep a proper count on traffic:
- Make the contour zone (y-axis) small to only allow for the vehicle to be detected once.
- Adding more validation, like similar contour, or only incrementing once and freezing the count until the x axis position has no contour present.
- Tracking each lane if there are multiple (Single Lane Tracking is a lot more accurate).
- The monitor could process the frame contours of each vehicle lane seperately
"""
