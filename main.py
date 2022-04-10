import cv2 # opencv library
import numpy as np
from os.path import isfile, join
import matplotlib.pyplot as plt
from VideoGet import VideoGet
from threading import Thread

cap = cv2.VideoCapture()
vehicle_count = 0
consecutive_frame = False
consecutive_x = 0

# List for divided images from previous frame (0:3)
prev_images = []

# kernel for image dilation
kernel = np.ones((4,4),np.uint8)

# font style
font = cv2.FONT_HERSHEY_SIMPLEX

def threadVideoGet(source):
    """
    Dedicated thread for grabbing video frames with VideoGet object.
    Main thread shows video frames.
    """
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
        
        print("test")
        if (initial == False):
            for i in range(len(images)):
                print("test")
                processed_frames.append(traffic_detection(images[i], prev_images[i]))

        prev_images = images
        
        for x in range(0, 2):
            for y in range(0, 2):
                cv2.imshow(str(x*2+y+1), processed_frames[x*2+y])
                cv2.moveWindow(str(x*2+y+1), 100+(y*roi_width), 50+(x*roi_height))
        #cv2.imshow("Video", frame)
        initial = False

def traffic_detection(frame, lastFrame):

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
            print(cv2.contourArea(cntr))
            valid_cntrs.append(cntr)
            vehicle_count += 1
            cntr_found = True
            consecutive_x = x

    if cntr_found == True:
        consecutive_frame = True
    else:
        consecutive_frame = False
    print("next frame" , cntr_found)
    # add contours to original frames
    img = frame.copy()
    cv2.drawContours(img, valid_cntrs, -1, (127,200,0), 2)
    
    cv2.putText(img, "vehicles: " + str(vehicle_count), (55, 15), font, 0.6, (0, 180, 0), 2)
    cv2.line(img, (0, 90),(256,90),(100, 255, 255))
    return(img)
    #cv2.imwrite(pathIn+str(i)+'.png',img)  

def thingspeak_write():
    pass

def traffic_monitor():
    #Begin image detection
    threadVideoGet(0)
    #Thread(target=threadVideoGet(0), args=()).start()

    #Create another thread with timed loop for uploading data to Thingspeak


if __name__ == "__main__":
    traffic_monitor()
    cap.release()
    cv2.destroyAllWindows()

"""
Modification notes to keep a proper count on traffic:
- Make the contour zone (y-axis) small to only allow for the vehicle to be detected once.
- Adding more validation, like similar contour, or only incrementing once and freezing the count until the x axis position has no contour present.
- Tracking each lane if there are multiple (Single Lane Tracking is a lot more accurate).
- The monitor could process the frame contours of each vehicle lane seperately
"""