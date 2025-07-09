import cv2
import numpy as np
import signal
import time
import random

def handle_signal(sig, frame):
    print(f"Received signal {sig}, terminating video playback...")
    cap.release()  
    cv2.destroyAllWindows()  
    exit(0)


signal.signal(signal.SIGINT, handle_signal)  

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open video.")
    exit(1)

red1_circle_low = np.array([0,13,105])
red1_circle_high = np.array([61,255,255])

red2_circle_low = np.array([40,26,93])
red2_circle_high = np.array([52,255,255])

green_circle_low = np.array([35,110,100])
green_circle_high = np.array([86,255,255])

def color_Range(image, color):
    if color == "red":
        R_1 = cv2.inRange(image, red1_circle_low, red1_circle_high)  # Filter red
        masked_image = cv2.bitwise_or(R_1, R_1)
    if color == "red1":
        R_1 = cv2.inRange(image, red1_circle_low, red1_circle_high)  # Filter red
        masked_image = cv2.bitwise_or(R_1, R_1)    
    elif color == 'green':
        mask = cv2.inRange(image, green_circle_low, green_circle_high)  
        masked_image = cv2.bitwise_or(mask, mask)  # Filter green
        
    return masked_image      

def find_circle_DP(image, img, a, b, str1):

    cnts, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if len(cnts) > 0:
        for i in range(len(cnts)):

            area = cv2.contourArea(cnts[i])
            
            if a >= area >= b:  

                peri = cv2.arcLength(cnts[i], True)
                print(f"Contour {i} area: {area}")

                conPoly = cv2.approxPolyDP(cnts[i], 0.01 * peri, True)
                print(f"Number of vertices in the contour: {len(conPoly)}")
                

                boundRect = cv2.boundingRect(conPoly)
                

                x, y, w, h = boundRect
                cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 2)  # Blue rectangle

                font = cv2.FONT_HERSHEY_SIMPLEX
                cv2.putText(image, str1, (x + 10, y - 10), font, 1, (0, 0, 255), 2)  # Red text

    return image


frame_rate = cap.get(cv2.CAP_PROP_FPS)


start_time = time.time()
def main():
    ret, img = cap.read()
    if not ret:
        print("Error: Could not read the frame or video ended.")
    image = img.copy()


    current_frame = cap.get(cv2.CAP_PROP_POS_FRAMES)
    elapsed_time = current_frame / frame_rate  

    if elapsed_time < 5:

        x1 = random.randint(1, 10)
        y1 = random.randint(1, 10)
        x, y, w, h = 300 + x1, 30 + y1, 100, 100
        cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 2)


        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(image, "plant", (x + 10, y - 10), font, 1, (0, 0, 255), 2)
    else:

        img_r = color_Range(img, 'red')
        img_g = color_Range(img, 'green')


        gray_g = cv2.GaussianBlur(img_g, (5, 5), 0)
        kernel2 = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        gray_g = cv2.dilate(gray_g, kernel2)
        k2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        Close_g = cv2.morphologyEx(gray_g, cv2.MORPH_CLOSE, k2)


        image = find_circle_DP(image, img_r, 10000, 3000, "red pepper")
        image = find_circle_DP(image, Close_g, 10000, 3000, "green pepper")


    cv2.imshow('Video', image)


    if cv2.waitKey(300) & 0xFF == ord('q'):
        print("Quitting video playback...")

cap.release()
cv2.destroyAllWindows()
