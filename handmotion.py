import cv2
import sys
import numpy as np
import math


"""============================================================================
                                     MAIN
============================================================================"""
def main():
    # Open camera for video capture
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

    # Check for any errors opening the camera
    if not cap.isOpened():
        print("Error: Failed to open camera.")
        sys.exit()

    # Attain width and height information of the camera frame
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Reduce the w and h by half
    w2 = w // 2
    h2 = h // 2

    # Reading frames from camera
    ret, frame = cap.read()

    # Checking camera input 
    if not ret:
        print('Frame read failed!')
        sys.exit()

    # Flip horizontally / mirror image
    frame = cv2.flip(frame, 1)  

    # Convert to grayscale
    gray1 = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Resize the frame to half its size 
    # - cv2.INTER_AREA helps enhance img quality when size is reduced 
    gray1 = cv2.resize(gray1, (w2, h2), interpolation=cv2.INTER_AREA) 

    # Initialize variables for brightness and contrast values
    # - must declare outside the while loop
    brightness_add = 0
    alpha_add = 0

    # Calculating optical flow for each and every frame
    while True:

        # Reading frames from camera
        ret, frame = cap.read()

        # Checking camera input
        if not ret:
            print('Frame read failed!')
            break

        # Flip horizontally / mirror image
        frame = cv2.flip(frame, 1)

        # Attain heigth and width information of given frame
        fr_h, fr_w = frame.shape[:2]

        # Convert to grayscale
        gray2 = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Resize the frame to half its size 
        gray2 = cv2.resize(gray2, (w2, h2), interpolation=cv2.INTER_AREA)

        # Calculate optical flow using Farneback method
        # - gray1: previous frame, gray2: current frame
        # - need two frames to calculate the flow and/or motion
        flow = cv2.calcOpticalFlowFarneback(gray1, gray2, None, 
                                            0.5, 3, 15, 3, 5, 1.1, 0)
        vx, vy = flow[..., 0], flow[..., 1]  # motion vector in xy direction
        mag, ang = cv2.cartToPolar(vx, vy)   # convert cartesian to polar

        # Update the frame based on the change in brightness / contrast values
        frame = np.clip(frame + float(brightness_add), 
                        0, 255).astype(np.uint8)

        frame = np.clip((1 + alpha_add) * frame - 128 * alpha_add, 
                        0, 255).astype(np.uint8)

        bright = "Brightness: %.2f" % brightness_add
        alpha  = "Contrast: %.2f" % alpha_add

        cv2.putText(frame, bright, (30, fr_h - 60), cv2.FONT_HERSHEY_SIMPLEX, 
                    1, (150, 0, 255), thickness=2, lineType=cv2.LINE_AA)
                    
        cv2.putText(frame, alpha, (30, fr_h - 30), cv2.FONT_HERSHEY_SIMPLEX, 
                    1, (150, 0, 255), thickness=2, lineType=cv2.LINE_AA)

        # Visualize motion vectors in HSV colorspace 
        hsv = np.zeros((h2, w2, 3), dtype=np.uint8)
        hsv[..., 0] = ang * 180 / np.pi /  2
        hsv[..., 1] = 255
        hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
        bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        cv2.imshow("Optical Flow", bgr)
        
        # Create mask to determine motions of interest
        # - mag > 2.0 is white; mag <= 2.0 is black
        # - 'significant motion' changes 2 pixels, in this case
        # - this helps eliminate any minor motions caused by noise
        motion_mask = np.zeros((h2, w2), dtype=np.uint8)
        motion_mask[mag > 2.0] = 255  

        # Mean of vx and vy in white areas 
        mx = cv2.mean(vx, mask=motion_mask)[0]
        my = cv2.mean(vy, mask=motion_mask)[0]
        m_mag = math.sqrt(mx * mx + my * my)

        # If the motion is large / significant enough
        if m_mag > 4.0:

            # rom -pi to +pi --> from -180 to +180
            m_ang = math.atan2(my, mx) * 180 / math.pi

            # from -180 to +180 --> from 0 to 360
            m_ang += 180                                

            # origin point for guiding directions
            x1 = fr_w - 100
            y1 = fr_h - 70

            if m_ang >= 45 and m_ang < 135:      # UP
                x2 = x1
                y2 = y1 - 50
                if alpha_add < 5.00:
                    alpha_add += 0.20

            elif m_ang >= 135 and m_ang < 225:   # RIGHT
                x2 = x1 + 50
                y2 = y1
                if brightness_add < 200:
                    brightness_add += 10

            elif m_ang >= 225 and m_ang < 315:   # DOWN
                x2 = x1
                y2 = y1 + 50
                if alpha_add > -1.00:
                    alpha_add -= 0.20

            else:                                # LEFT 
                x2 = x1 - 50
                y2 = y1
                if brightness_add > -200:
                    brightness_add -= 10

            # Draw the direction arrow 
            cv2.arrowedLine(frame, (x1, y1), (x2, y2), (150, 0, 255),
                            5, cv2.LINE_AA, tipLength=0.7)

        # Display the result frame
        cv2.imshow("Handmotion Control", frame)
        cv2.imshow("Motion Mask", motion_mask)

        key = cv2.waitKey(1)

        if key == 27:  # ESC
            break

        if key == 32:  # Space
            brightness_add = 0
            alpha_add = 0

        # Current frame now becomes previous frame
        # Goes back to top of the loop to get current frame
        gray1 = gray2

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()