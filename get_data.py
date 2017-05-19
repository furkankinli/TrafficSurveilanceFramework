import numpy as np
import cv2


def read_data(video_name):
    cap = cv2.VideoCapture(video_name)
    element = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    mog2_bgs = cv2.createBackgroundSubtractorMOG2()
    cv2.ocl.setUseOpenCL(False)
    cropped_images = []

    while True:
        if cap.get(cv2.CAP_PROP_POS_FRAMES) == cap.get(cv2.CAP_PROP_FRAME_COUNT):
            # If the number of captured frames is equal to the total number of frames,
            # we stop
            break
        ret, frame = cap.read()

        # each frame has been gaussian blurred with 7 x 7 filter
        gaussian_blurred = cv2.GaussianBlur(frame, (7, 7), 0)
        # Background has been substracted for each frame with MOG2
        foreground_mask = mog2_bgs.apply(gaussian_blurred)
        # Binarized each frame
        _, thresholded = cv2.threshold(foreground_mask, float(70.0), 255, cv2.THRESH_BINARY)
        # Morpholized each frame
        gradient = cv2.morphologyEx(thresholded, cv2.MORPH_OPEN, element)
        # Dilation and Erosion have been implemented for each frame to create proper BLOBs.
        dilated = cv2.dilate(gradient, kernel=np.ones((5, 5), np.uint8), iterations=1)
        eroded = cv2.erode(dilated, kernel=np.ones((5, 5), np.uint8), iterations=1)

        foreground = eroded

        image, contours, hierarchy = cv2.findContours(foreground.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for i, cnt in enumerate(contours):
            area = cv2.contourArea(cnt)
            tmp_size = np.size(frame)
            if not (area < 500 or area > tmp_size / 8):

                x, y, w, h = cv2.boundingRect(cnt)
                cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 255, 0), 2)
                crop_img = frame[y: y + h, x: x + w]
                rszd_img = cv2.resize(crop_img, (40, 40))

                # To save images and manually label images, it is used.
                # cv2.imwrite("C:/Users/Furkan/Desktop/images/images_" + str(counter) + ".jpg", rszd_img)
                # counter += 1
                cropped_images.append(rszd_img)


    # Frequency of the same object in consecutive frames as a same image is approxiamtely 50.
    matrix = np.asarray([np.array(img).flatten() for i, img in enumerate(cropped_images) if i % 50 == 0], "f")
    print("Size: %d" % len(matrix))

    return matrix