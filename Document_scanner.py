#Document Scanner BY Adarsh Malviya
import cv2
import numpy as np


def birdpoint(old_pt):
    olt_pt = old_pt.reshape((4,2))
    add = old_pt.sum(2)
    new_pt = np.zeros((4,1,2),np.int32)

    new_pt[0] = old_pt[np.argmin(add)]
    
    new_pt[3] = old_pt[np.argmax(add)]
    diff = np.diff(old_pt, axis = 2)

    new_pt[1] = old_pt[np.argmin(diff)]
    new_pt[2] = old_pt[np.argmax(diff)] 

    return new_pt

def imgwrap(img, biggest):
    biggest = birdpoint(biggest)
    pt1 = np.float32(biggest)
    pt2 = np.float32([[0,0],[img.shape[1],0], [0,img.shape[0]], [img.shape[1], img.shape[0]]])
    matrix = cv2.getPerspectiveTransform(pt1, pt2)
    wrapimg = cv2.warpPerspective(img, matrix, (img.shape[1], img.shape[0]))

    return wrapimg

if __name__ == "__main__":
    frame = cv2.imread("D:\dataset.jpg")
    frame = cv2.resize(frame, (730,737))
    gray_vid = cv2.cvtColor(frame, cv2.IMREAD_GRAYSCALE)
    frame_copy = frame.copy()
    cv2.imshow('Input Image',frame_copy)
    imgblur = cv2.GaussianBlur(frame, (7,7),1)
    edged_frame = cv2.Canny(imgblur,100,200)

    contours , _ = cv2.findContours(edged_frame, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    maxarea = 0
    for contour in contours:
        area = cv2.contourArea(contour)
        peri = cv2.arcLength(contour,True)
        approx = cv2.approxPolyDP(contour, 0.02*peri, True)
        if len(approx) == 4 and area > maxarea:
            maxarea = area
            biggest = approx
    cv2.drawContours(frame, biggest, -1, (0,0,255,3), 20)
    cv2.waitKey(3000)
    cv2.imshow("Document Detected ", frame)

    wrapimg = imgwrap(frame, biggest)
    
    cv2.waitKey(3000)
    cv2.imshow('Output Image',wrapimg)
    
    frame_copy = cv2.resize(frame_copy, (430,437))
    frame = cv2.resize(frame, (430,437))
    wrapimg = cv2.resize(wrapimg, (430,437))
    cv2.waitKey(2000)
    imghor = np.hstack((frame_copy, frame, wrapimg))
    cv2.imshow("All Procedure : ", imghor)
    cv2.waitKey(0)
    cv2.destroyAllWindows()