import cv2
from cv2 import CALIB_CB_FAST_CHECK
from cv2 import drawChessboardCorners
import numpy as np
import glob

#Find image paths
left_paths = sorted(glob.glob("img/left/*.jpg"))
right_paths = sorted(glob.glob("img/right/*.jpg"))

print(left_paths)
print(right_paths)

#Setup corresponding points for calibration
chessboard_size = (8,6)
cell_size = 60 #mm
termination_criteria = criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 50, 0.01)

image_points_L = []
image_points_R = []
image_sizes = None
object_points_single = [ [j*cell_size, i*cell_size, 0] for i in range(0,chessboard_size[1]) for j in range(0,chessboard_size[0])]
object_points = []
for left_path, right_path in zip(left_paths,right_paths):

        imgL = cv2.imread(left_path)
        imgL_grey = cv2.imread(left_path, cv2.IMREAD_GRAYSCALE)
        
        imgR = cv2.imread(right_path)
        imgR_grey = cv2.imread(right_path, cv2.IMREAD_GRAYSCALE)
        
        if image_sizes == None:
            image_sizes = [imgL_grey.shape, imgR_grey.shape]
        elif not image_sizes == [imgL_grey.shape, imgR_grey.shape]:
            print("Error.  Input images are not the same size!")
            exit()
        
        retvalL, cornersL = cv2.findChessboardCorners(imgL, chessboard_size, CALIB_CB_FAST_CHECK)
        retvalR, cornersR = cv2.findChessboardCorners(imgR, chessboard_size, CALIB_CB_FAST_CHECK)

        if retvalL and retvalR:
            cornersL = cv2.cornerSubPix(imgL_grey, cornersL, (11,11), (-1,-1), termination_criteria)
            cornersR = cv2.cornerSubPix(imgR_grey, cornersR, (11,11), (-1,-1), termination_criteria)
            image_points_L.append(cornersL)
            image_points_R.append(cornersR)
            object_points.append(np.asarray(object_points_single, dtype=np.float32))
        else:
            print(f"ERROR:  Did not find a chessboard in the image at path {left_path} or {right_path}")
        

print(object_points)
print("=================")
print(image_points_L)
input("")
#Calibrate left camera for initial intrinsics
retvalL, M1, D1, rvec1, tvec1 = cv2.calibrateCamera(object_points, image_points_L, image_sizes[0], None, None)
retvalR, M2, D2, rvec2, tvec2 = cv2.calibrateCamera(object_points, image_points_R, image_sizes[1], None, None)

#Test single camera calibration
for left_path, right_path in zip(left_paths,right_paths):
    # undistort 
    leftSrc = cv2.imread(left_path)
    rightSrc = cv2.imread(right_path)
    newcameramtxL, roiL = cv2.getOptimalNewCameraMatrix(M1, D1, image_sizes[0], 1, image_sizes[0])
    newcameramtxR, roiR = cv2.getOptimalNewCameraMatrix(M2, D2, image_sizes[1], 1, image_sizes[1])
    dstL = cv2.undistort(leftSrc, M1, D1, None, newcameramtxL)
    dstR = cv2.undistort(rightSrc, M2, D2, None, newcameramtxR)

    concat = cv2.hconcat([leftSrc,rightSrc])
    cv2.imshow("Calib", concat)
    cv2.waitKey(0)

if not retvalL and retvalR:
    print(f"Error could not calculate an initial camera calibration L={retvalL} R={retvalR}")
    exit()

#Calibrate the stereo Pair
flags = cv2.CALIB_FIX_INTRINSIC  
retval , M1, D1, M2, D2, R, T, E, F = cv2.stereoCalibrate(object_points, image_points_L, image_points_R, M1, D1, M2, D2, image_sizes[0], flags)
print(f"Retval: {retval}")

#Rectification
# flags = cv2.CALIB_ZERO_DISPARITY
R1, R2, P1, P2, Q, ROI1, ROI2 = cv2.stereoRectify(M1,D1,M2,D2, image_sizes[0], R,T, flags)


#Create a map which we can use to recfy images
map1x, map1y = cv2.initUndistortRectifyMap(M1,D1,R,P1,image_sizes[0], cv2.CV_16SC2, cv2.INTER_LINEAR)
map2x, map2y = cv2.initUndistortRectifyMap(M2,D2,R,P2,image_sizes[1], cv2.CV_16SC2, cv2.INTER_LINEAR)

#test
for left_path, right_path in zip(left_paths,right_paths):
    lim = cv2.imread(left_path)
    rim = cv2.imread(right_path)

    lim_rect = cv2.remap(lim,map1x,map1y,cv2.INTER_LINEAR)
    rim_rect = cv2.remap(rim,map2x,map2y,cv2.INTER_LINEAR)

    norma_concat = cv2.hconcat([lim,rim])
    rect_concat = cv2.hconcat([lim_rect,rim_rect])

    #draw lines
    nlines = 50
    for img in [norma_concat,rect_concat]:
        for l in range(0,nlines):
            hoffset = int(l*(img.shape[1]/nlines))
            cv2.line(img,(0,hoffset), (2*img.shape[0],hoffset), (0,0,255), 1)

    cv2.imshow("Normal",norma_concat)
    cv2.imshow("Rectified",rect_concat)
    cv2.waitKey(0)
    cv2.destroyAllWindows()






