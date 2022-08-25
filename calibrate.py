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
object_points = [ [j*cell_size, i*cell_size, 0] for i in range(0,chessboard_size[1]) for j in range(0,chessboard_size[0])]

for left_path, right_path in zip(left_paths,right_paths):
    for impath in [left_path, right_path]:
        img = cv2.imread(impath)
        chessbord_corners = []
        retval, corners = cv2.findChessboardCorners(img, chessboard_size, CALIB_CB_FAST_CHECK)
        if retval:
            drawChessboardCorners(img, chessboard_size, cv2.Mat(corners), retval)
            cv2.imshow(impath,img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        else:
            print(f"ERROR:  Did not find a chessboard in the image at path {impath}")
            exit()
