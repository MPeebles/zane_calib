import cv2

def load_calibration(path):
    storage = cv2.FileStorage(path, cv2.FileStorage_READ)      
    allMats = ["Q", "M1", "M2", "D1", "D2", "P1", "P2", "R"]    
    return {key:storage.getNode(key).mat() for key in allMats}



def rectify_images(left_image, right_image, c):
    left = cv2.imread(left_image)
    right = cv2.imread(right_image)
    mapxL, mapyL = cv2.initUndistortRectifyMap(c["M1"], c["D1"], c["R"], c["P1"], (left.shape[1], left.shape[0]), cv2.CV_16SC2)
    mapxR, mapyR = cv2.initUndistortRectifyMap(c["M2"], c["D2"], c["R"], c["P2"], (right.shape[1], right.shape[0]), cv2.CV_16SC2)

    left_rect = cv2.remap(left, mapxL, mapyL, cv2.INTER_AREA)
    right_rect = cv2.remap(right, mapxR, mapyR, cv2.INTER_AREA)
    return left_rect, right_rect

def find_disparity(left, right, factor):

    width = int(left.shape[1] * factor )
    height = int(left.shape[0] * factor)
    dsize = (width, height)
    left = cv2.resize(left,dsize)
    right = cv2.resize(right,dsize)

    def do_nothing(int):
        pass

    cv2.namedWindow("d", cv2.WINDOW_NORMAL)
    cv2.resizeWindow('d',1920,1080)

    cv2.createTrackbar("numDisparities", "d", 5, 50, do_nothing)
    cv2.createTrackbar("blockSize", "d", 1, 100, do_nothing)
    cv2.createTrackbar("P1", "d", 1, 500, do_nothing)
    cv2.createTrackbar("P2", "d", 50000, 200000, do_nothing)
    cv2.createTrackbar("minDisparity", "d", 1, 20, do_nothing)
    SGBM = cv2.StereoSGBM.create()

    while True:
        numDisparities = cv2.getTrackbarPos("numDisparities", "d")*16
        blockSize = cv2.getTrackbarPos("blockSize", "d")*2 + 5
        P1 = cv2.getTrackbarPos("P1", "d")
        P2 = cv2.getTrackbarPos("P2", "d")
        minDisparity = cv2.getTrackbarPos("minDisparity", "d")*16

        SGBM.setNumDisparities(numDisparities)
        SGBM.setBlockSize(blockSize)
        SGBM.setP1(P1)
        SGBM.setP2(P2)
        SGBM.setMinDisparity(minDisparity)

        disparity = SGBM.compute(left,right)
        disparity = (disparity/16.0 - minDisparity)/numDisparities

        cv2.imshow("d", disparity)
        if cv2.waitKey(1) == 27:
            break



if __name__=="__main__":
    calib = load_calibration("./calibration/stereoMap.xml")
    left_rect, right_rect = rectify_images("./calibration/0-3.jpg", "./calibration/1-3.jpg", calib)
    find_disparity(left_rect,right_rect, 0.5)