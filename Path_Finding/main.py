import cv2
import numpy as np
import math
import copy
from imutils.perspective import four_point_transform
from imutils import contours
import imutils
from app import *

ENABLE_DEBUG = 1
ENABLE_GRID  = 1

def showImage(caption, image):
    if ENABLE_DEBUG:
        cv2.namedWindow(caption, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(caption, 300,300)
        cv2.imshow(caption, image)

debug_window_dict = {}
def debugWindowAppend(caption, image):
    if ENABLE_DEBUG:
        if len(image.shape) == 2:
            debug_window_dict[caption] = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        else:
            debug_window_dict[caption] = image

def random_color():
    rgbl=[255,30,200,40]
    np.random.shuffle(rgbl)
    return tuple(rgbl[0:3])

def apply_overlay(output, overlay, alpha):
    cv2.addWeighted(overlay, alpha, output, 1 - alpha, 0, output)

def debugWindowShow(title='Debug_Window', scale = 0.2):
    if ENABLE_DEBUG:
        size = len(debug_window_dict)
        width = int(round(np.sqrt(size)))
        im_list_2d = []
        temp = []
        img_shape = []
        index = 0
        for img_name in debug_window_dict:
            if index == 0:
                img_shape = debug_window_dict[img_name].shape
            else:
                if img_shape < debug_window_dict[img_name].shape:
                    img_shape = debug_window_dict[img_name].shape
            index += 1

        index = 0
        for img_name in debug_window_dict:
            dummy_img = np.full(img_shape, 125, np.uint8)
            if index%width == 0 and index!=0:
                im_list_2d.append(temp)
                temp = []
            last_image = debug_window_dict[img_name]
            centralPIP(dummy_img,last_image)
            cv2.putText(dummy_img, img_name, (20, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 3, (125,125,255), 5)
            temp.append(dummy_img)
            index = index + 1
        # img1.paste(img, box=(x1, y1, x2, y2), mask=img)
        tot_dummies = width - len(temp)
        dummy_img = np.full(img_shape, 125, np.uint8)
        for i in range(0, tot_dummies):
            temp.append(dummy_img)
        im_list_2d.append(temp)
        if (size < 20):
            cv2.imshow(title, concat_tile(im_list_2d, scale))

def centralPIP(bkg_img, frg_img, autoFit=True):
    bw, bh, bc = bkg_img.shape
    fw, fh, fc = frg_img.shape
    scale_factor = 1
    front = frg_img
    if autoFit:
        rw = bw/fw
        rh = bh/fh
        r = min(rw, rh)
        front = cv2.resize(frg_img, dsize=(int(fh*r), int(fw*r)), interpolation=cv2.INTER_CUBIC)
        fw, fh, fc = front.shape
    x = int(bw/2) - int(fw/2)
    y = int(bh/2) - int(fh/2)
    bkg_img[x:x+fw, y:y+fh] = front

def imageScale(img, factor):
    return cv2.resize(img, (0,0), fx=factor, fy=factor)

def concat_tile(im_list_2d, scale):
    im_list_v = []
    for im_list_h in im_list_2d:
        newim_list_h = []
        for im_ in im_list_h:
            newim_list_h.append(imageScale(im_, scale))
        im_list_v.append(cv2.hconcat(newim_list_h))
    return cv2.vconcat(im_list_v)

def detectMark(frame, list_of_bounds, CV2_VERSION, scale = 0.1):
    frame_cpy = copy.deepcopy(frame)
    frame_hsv = cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)
    frame_hsv_gauss = cv2.GaussianBlur(frame_hsv,(5,5),cv2.BORDER_DEFAULT)
    # define range of color in HSV
    list_of_masks = []
    for bnd in list_of_bounds:
        list_of_masks.append({'tag':bnd['tag'],
        'mask': cv2.inRange(frame_hsv_gauss, np.array(bnd['lower']), np.array(bnd['upper'])),
        'minArea': bnd['minArea']})

    coords = {}

    feature_mask = np.full(frame.shape[0:2], 255, np.uint8)
    for mask in list_of_masks:
        # extract the maze image only
        if CV2_VERSION == '3.4.3':
            _, contours, hierarchy = cv2.findContours(mask['mask'], cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        else:
            contours, hierarchy = cv2.findContours(mask['mask'], cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = sorted(contours, key=cv2.contourArea, reverse=True)
        if len(cnts) > 0:
            # compute the center of the contour
            M = cv2.moments(cnts[0])
            area = cv2.contourArea(cnts[0])
            if area > mask['minArea']: 
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])
                clr = random_color()
                # cv2.drawContours(frame_cpy, cnts[0], -1, (0, 255, 0), 2)
                cv2.circle(frame_cpy, (cX, cY), 10, clr, -1)
                if mask['tag'] != 'start':
                    padding = 0
                    side = int(np.sqrt(area)/2+padding)
                    cv2.circle(feature_mask, (cX, cY), side, 0, -1)
                cv2.circle(frame_cpy, (cX, cY), int(np.sqrt(area)/2), clr, 8) #area as the circle size for confidence measure
                cv2.putText(frame_cpy, mask['tag'], (cX - 30, cY - 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, clr, 3)
            else:
                cX = -1
                cY = -1
                area = -1
            coords[mask['tag']] = [cX, cY, area]

    merged_mask = list_of_masks[0]['mask']
    for mask in list_of_masks:
        merged_mask = merged_mask | mask['mask']
    res = cv2.bitwise_and(frame, frame, mask= merged_mask)

    debugWindowAppend('highlighted', frame_cpy)
    debugWindowAppend('mask', merged_mask)
    debugWindowAppend('result', res)
    debugWindowAppend('feature_mask', feature_mask)
    return coords, feature_mask

def extractMaze(frame, cv2_version):
    frame_cpy = copy.deepcopy(frame)
    # Gray image op
    frame_gray = cv2.cvtColor(frame,cv2.COLOR_RGB2GRAY)
    frame_gray_gauss = cv2.GaussianBlur(frame_gray,(5,5),cv2.BORDER_DEFAULT)
    # Inverting tholdolding will give us a binary image with a white wall and a black background.
    ret, thresh = cv2.threshold(frame_gray_gauss, 75, 255, cv2.THRESH_BINARY_INV)

     # extract the maze image only
    if cv2_version == '3.4.3':
        _, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    else:
        contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = sorted(contours, key=cv2.contourArea, reverse=True)
    cv2.drawContours(frame, contours, -1, (0,255,0), 3)
    showImage("contour", frame)
    displayCnt = None

    for c in cnts:
    	# approximate the contour
    	peri = cv2.arcLength(c, True)
    	approx = cv2.approxPolyDP(c, 0.02 * peri, True)

    	if len(approx) == 4:
            displayCnt = approx
            cv2.drawContours(frame, displayCnt, -1, (0,255,255), 3)
            debugWindowAppend('approx', frame)
            break

    maze_extracted = four_point_transform(frame, displayCnt.reshape(4, 2))
    debugWindowAppend('maze_extracted', maze_extracted)
    return maze_extracted

def grid_on(frame_grid, pixel_step_size):
    #assign grid to maze
    maze_pixel_height = frame_grid.shape[0]
    maze_pixel_width  = frame_grid.shape[1]
    #this is just for debug purpose -> not necessary to show in operation
    # horizontal line
    for i in range(0, maze_pixel_height, pixel_step_size):
        frame_grid = cv2.line(frame_grid,(0,i),(maze_pixel_width, i),(169,169,169),1)

    #vertical line
    for j in range(0, maze_pixel_width, pixel_step_size):
        frame_grid = cv2.line(frame_grid,(j,0),(j, maze_pixel_height),(169,169,169),1)

    debugWindowAppend('grid', frame_grid)

def mapMaze_Array(frame, feature_coord, feature_mask, grid_size):
    temp = frame.copy()
    maze_gray = cv2.cvtColor(frame,cv2.COLOR_RGB2GRAY)
    maze_gray_gauss = cv2.GaussianBlur(maze_gray,(5,5),cv2.BORDER_DEFAULT)
    # Inverting tholdolding will give us a binary image with a white wall and a black background.
    debugWindowAppend('gray', maze_gray_gauss)

    ret, thresh_maze = cv2.threshold(maze_gray_gauss, 65, 255, cv2.THRESH_BINARY_INV)
    debugWindowAppend('maze1', thresh_maze)
    
    frame_hsv = cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)
    frame_hsv_gauss = cv2.GaussianBlur(frame_hsv,(5,5),cv2.BORDER_DEFAULT)
    # define range of color in HSV
    thresh_maze = cv2.inRange(frame_hsv_gauss, np.array([0,0,0]), np.array([255,67,115]))
    # maze_hsv = cv2.inRange(frame_hsv_gauss, np.array([4,0,79]), np.array([255,255,255]))

    thresh_maze = thresh_maze&feature_mask
    debugWindowAppend('maze2 in use', thresh_maze)
    # Kernel
    ke = 30
    kernel = np.ones((ke, ke), np.uint8)
    # Dilation
    dilation_maze = cv2.dilate(thresh_maze, kernel, iterations=1)

    ke = 25
    kernel = np.ones((ke, ke), np.uint8)
    # Erosion
    filtered_maze = cv2.erode(dilation_maze, kernel, iterations=1)

    debugWindowAppend('filtered', filtered_maze)

    if ENABLE_GRID:
        grid_image = filtered_maze
        grid_on(grid_image, grid_size)

    filter_maze_pixel_height = filtered_maze.shape[0]
    filter_maze_pixel_width  = filtered_maze.shape[1]

    start_coord = feature_coord['start'][0:2]
    end_coord   = feature_coord['end'][0:2]
    ball_coord  = feature_coord['ball'][0:2]
    map_array = []
    start_array = [ (math.floor( start_coord[0] /grid_size)) , (math.floor(start_coord[1] /grid_size)) ]
    end_array = [(math.floor( end_coord[0] /grid_size)) , (math.floor(end_coord[1] /grid_size))]
    ball_array = [(math.floor( ball_coord[0] /grid_size)) , (math.floor(ball_coord[1] /grid_size))]


    for j in range(0, filter_maze_pixel_width, grid_size):
        map_array_1D = []
        for i in range(0, filter_maze_pixel_height, grid_size):

            roi = filtered_maze[j: j + grid_size , i: i + grid_size]
            if ( np.sum(roi) > (255 * grid_size * grid_size / 8) ):
            #filtered_maze[grid_size, grid_size] = (0, 255,0)
                cv2.rectangle(temp,(i,j),(i+grid_size, j+grid_size),(0,125,255),-1)
                # obstacle
                map_array_1D.append(0)
            else:
                # path
                map_array_1D.append(1)
        map_array.append(map_array_1D)
    
    map_array[start_array[1]][start_array[0]] = 1
    map_array[end_array[1]][end_array[0]] = 1
    map_array[ball_array[1]][ball_array[0]] = 1
    #add start and end color fill
    start_x_pixel = start_array[0] * grid_size
    start_y_pixel = start_array[1] * grid_size
    end_x_pixel = end_array[0] * grid_size
    end_y_pixel = end_array[1] * grid_size
    ball_pixel = [ball_array[0] * grid_size, ball_array[1] * grid_size]

    apply_overlay(frame, temp, 0.5)
    cv2.rectangle(frame,(start_x_pixel , start_y_pixel),( start_x_pixel + grid_size, start_y_pixel + grid_size),(0,0,255),-1)
    cv2.rectangle(frame,(end_x_pixel , end_y_pixel),( end_x_pixel + grid_size, end_y_pixel + grid_size),(255,0,0),-1)
    cv2.rectangle(frame,(ball_pixel[0], ball_pixel[1]),( ball_pixel[0] + grid_size, ball_pixel[1] + grid_size),(255,255,255),-1)

    # debugWindowAppend('obstacle', frame)

    return map_array, start_array, end_array, ball_array

def paintPath(maze_frame, path, grid_size, color=(0,255,0)):
    temp = maze_frame.copy()
    for coord in path:
        x = coord[0] * grid_size
        y = coord[1] * grid_size
        cv2.rectangle(temp,(x , y),( x + grid_size, y + grid_size),color,-1)
    apply_overlay(maze_frame, temp, 0.4)
    return maze_frame
  
def mazeSolver_Phase1(frame, cv2_version, grid_size_percent):
    #maze extraction from captured image
    maze_frame = extractMaze(frame, cv2_version)
    maze_dim = maze_frame.shape
    grid_size = int(max(maze_dim)*grid_size_percent)
    print('maze_dimension: ', maze_dim, ' - grid size', grid_size)

    # marker detection
    list_of_bounds =[   {'tag': 'end', 'lower':[30,46,6], 'upper':[69,224,137], 'minArea':1000},
                        {'tag': 'start','lower':[0,106,0],  'upper':[15, 255, 255], 'minArea':1000},
                        {'tag': 'ball', 'lower':[0,0,215], 'upper':[255,255,255], 'minArea':1000},
                    ]
    feature_coord, feature_mask = detectMark(maze_frame, list_of_bounds, cv2_version)
    #print(coord)

    #create 2D binarized array of maze (1-> path, 0->obstacle)
    All_tags_exist = True
    for feature in list_of_bounds:
        if feature['tag'] not in feature_coord:
            All_tags_exist = False
            print('[ERR] UNABLE to find feature::', feature['tag'])
        else:
            if feature_coord[feature['tag']][2] == -1:
                All_tags_exist = False
                print('[ERR] Invalid feature::', feature['tag'], feature_coord[feature['tag']])
    maze = []
    start = []
    end = []
    ball = []
    if All_tags_exist:
        maze, start, end, ball = mapMaze_Array(maze_frame, feature_coord, feature_mask, grid_size)
    return maze, start, end, ball, maze_frame, grid_size

def detectBall(frame_out, frame_gray):
    # detect circles in the image
    circles = cv2.HoughCircles(frame_gray, cv2.HOUGH_GRADIENT, 1.2, 100)
    # ensure at least some circles were found
    if circles is not None:
        # convert the (x, y) coordinates and radius of the circles to integers
        circles = np.round(circles[0, :]).astype("int")

        # loop over the (x, y) coordinates and radius of the circles
        for (x, y, r) in circles:
            # draw the circle in the output image, then draw a rectangle
            # corresponding to the center of the circle
            cv2.circle(frame_out, (x, y), r, (0, 255, 0), 4)
            cv2.rectangle(frame_out, (x - 5, y - 5), (x + 5, y + 5), (0, 128, 255), -1)

        # show the output image
        # cv2.imshow("output", imageScale(np.hstack([frame, output]),scale))

def init_webCam():
    cam = cv2.VideoCapture(0)
    if not cam.isOpened():
        raise IOError("Cannot open webcam")
    return cam

def grab_webCam_feed(cam, mirror=False):
    ret_val, img = cam.read()
    if mirror:
        img = cv2.flip(img, 1)
    return img

def nothing(x):
    pass

def showUtilities(properties):
    # Create a black image, a window
    img = np.zeros((300,512,3), np.uint8)
    cv2.namedWindow('calibrate_window')
    # create trackbars for color change
    for prop in properties:
        cv2.createTrackbar(prop,'calibrate_window',0,255,nothing)
    return 'calibrate_window'

def obtainSlides(properties):
    vals = []
    for prop in properties:
        vals.append(cv2.getTrackbarPos(prop,'calibrate_window'))
    return vals

def main():
    CV2_VERSION = cv2.__version__
    print('CV2 VERSION:' ,CV2_VERSION)
    ## MODE SELECTION ##
    MODE = "CALIBRATION_HSV"
    # MODE = "TESTING_RUN"
    # MODE = "TESTING_STATIC"
    RUNONCE = True
    GRID_SIZE_PERCENT = 0.06
    ##### FOR TESTING RUN_TIME ######
    if "TESTING_RUN" == MODE or "TESTING_STATIC" == MODE:
        cam = []
        if "TESTING_RUN" == MODE:
            cam = init_webCam()
            frame = grab_webCam_feed(cam, mirror=True)
            cv2.imwrite('chicken.png', frame)
            # mirror the image
            frame = cv2.flip(frame, 0)
        while True:
            if "TESTING_STATIC" == MODE:
                frame = cv2.imread('chicken.png')
            # extract maze bndry
            maze, start, end, ball, maze_frame, grid_size = mazeSolver_Phase1(frame, CV2_VERSION, GRID_SIZE_PERCENT)
            if len(maze) == 0:
                print('[ERR] UNABLE to recognize Maze')
                debugWindowShow()
            else:
                path = find_path(maze, start, end)
                path_realTime = find_path(maze, ball, end)
                if len(path) == 0:
                    print('[ERR] No Path Found')
                    debugWindowShow()
                else:
                    temp = maze_frame.copy()
                    temp2 = maze_frame.copy()
                    path_frame1 = paintPath(temp, path, grid_size)
                    path_frame2 = paintPath(temp2, path_realTime, grid_size, color=(255,125,125))
                    debugWindowAppend('path1', path_frame1)
                    debugWindowAppend('path realtime', path_frame2)
                    debugWindowShow()
                    while True:
                        if cv2.waitKey(1) ==  ord('g'):
                            break 
                    print("lets start")
                    send_path(path_realTime)
            if RUNONCE:
                # HOLD till esc to quit
                while True:
                    if cv2.waitKey(1) == 27:
                        break 
                break
            if cv2.waitKey(1) == 27:
                break  # esc to quit
        if "TESTING_RUN" == MODE:
            cam.release() # kill camera

    ##### FOR CALIBRATION ######
    elif "CALIBRATION_HSV" == MODE:
        SLIDE_NAME = ['HL', 'SL', 'VL', 'H', 'S', 'V']
        windowName = showUtilities(SLIDE_NAME)
        while True:
            test_frame = cv2.imread('chicken.png')
            # extract maze bndry
            maze_frame = extractMaze(test_frame, CV2_VERSION)
            # marker runing
            [Hl_val,Sl_val,Vl_val,H_val,S_val,V_val] = obtainSlides(SLIDE_NAME)
            bound = [ {'tag':'DEBUG', 'lower':[Hl_val,Sl_val,Vl_val], 'upper':[H_val,S_val,V_val], 'minArea':0 } ]
            coord = detectMark(maze_frame, bound, CV2_VERSION)
            print(coord)
            debugWindowShow(windowName, scale=0.1)
            if cv2.waitKey(1) == 27:
                break  # esc to quit

    cv2.destroyAllWindows() # close all windows


if __name__ == '__main__':
    main()
