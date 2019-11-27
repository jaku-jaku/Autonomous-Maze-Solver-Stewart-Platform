import cv2
import numpy as np
from debug_framework import *
from cv_helper import *
import math
import copy
from imutils.perspective import four_point_transform
from imutils import contours
import imutils
from config import MAX_HEAT_MAP_VALUE, MAX_HEAT_MAP_POWER, FEATURE_TARGET

def detectMark(frame, list_of_bounds, CV2_VERSION, scale = 0.1):
    frame_cpy = copy.deepcopy(frame)
    frame_hsv = cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)
    frame_hsv_gauss = cv2.GaussianBlur(frame_hsv,(5,5),cv2.BORDER_DEFAULT)
    # define range of color in HSV
    list_of_masks = []
    for bnd in list_of_bounds:
        list_of_masks.append({'tag':bnd['tag'],
        'mask': cv2.inRange(frame_hsv_gauss, np.array(bnd['lower']), np.array(bnd['upper'])),
        'minArea': bnd['minArea'],
        'maskSize': bnd['maskSize']})

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
                size = mask['maskSize']
                if size < 0:
                    size = int(np.sqrt(area)/2)
                cv2.circle(feature_mask, (cX, cY), size, 0, -1)
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

def extractMaze(frame, cv2_version, FOCAL_DIST_TOL=0.7):
    frame_cpy = copy.deepcopy(frame)
    w,h = frame_cpy.shape[:2]
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
    cnts = sorted(contours, key=cv2.contourArea, reverse=True)[:3]
    cv2.drawContours(frame_cpy, cnts, -1, (0,255,0), 3)
    displayCnt = None

    for c in cnts:
    	# approximate the contour
        # compute the center of the contour
        M = cv2.moments(c)
        x = (M["m10"] / M["m00"])
        dX = x - w/2
        y = (M["m01"] / M["m00"]) 
        dY = y - h/2
        dist_rate_from_focal = np.sqrt(dX*dX+dY*dY)/(h/2)
        # if not within focal region, abort
        if dist_rate_from_focal < FOCAL_DIST_TOL:
            DPRINT("Found at", x, y, " PercentDist:", dist_rate_from_focal)
            peri = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, 0.02 * peri, True)
            if len(approx) == 4:
                displayCnt = approx
                cv2.drawContours(frame_cpy, [approx], -1, (0,255,255), 3)
                debugWindowAppend('approx', frame_cpy)
                break
            
    #highlight largest contour
    cv2.drawContours(frame_cpy, cnts[0], -1, (0,0,255), 3)
    #draw orientation line
    cnt = cnts[0]
    [vx,vy,x,y] = cv2.fitLine(cnt, cv2.DIST_L2,0,0.01,0.01)
    val = np.arctan2(vx, vy)*180/np.pi
    DPRINT('--<  Pos  :', x, y)
    DPRINT('--<  Angle:', val, 'degree')
    if vx != 0:
        lefty = int((-x*vy/vx) + y)
        righty = int(((h-x)*vy/vx)+y)
        cv2.line(frame_cpy,(h-1,righty),(0,lefty),(0,255,0),2)
        botty = int(x-(vy/vx)*(w-y))
        toppy = int(x+(vy/vx)*y)
        cv2.line(frame_cpy,(botty,w-1),(toppy,0),(0,0,255),2)

    # display contour
    showImage("contour", frame_cpy)

    if displayCnt is not None:
        try:
            displayCnt_2D = displayCnt.reshape(4, 2)
            maze_extracted = four_point_transform(frame, displayCnt_2D )
            delta_x_transform = displayCnt_2D[0][0] - displayCnt_2D[1][0]
            delta_y_transform = displayCnt_2D[1][1] - displayCnt_2D[0][1]
            angle = np.arcsin(delta_y_transform / delta_x_transform)
            DPRINT('Dx:', delta_x_transform, 'Dy:', delta_y_transform, 'Angle:', angle)
        except:
            EPRINT("ERROR TO PERFORM 4 PT TRANSFORM")
            return None
    else:
        EPRINT("UNABLE TO PERFORM 4 PT TRANSFORM")
        return None

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

def mapMaze_Array(frame, feature_coord, feature_mask, grid_size, ENABLE_GRID=True):
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
                # obstacle
                map_array_1D.append(0)
            else:
                # path
                map_array_1D.append(1)
        map_array.append(map_array_1D)

    map_array[start_array[1]][start_array[0]] = 1
    map_array[end_array[1]][end_array[0]] = 1
    map_array[ball_array[1]][ball_array[0]] = 1

    # highlight walls (visual)
    highlightMapCells(frame, map_array, grid_size,(0,125,255), mark_val=0)

    # highlight start and end color fill (visual)
    highlightMapCellAt(frame, start_array, grid_size, (0,0,255))
    highlightMapCellAt(frame,   end_array, grid_size, (255,0,0))
    highlightMapCellAt(frame,  ball_array, grid_size, (255,255,255))

    return map_array, start_array, end_array, ball_array

def generateContour(maze, bndry):
    w = len(maze)
    h = len(maze[0])
    contour = np.zeros((w,h))
    WALL = 0
    for i in range(0,w):
        for j in range(0,h):
            obj = maze[i][j]
            if obj == WALL: # a wall
                contour[i][j] = 255
                for dir_w in range(-1,2):
                    for dir_h in range(-1,2):
                        for offset in range(1,bndry+1):
                            kw = dir_w*offset
                            kh = dir_h*offset
                            if i+kw<w and j+kh<h and i+kw>0 and j+kh>0:
                                val =  int(MAX_HEAT_MAP_VALUE/np.power(offset, MAX_HEAT_MAP_POWER))
                                if maze[i+kw][j+kh]!=WALL and contour[i+kw][j+kh] < val:
                                    contour[i+kw][j+kh] = val
                                    #DPRINT("+>",i+kw,j+kh, contour[i+kw][j+kh])
    return contour

def mazeSolver_Phase1(frame, cv2_version, grid_size_percent, gradientFactor):
    #maze extraction from captured image
    maze_frame = extractMaze(frame, cv2_version)
    maze_contour = None
    if maze_frame is not None:
        temp2 = maze_frame.copy()
        maze_dim = maze_frame.shape
        grid_size = int(max(maze_dim)*grid_size_percent)
        DPRINT('maze_dimension: ', maze_dim, ' - grid size', grid_size)

        # marker detection
        list_of_bounds = FEATURE_TARGET
        feature_coord, feature_mask = detectMark(maze_frame, list_of_bounds, cv2_version)
        #print(coord)

        #create 2D binarized array of maze (1-> path, 0->obstacle)
        All_tags_exist = True
        for feature in list_of_bounds:
            if feature['tag'] not in feature_coord:
                All_tags_exist = False
                EPRINT('UNABLE to find feature::', feature['tag'])
            else:
                if feature_coord[feature['tag']][2] == -1:
                    All_tags_exist = False
                    DPRINT('Invalid feature::', feature['tag'], feature_coord[feature['tag']])
        maze = []
        start = []
        end = []
        ball = []
        if All_tags_exist:
            maze, start, end, ball = mapMaze_Array(maze_frame, feature_coord, feature_mask, grid_size)
            if maze is not None:
                cnt_mp = temp2
                maze_contour = generateContour(maze, bndry=gradientFactor)
                highlightMapCells(cnt_mp, maze_contour, grid_size,(125,125,255), mark_val=255)
                for i in range(1, gradientFactor):
                    gValue = int(MAX_HEAT_MAP_VALUE/np.power(i, MAX_HEAT_MAP_POWER))
                    highlightMapCells(cnt_mp, maze_contour, grid_size, hsv2bgr(i/gradientFactor,1,1), mark_val=gValue)
                debugWindowAppend('ContourMap', cnt_mp)
    else:
        EPRINT("mazeSolver_Phase1 - UNABLE TO EXTRACT MAZE FRAME")
        return None, None, None, None, None, None, None
    return maze, start, end, ball, maze_frame, grid_size, maze_contour

def pathOptimization(path, counter_map):
    new_path = []
    for x,y in path:
        new_path.append(counter_map[y][x])
    print(new_path)
    return None
