from core import *
from app import *
from config import GRID_SIZE_PERCENT, GRADIENT_FACTOR, ANIMATION_FPS
# +------------------------------------------------------+
# |######################################################|
# |###################### MAIN ##########################|
# |######################################################|
# +------------------------------------------------------+
def main(argv):
    MODE, CAM_LIVE = parseCML(argv)
    CV2_VERSION = cv2.__version__
    SPRINT('CV2 VERSION:' ,CV2_VERSION)
    ##### FOR TESTING RUN_TIME ######
    SPRINT("--> Running", MODE)
    cam = []
    private_index_list = {'begin':0, 'maze':0, 'manual':0, 'debugWindow':0}
    setFPS_Timer('Animation', ANIMATION_FPS)

    STATIC_IMG_SRC = 'img/frame_maze_1.png'
    if CAM_LIVE:
        cam = init_webCam()
        frame = grab_webCam_feed(cam, mirror=False)
        save_frame('begin', frame, private_index_list, override=True)

    if "TESTING_RUN" == MODE or "TESTING_LOCAL" == MODE:
        TERMINATE = False
        while TERMINATE == False:
            if CAM_LIVE: #live feed
                frame = grab_webCam_feed(cam, mirror=False)
            else: # last run
                try:
                    frame = cv2.imread(STATIC_IMG_SRC)
                    if frame is None:
                        EPRINT('NO image available at the src')
                        break
                except:
                    EPRINT('FAIL to read')
            # extract maze bndry
            maze, start, end, ball, maze_frame, grid_size, counter_map, tilt_angle = mazeSolver_Phase1(frame, CV2_VERSION, GRID_SIZE_PERCENT, GRADIENT_FACTOR)
            tilt_angle = float(tilt_angle)
            if maze is None:
                EPRINT('UNABLE to recognize Maze')
            elif len(maze) == 0:
                EPRINT('UNABLE to recognize Maze')
                debugWindowRender()
            else:
                path = find_path(maze, start, end)
                path_realTime = find_path(maze, ball, start)
                path_Custom = find_path(maze, start, end, heat_map=counter_map)
                path_realTime_Custom = find_path(maze, ball, start, heat_map=counter_map)
                if len(path) == 0:
                    EPRINT('No Path Found')
                    debugWindowRender()
                else:
                    # save this working frame
                    save_frame('maze', frame, private_index_list, override=False)
                    SPRINT("--> PATH FOUND <-- ")
                    SPRINT("  > Waiting for 'g' key to cmd, ' ' to abort, 'esc' to quit")
                    temp = maze_frame.copy()
                    temp2 = maze_frame.copy()
                    animation_frame = maze_frame.copy()
                    path_frame1 = paintPath(temp, path, grid_size)
                    path_frame2 = paintPath(temp2, path_realTime, grid_size, color=(255,125,125))
                    path_frame1_cust = paintPath(temp, path_Custom, grid_size, color=(200,45,225))
                    path_frame2_cust = paintPath(temp2, path_realTime_Custom, grid_size, color=(125,125,255))
                    debugWindowAppend('path FIX', path_frame1)
                    debugWindowAppend('path FIX opt', path_frame1_cust)
                    debugWindowAppend('path RT', path_frame2)
                    debugWindowAppend('path RT opt', path_frame2_cust)
                    debugWindowAppend('path Selected', animation_frame)
                    save_frame('debugWindow', debugWindowRender(), private_index_list, override=True)
                    IFRUN = False
                    selected_PATH = path_realTime_Custom
                    index = 0
                    while True:
                        key = cv2.waitKey(1)
                        if key ==  ord('g'):
                            IFRUN = True
                            SPRINT("--> lets start")
                            break
                        elif key ==  ord(' '): # esc to quit
                            SPRINT("--> abort current path, rerun")
                            break
                        elif key ==  27 or key == ord('q'): # esc to quit
                            SPRINT("--> terminate")
                            TERMINATE = True
                            break
                        # animation
                        
                        if getFPS_Timer('Animation'):
                            factor = 5
                            anim_frame = paintPath(animation_frame, selected_PATH[index:index+factor], grid_size, color=(125,125,255), alpha=1)
                            debugWindowAppend('path Selected', anim_frame)
                            debugWindowRender()
                            index = index+factor
                            if index >= len(selected_PATH):
                                index = 0
                                animation_frame = maze_frame.copy()
                            
                    if IFRUN:
                        send_path(selected_PATH, tilt_angle)
            key = cv2.waitKey(1)
            if key == ord('s'):
                save_frame('manual', frame, private_index_list, override=False)
            if key == 27 or key == ord('q'):
                SPRINT("--> terminate")
                TERMINATE = True  # esc to quit

    ##### FOR CALIBRATION ######
    elif "CALIBRATION_HSV" == MODE:
        SLIDE_NAME = ['HL', 'SL', 'VL', 'H', 'S', 'V']
        windowName = showUtilities(SLIDE_NAME)
        while True:
            if CAM_LIVE: #live feed
                test_frame = grab_webCam_feed(cam, mirror=False)
            else: # last run
                try:
                    test_frame = cv2.imread(STATIC_IMG_SRC)
                    if test_frame is None:
                        EPRINT('NO image available at the src')
                        break
                except:
                    EPRINT('FAIL to read')
            # extract maze bndry
            maze_frame, tilt_angle = extractMaze(test_frame, CV2_VERSION)
            if maze_frame is not None:
                # marker runing
                [Hl_val,Sl_val,Vl_val,H_val,S_val,V_val] = obtainSlides(SLIDE_NAME)
                bound = [ {'tag':'DEBUG', 'lower':[Hl_val,Sl_val,Vl_val], 'upper':[H_val,S_val,V_val], 'minArea':0, 'maskSize':-1} ]
                coord = detectMark(maze_frame, bound, CV2_VERSION)
            # print(coord)
            debugWindowRender(windowName, scale=0.1)
            if cv2.waitKey(1) == 27:
                break  # esc to quit

    if CAM_LIVE:
        cam.release() # kill camera
    cv2.destroyAllWindows() # close all windows
    SPRINT("--> END <--")

if __name__ == '__main__':
    main(sys.argv[1:])
