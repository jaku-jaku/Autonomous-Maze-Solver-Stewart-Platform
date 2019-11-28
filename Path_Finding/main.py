from core import *
from app import *
from config import GRID_SIZE_PERCENT, GRADIENT_FACTOR, ANIMATION_FPS, COLOR_STRIP
# +------------------------------------------------------+
# |######################################################|
# |###################### MAIN ##########################|
# |######################################################|
# +------------------------------------------------------+
def main(argv):
    MODE, CAM_LIVE, ARG_PORT = parseCML(argv)
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

    if "TESTING_RUN" == MODE or "TESTING_LOCAL" == MODE or "TESTING_LOOP" == MODE:
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
            maze, features_uv, maze_frame, grid_size, heat_map, tilt_angle = mazeSolver_Phase1(frame, CV2_VERSION, GRID_SIZE_PERCENT, GRADIENT_FACTOR)
            if tilt_angle !=  None :
                tilt_angle = float(tilt_angle)
            if maze is None:
                EPRINT('UNABLE to recognize Maze')
            elif len(maze) == 0:
                EPRINT('UNABLE to recognize Maze')
                debugWindowRender()
            else:
                START_TAG = 'ball'
                if START_TAG not in features_uv or features_uv['ball'] is None:
                    EPRINT(' Unable to find balls, abort path planning, continue searching ...')
                else:
                    ball = features_uv['ball']
                    list_mark_tags = ['green_mark', 'blue_mark', 'red_mark']
                    path_dict = {}
                    clr_i = 0
                    for tag in list_mark_tags:
                        if tag in features_uv and features_uv[tag] is not None:
                            path = find_path(maze, ball, features_uv[tag])
                            path_optimized = find_path(maze, ball, features_uv[tag], heat_map=heat_map)
                            if len(path) == 0 and len(path_optimized) == 0:
                                EPRINT('--> No Path Found for:', tag)
                            else:
                                path_dict.update({tag:{'norm':path, 'optimized':path_optimized, 'color':[COLOR_STRIP[clr_i*2], COLOR_STRIP[clr_i*2+1]]}})
                                DPRINT('--> Path Found for:', tag)
                        else:
                            EPRINT('--X Unable to find:', tag)
                        clr_i = clr_i+1

                    if len(path_dict) == 0:
                        EPRINT('No Path Found For any marks')
                        debugWindowRender()
                    else:
                        # save this working frame
                        save_frame('maze', frame, private_index_list, override=False)
                        SPRINT("--> PATH FOUND <-- ")
                        SPRINT("  > Waiting for 'g' key to cmd, ' ' to abort, 'esc' to quit")
                        for tag in list_mark_tags:
                            temp = maze_frame.copy()
                            if tag in path_dict and path_dict[tag]['optimized'] is not None:
                                temp = paintPath(temp, path_dict[tag]['norm'], grid_size, color=path_dict[tag]['color'][0])
                                temp  = paintPath(temp, path_dict[tag]['optimized'], grid_size, color=path_dict[tag]['color'][1])
                            debugWindowAppend(('path:'+tag), temp)

                        # save the debugwindow as well
                        save_frame('debugWindow', debugWindowRender(), private_index_list, override=True)

                        # Select path, and wait for sending
                        animation_frame = maze_frame.copy()
                        debugWindowAppend('path Selected', animation_frame)
                        IFRUN = False
                        selectionIndex = 0
                        tick_index = 0
                        DPRINT("Below is the path dictionary", path_dict)
                        while True:
                            key = cv2.waitKey(1)
                            if key ==  ord('g'):
                                IFRUN = True
                                SPRINT("--> lets start")
                                break
                            elif key == ord('f'): #<-
                                selectionIndex = selectionIndex-1
                                if selectionIndex < 0:
                                    selectionIndex = len(list_mark_tags)*2-1
                                tick_index = 0
                            elif key == ord('h'): #->
                                selectionIndex = selectionIndex+1
                                if selectionIndex >= len(list_mark_tags)*2:
                                    selectionIndex = 0
                                tick_index = 0
                            elif key ==  ord(' '): # esc to quit
                                SPRINT("--> abort current path, rerun")
                                break
                            elif key ==  27 or key == ord('q'): # esc to quit
                                SPRINT("--> terminate")
                                TERMINATE = True
                                break
                            # animation
                            selected_PATH_TAG = list_mark_tags[int(selectionIndex/2)]
                            selected_PATH_CLR = (0,0,0)
                            if selected_PATH_TAG in path_dict and path_dict[selected_PATH_TAG]['optimized'] is not None:
                                if selectionIndex%2 == 0:
                                    selected_PATH_CLR = path_dict[selected_PATH_TAG]['color'][1]
                                    selected_PATH = path_dict[selected_PATH_TAG]['optimized']
                                else:
                                    selected_PATH_CLR = path_dict[selected_PATH_TAG]['color'][0]
                                    selected_PATH = path_dict[selected_PATH_TAG]['norm']
                                if getFPS_Timer('Animation'):
                                    if tick_index == 0:
                                        animation_frame = maze_frame.copy()
                                    factor = 5
                                    anim_frame = paintPath(animation_frame, selected_PATH[tick_index:tick_index+factor], grid_size, color=selected_PATH_CLR, alpha=1)
                                    debugWindowAppend('path Selected', anim_frame)
                                    debugWindowRender()
                                    tick_index = tick_index+factor
                                    if tick_index >= len(selected_PATH):
                                        tick_index = 0
                            else:
                                EPRINT(' selected target was not recognized, [space] to re-search --< ', selected_PATH_TAG)
                                selectionIndex += 2 # skip selection
                                # break
                        if IFRUN:
                            if ARG_PORT is None:
                                send_path(selected_PATH, tilt_angle) # use default
                            else:
                                send_path(selected_PATH, tilt_angle, port=ARG_PORT)

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
