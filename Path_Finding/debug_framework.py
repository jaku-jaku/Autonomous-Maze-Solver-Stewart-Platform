import cv2
import numpy as np
import sys, getopt
import time

from config import ENABLE_DEBUG, ENABLE_DPRINT, ENABLE_EPRINT, ENABLE_SPRINT

def SPRINT(*args):
    if ENABLE_SPRINT:
        print( "[SYS]   "+" ".join(map(str,args)))

def DPRINT(*args):
    if ENABLE_DPRINT:
        print( "[DEBUG] "+" ".join(map(str,args)))
        
def EPRINT(*args):
    if ENABLE_EPRINT:
        print( "[ERROR] --x "+" ".join(map(str,args)))

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

def debugWindowRender(title='Debug_Window', scale = 0.2):
    window = None
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
            window = concat_tile(im_list_2d, scale)
            cv2.imshow(title, window)
    return window

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

def parseCML(argv):
    # default
    mode = "TESTING_LOCAL"
    camera_live = False
    # parsing
    try:
        opts, args = getopt.getopt(argv,"m:c:",["mode=","camera="])
    except getopt.GetoptError:
        SPRINT('main.py -m <mode:calib/static/run> -c <live>')
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            SPRINT('PLEASE USE: main.py -m <mode:calib/static/run> -c <live>')
            sys.exit()
        elif opt in ("-m", "--mode"):
            mode = arg
            # remap
            if mode == 'calib':
                mode = "CALIBRATION_HSV"
            elif mode == 'static':
                mode = "TESTING_LOCAL"
                camera_live = False
            elif mode == 'run':
                mode = "TESTING_RUN"
                camera_live = True
            else:
                mode = "UNDEFINED"
                EPRINT("INVALID MODE")
                SPRINT('PLEASE USE: main.py -m <mode:calib/static/run> -c <live>')
        elif opt in ("-c", "--camera"):
            temp = arg
            if arg == 'live':
                camera_live = True

    return mode, camera_live

def init_webCam():
    cam = cv2.VideoCapture(1)
    if not cam.isOpened():
        raise IOError("[ERROR] Cannot open webcam")
    return cam

def grab_webCam_feed(cam, mirror=False):
    ret_val, img = cam.read()
    if mirror:
        img = cv2.flip(img, 1)
    return img

fps_timer_dict = {}
def setFPS_Timer(tag, fps):
    fps_timer_dict.update({tag:{'fps':fps, 'last_time':time.time()}})

def getFPS_Timer(tag):
    if tag in fps_timer_dict:
        if (1/(time.time() - fps_timer_dict[tag]['last_time'])) < fps_timer_dict[tag]['fps']:
            fps_timer_dict[tag]['last_time'] = time.time()
            return True
        else:
            return False
    else:
        return False

def nothing(x):
    pass