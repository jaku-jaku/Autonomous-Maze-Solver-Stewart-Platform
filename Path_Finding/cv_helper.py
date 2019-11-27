import cv2
import numpy as np
import colorsys
from debug_framework import SPRINT

def save_frame(tag, frame, private_index_list, override=False):
    if not override: # we will counting
        private_index_list[tag] = private_index_list[tag]+1
        path = 'img/frame_'+tag+'_'+str(private_index_list[tag])+'.png'
    else:
        path = 'img/frame_'+tag+'.png'
    cv2.imwrite(path, frame)
    SPRINT("--> Image saved ", path)

def random_color(r=None,g=None,b=None):
    rgbl=[255,30,200,40,123, 100, 80, 40, 0, 244]
    np.random.shuffle(rgbl)
    clr = rgbl[0:3]
    if r is not None:
        clr[0] = r
    if g is not None:
        clr[1] = g
    if b is not None:
        clr[2] = b
    return tuple(clr)

def apply_overlay(output, overlay, alpha):
    cv2.addWeighted(overlay, alpha, output, 1 - alpha, 0, output)

def highlightMapCellAt(frame, coord, grid_size, clr, alpha = 0.5):
    temp = frame.copy()
    cv2.rectangle(temp,(coord[0]*grid_size,coord[1]*grid_size),((coord[0]+1)*grid_size, (coord[1]+1)*grid_size),clr,-1)
    apply_overlay(frame, temp, alpha)

def highlightMapCells(frame, mapMask, grid_size, clr, mark_val = 1, alpha = 0.5):
    temp = frame.copy()
    for j in range(0, len(mapMask)):
        for i in range(0, len(mapMask[0])):
            if mapMask[j][i] == mark_val:
                cv2.rectangle(temp,(i*grid_size,j*grid_size),((i+1)*grid_size, (j+1)*grid_size),clr,-1)
    apply_overlay(frame, temp, alpha)

def paintPath(maze_frame, path, grid_size, color=(0,255,0), alpha = 0.5):
    temp = maze_frame.copy()
    for coord in path:
        x = coord[0] * grid_size
        y = coord[1] * grid_size
        cv2.rectangle(temp,(x , y),( x + grid_size, y + grid_size),color,-1)
    apply_overlay(maze_frame, temp, alpha)
    return temp

def paintPathNode(maze_frame, coord, grid_size, color=(0,255,0), alpha = 0.9):
    temp = maze_frame.copy()
    x = coord[0] * grid_size
    y = coord[1] * grid_size
    cv2.rectangle(temp,(x , y),( x + grid_size, y + grid_size),color,-1)
    apply_overlay(maze_frame, temp, alpha)
    return temp

def hsv2bgr(h,s,v):
    r, g, b = tuple(round(i * 255) for i in colorsys.hsv_to_rgb(h,s,v))
    return (b,g,r)
