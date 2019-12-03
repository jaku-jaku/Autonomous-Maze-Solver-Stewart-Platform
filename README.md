# Proj_StewartPlatform
UW MTE 380 Course Project - 2021 Tron
**Authors: Jack, Jeremy, Tsugumi, Zong

## Summary
Autonomous maze solver with Stewart Platform via a single WebCam.
**Program Directory: https://github.com/JXproject/Proj__StewartPlatform/tree/demo/nov28/Path_Finding

## Possible Command line inputs:
- For local image processing (demo) based on camera feeds from last successful detection
  > python3 main.py -m static 
- For live image processing with live camera feeds
  > python3 main.py -m run
- For hsv calibration
  > python3 main.py -m calib
  > python3 main.py -m calib -c live
- Optional* specifying usb.serial port:
  > python3 main.py -m run -p /dev/tty.usbmodem146300
  
## CV Outputs:
![Software Demo](https://github.com/JXproject/Proj__StewartPlatform/blob/demo/nov28/Path_Finding/img/frame_debugWindow_1.png)

## CAD Renders:
![Stewart Platform CAD Render 1](https://github.com/JXproject/Proj__StewartPlatform/blob/demo/nov28/CAD_Render/render2.png)
![Stewart Platform CAD Render 2](https://github.com/JXproject/Proj__StewartPlatform/blob/demo/nov28/CAD_Render/render3.png)

## List of Items
### Hardware List:
- Servos x6
- MCU board
- WebCam: Logitech C270

### Repo Contains:
- Low level C code
- Inverse Kinematics Python
- Full integration of the solution
  - CV for maze detection and extraction
  - Framework for deploying algorithms
  - CV for marker detection and extraction
  - Grid Occupancy for binarizing maze
  - ~~A star algorithm (using library)~~
  - Custom A* algorithm (Currently Used), with energy/heat map to make the path more centered and optimized
  - Control Algorithm and integration pipelines to hardware interface

## External_API
-   http://kieranwynn.github.io/pyquaternion/


