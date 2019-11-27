#ifndef MOTION_PROFILE_H
#define MOTION_PROFILE_H
#include "../Servo/Servo.h"
#include "../Stewart/Stewart.h"

namespace COMPASS_MOTION{
    typedef enum{
        DIRECTION_NULL,
        DIRECTION_N,
        DIRECTION_E,
        DIRECTION_W,
        DIRECTION_S,
        DIRECTION_NE,
        DIRECTION_NW,
        DIRECTION_SE,
        DIRECTION_SW,
        DIRECTION_NNE,
        DIRECTION_ENE,
        DIRECTION_NNW,
        DIRECTION_WNW,
        DIRECTION_SSE,
        DIRECTION_ESE,
        DIRECTION_SSW,
        DIRECTION_WSW,
        TOTAL_NUM_DIRECTIONS
    } direction_E;

    // STEP 2: Enter these based on the offline computation
    Servo::degree_t angles[TOTAL_NUM_DIRECTIONS][MAX_NUM_ACTUATORS] = {
        {0,0,0,0,0,0},// NULL
        {16, -17, 1, 1, -17, 16},// N
        {10, -8, -20, 18, 8, -11},// E
        {-11, 7, 20, -18, -7, 11},// W
        {-16, 1, -1, -1, 16, -16},// S
        {27, -26, -18, 20, -9, 4},// NE
        {5, -8, 21, -17, -25, 28},// NW - g
        {-4, 9, -22, 17, 26, -28},// SE - h 
        {-29, 25, 18, -20, 8, -5},// SW - i
        {21, -21, -8, 10, -13, 10},//NNE - j
        {19, -17, -19, 19, 0, -3},//NEE - k
        {10, -12, 11, -7, -21, 22},//NNW - l
        {-2, 0, -20, -17, -16, 19},//NWW - m
        {-10, 12, -11, 7, 21, -22},//SSE - n 
        {3, 0, -21, 18, 17, -19},//SEE - o
        {-22, 21, 8, -10, 12, -10},//SSW - p
        {-19, 16, 19, -19, 0, 3},//SWW - q
    };

    void driveTo(int dir, Stewart &platform)
    {
        if (dir < TOTAL_NUM_DIRECTIONS)
        {
            for (int i = 0; i < MAX_NUM_ACTUATORS; i ++)
            {
                platform.actuate(i, angles[dir][i]);
            }
        }
        else
        {
            // sth. that calls this function is wrong
        }
    }
}

#endif
