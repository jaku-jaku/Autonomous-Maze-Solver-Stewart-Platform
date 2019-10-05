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
        TOTAL_NUM_DIRECTIONS
    } direction_E;

    // STEP 2: Enter these based on the offline computation 
    Servo::degree_t angles[TOTAL_NUM_DIRECTIONS][MAX_NUM_ACTUATORS] = {
        {0,0,0,0,0,0},// NULL
        {0,0,0,0,0,0},// N
        {0,0,0,0,0,0},// E
        {0,0,0,0,0,0},// W
        {0,0,0,0,0,0},// S
        {0,0,0,0,0,0},// NE
        {0,0,0,0,0,0},// NW
        {0,0,0,0,0,0},// SE
        {0,0,0,0,0,0},// SW
    };
    
    void driveTo(direction_E dir, Stewart &platform)
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
