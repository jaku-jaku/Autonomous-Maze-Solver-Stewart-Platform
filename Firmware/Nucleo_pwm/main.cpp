// Continuously sweep the servo through it's full range
#include "mbed.h"
#include "common.h"
#include "Servo/Servo.h"
#include "Stewart/Stewart.h"
#include "Stewart/MotionProfile.h"

#define EN_CALIBRATION_MODE 0
#define EN_TEST_MODE        1

#if (EN_TEST_MODE)

// STEP 1: need calibrate individual servo with this one
const Servo::calib_params_t calib1 = {
    .max = {30  , 2000},
    .def = {0   , 1500},
    .min = {-30 , 1000},
};
Servo servo2calib(D9, &calib1);

#if (EN_CALIBRATION_MODE)
int main() {
    while(1) {
        // Tune .def value & try goToDefault and cycle through to calibrate
        goToDefault();
        // try 30 degree and calibrate, change calib1
        // servo2calib = 30;
    }
}
#else
int main() {
    while(1) {
        // step through every single position
        for(int i = calib1.min.ang; i < calib1.max.ang; i++)
        {
            servo2calib = i;
            wait(0.01); // 10ms delay
        }
    }
}
#endif //EN_CALIBRATION_MODE

#else
// STEP 1.5: copy calibrated values below
const Servo::calib_params_t calibs[MAX_NUM_ACTUATORS] = {
    {   // #1
        .max = {30  , 2000},
        .def = {0   , 1500},
        .min = {-30 , 1000},
    },
    {   // #2
        .max = {30  , 2000},
        .def = {0   , 1500},
        .min = {-30 , 1000},
    },
    {   // #3
        .max = {30  , 2000},
        .def = {0   , 1500},
        .min = {-30 , 1000},
    },
    {   // #4
        .max = {30  , 2000},
        .def = {0   , 1500},
        .min = {-30 , 1000},
    },
    {   // #5
        .max = {30  , 2000},
        .def = {0   , 1500},
        .min = {-30 , 1000},
    },
    {   // #6
        .max = {30  , 2000},
        .def = {0   , 1500},
        .min = {-30 , 1000},
    },
};

// map pins properly
Servo m_servos[MAX_NUM_ACTUATORS] = { Servo(D9, &calibs[0]),
                                      Servo(D9, &calibs[1]),
                                      Servo(D9, &calibs[2]),
                                      Servo(D9, &calibs[3]),
                                      Servo(D9, &calibs[4]),
                                      Servo(D9, &calibs[5]) };
Stewart m_platform(m_servos);

//////// MAIN ////////////
int main() {
    while(1) {
        for(int i=-90; i<90; i++) {
            // m_servos[0] = i;
            // m_platform.actuate(0, i);
            COMPASS_MOTION::driveTo(COMPASS_MOTION::DIRECTION_N, m_platform);
            wait(0.01);
        }
    }
}
#endif //(EN_TEST_MODE)
