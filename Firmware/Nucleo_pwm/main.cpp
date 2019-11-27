// Continuously sweep the servo through it's full range
#include "mbed.h"
#include "common.h"
#include "Servo/Servo.h"
#include "Stewart/Stewart.h"
#include "Stewart/MotionProfile.h"

#define EN_CALIBRATION_MODE 0
#define EN_TEST_MODE        0


#if (EN_TEST_MODE)
// STEP 1: need calibrate individual servo with this one
const Servo::calib_params_t calib1 = {
    .max = {60  , 2200},
    .def = {0   , 1350},
    .min = {-60 , 500},
};
Servo servo2calib(D9, &calib1);

#if (EN_CALIBRATION_MODE)
int main() {
    while(1) {
        // Tune .def value & try goToDefault and cycle through to calibrate
        //servo2calib.goToDefault();
        // try 30 degree and calibrate, change calib1
        servo2calib = 30;
    }
}
#else
int main() {
    PwmOut _pwm(D9);
    while(1) {


        _pwm.pulsewidth_us(1400);
        wait(10);
        _pwm.pulsewidth_us(1850);
        wait(10);

//        for(int i = 0; i > calib1.min.ang; i--)
//        {
//            servo2calib = i;
//            wait(0.02); // 10ms delay
//        }
//        wait(5);
//
//        // step through every single position
//        for(int i = calib1.min.ang; i < calib1.max.ang; i++)
//        {
//            servo2calib = i;
//            wait(0.02); // 10ms delay
//        }
//        wait(5);
//        for(int i = calib1.max.ang; i > calib1.min.ang; i--)
//        {
//            servo2calib = i;
//            wait(0.02); // 10ms delay
//        }
//        wait(1);
//        for(int i = calib1.min.ang; i <= 0; i++)
//        {
//            servo2calib = i;
//            wait(0.02); // 10ms delay
//        }
//        wait(5);
    }
}

#endif //EN_CALIBRATION_MODE

#else  //EN_TEST_MODE
// STEP 1.5: copy calibrated values below
const Servo::calib_params_t calibs[MAX_NUM_ACTUATORS] = {
    {   // #1
        .max = {30  , 1850}, //CCW //down
        .def = {0   , 1450},
        .min = {-30 , 1050}, //CW
    },
    {   // #2
        .max = {30  , 1800}, //CCW  //up
        .def = {0   , 1400},
        .min = {-30 , 1000}, //CW
    },
    {   // #3
        .max = {30  , 1800},
        .def = {0   , 1400},
        .min = {-30 , 1000},
    },
    {   // #4
        .max = {30  , 1850},
        .def = {0   , 1450},
        .min = {-30 , 1050},
    },
    {   // #5
        .max = {30  , 1850},
        .def = {0   , 1450},
        .min = {-30 , 1050},
    },
    {   // #6
        .max = {30  , 1800},
        .def = {0   , 1350},
        .min = {-30 , 1000},
    },
};

// map pins properly
Servo m_servos[MAX_NUM_ACTUATORS] = { Servo(D6, &calibs[0]), // checked - flip
                                      Servo(D11, &calibs[1]), // checked - good
                                      Servo(D10, &calibs[2]), // checked -- flip
                                      Servo(D9, &calibs[3]), // checked -- good
                                      Servo(D3, &calibs[4]), // checked - good
                                      Servo(D5, &calibs[5]) }; // checked - flip
Stewart m_platform(m_servos);

//////// MAIN ////////////
int main() {
    COMPASS_MOTION::driveTo(COMPASS_MOTION:: DIRECTION_NULL, m_platform);
    Serial pc(SERIAL_TX, SERIAL_RX);
    wait(5);
    while(1) {
      //  COMPASS_MOTION::driveTo(COMPASS_MOTION:: DIRECTION_NULL, m_platform);
      // wait(0.01);
      char direction = pc.getc();
      int direction_num = direction - 97;
      COMPASS_MOTION::driveTo(direction_num, m_platform);
      wait(0.01);
      pc.putc(direction);
      // COMPASS_MOTION::driveTo(COMPASS_MOTION::DIRECTION_N, m_platform);
      // wait(10);
      // COMPASS_MOTION::driveTo(COMPASS_MOTION::DIRECTION_E, m_platform);
      // wait(10);
      // COMPASS_MOTION::driveTo(COMPASS_MOTION::DIRECTION_W, m_platform);
      // wait(10);
      // COMPASS_MOTION::driveTo(COMPASS_MOTION::DIRECTION_S, m_platform);
      // wait(10);
      // COMPASS_MOTION::driveTo(COMPASS_MOTION::DIRECTION_NE, m_platform);
      // wait(10);
      // COMPASS_MOTION::driveTo(COMPASS_MOTION::DIRECTION_NW, m_platform);
      // wait(10);
      // COMPASS_MOTION::driveTo(COMPASS_MOTION::DIRECTION_SE, m_platform);
      // wait(10);
      // COMPASS_MOTION::driveTo(COMPASS_MOTION::DIRECTION_SW, m_platform);
      // wait(10);
    }
}
#endif //(EN_TEST_MODE)
