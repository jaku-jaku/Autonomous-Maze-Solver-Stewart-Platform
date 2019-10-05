#ifndef MTE360_SERVO_H
#define MTE360_SERVO_H

#include "mbed.h"

class Servo {
public:
    typedef int degree_t;
    typedef int pw_ms_t;
    typedef struct{
        degree_t   ang;
        pw_ms_t    pwm;
    } pwm_ang_pair_t;

    typedef struct{
        pwm_ang_pair_t max;
        pwm_ang_pair_t def;
        pwm_ang_pair_t min;
    } calib_params_t;

    /** Create a servo object connected to the specified PwmOut pin
     *
     * @param pin PwmOut pin to connect to 
     */
    Servo(PinName pin, const calib_params_t* calib);

    // set position shall be within -90 to 90
    void setPosition(const degree_t angle); 

    // go to default position
    void goToDefault(void);

    /**  Shorthand for the write and read functions */
    Servo& operator= (const degree_t angle);
    pwm_ang_pair_t getStatus(void);

protected:
    PwmOut _pwm;
    pwm_ang_pair_t _range;
    pwm_ang_pair_t _current;
    const calib_params_t* _calibration;
};

#endif
