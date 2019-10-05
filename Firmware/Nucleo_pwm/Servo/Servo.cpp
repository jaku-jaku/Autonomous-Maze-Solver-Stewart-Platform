#include "Servo.h"
#include "mbed.h"

Servo::Servo(PinName pin, const Servo::calib_params_t* calib) : _pwm(pin), _calibration(calib){
    setPosition(_calibration->def.pwm);
    _range.ang = _calibration->max.ang - _calibration->min.ang;
    _range.pwm = _calibration->max.pwm - _calibration->min.pwm;
}

void Servo::setPosition(const degree_t angle) {
    Servo::pwm_ang_pair_t temp;
    if (angle > _calibration->max.ang)
    {
        temp.ang = _calibration->max.ang;
    }
    else if (angle < _calibration->min.ang)
    {
        temp.ang = _calibration->min.ang;
    }
    else
    {
        temp.ang = angle; // do nothing
    }
    temp.pwm = (temp.ang - _calibration->min.ang)*_range.pwm/_range.ang + _calibration->min.pwm;
    _pwm.pulsewidth_us(temp.pwm);

    // Need a mutex if multi tasking
    _current = temp;
}

Servo::pwm_ang_pair_t Servo::getStatus(void){
    return _current;
}

Servo& Servo::operator= (degree_t angle) { 
    setPosition(angle);
    return *this;
}

