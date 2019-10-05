#ifndef STEWART_H
#define STEWART_H
#include "../common.h"
#include "../Servo/Servo.h"

#define MAX_NUM_ACTUATORS 6 

class Stewart {
public:
    typedef Servo* Actuators_t;
    // Creating obj. by attaching 6 series of servo
    Stewart(Actuators_t actuators);
    
    // Actuate a specific servo based on the angle
    void actuate(int index, Servo::degree_t angle);
protected:
    Actuators_t _actuators;
};

#endif
