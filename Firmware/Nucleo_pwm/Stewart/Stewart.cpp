#include "Stewart.h"

Stewart::Stewart(Actuators_t actuators) : _actuators(actuators)
{
    // do nothing
}

void Stewart::actuate(int index, Servo::degree_t angle)
{
    _actuators[index] = (angle);
}