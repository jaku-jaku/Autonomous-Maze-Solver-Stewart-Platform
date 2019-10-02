// Continuously sweep the servo through it's full range
#include "mbed.h"
#include "Servo.h"

Servo myservo(D9);

int main() {
    while(1) {
        for(int i=0; i<20; i++) {
            myservo = i/100.0;
            wait(0.01);
        }
        for(int i=20; i>0; i--) {
            myservo = i/100.0;
            wait(0.01);
        }
    }
}
