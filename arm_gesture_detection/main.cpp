#include "mbed.h"
#include "sensor_wu.h"

float data[16];
Serial pc(USBTX, USBRX); // tx, rx
        
int main() {
   pc.baud(38400); 
   //Set up
   connect_MPU9250(pc);
   while(1) {
      read_data(pc, data, 0);
   }
}
