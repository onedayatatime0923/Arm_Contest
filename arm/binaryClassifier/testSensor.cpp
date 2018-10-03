

#include <vector>
#include "mbed.h"
#include "sensor.h"

Serial pc(USBTX, USBRX); // tx, rx

        
int main() {
   pc.baud(38400); 
   //Set up
   connect_MPU9250(pc);
   while(1) {
      vector<float> data = read_data(pc);
      pc.printf("start");
      pc.printf("%f ",data[0]);
      pc.printf("%f ",data[1]);
      pc.printf("%f ",data[2]);
      pc.printf("%f ",data[3]);
      pc.printf("%f ",data[4]);
      pc.printf("%f ",data[5]);
   }
}
