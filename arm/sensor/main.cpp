
#include <vector>
#include "mbed.h"
#include "sensor.h"

int main() {
  // construct serial
  Serial pc(USBTX, USBRX); // tx, rx
  pc.baud(57600); 
  // construct mpu9250
  connect_MPU9250(pc);

          
  while(1) {
    vector<float> data = read_data(pc, 2);
    pc.printf("start");
    pc.printf("%f ",data[0]);
    pc.printf("%f ",data[1]);
    pc.printf("%f ",data[2]);
    pc.printf("%f ",data[3]);
    pc.printf("%f ",data[4]);
    pc.printf("%f ",data[5]);
  };
}
