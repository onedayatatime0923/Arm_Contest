
#include <vector>
#include "mbed.h"
#include "sensor/sensor.h"

Serial pc(USBTX, USBRX); // tx, rx

vector<int> index(3,4,5);
BinaryClassifier classifier(400, 2, index);
        
int main() {
   pc.baud(38400); 
   //Set up
   connect_MPU9250(pc);
   while(1) {
      vector<float> data = read_data(pc);
      pc.printf("start");
      pc.printf(data[0]);
      pc.printf(data[1]);
      pc.printf(data[2]);
      pc.printf(data[3]);
      pc.printf(data[4]);
      pc.printf(data[5]);
      pc.printf(classifier(data));
   }
}
