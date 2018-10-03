
#include <vector>
#include "mbed.h"
#include "sensor.h"
#include "binaryClassifier.h"

int main() {

  Serial pc(USBTX, USBRX); // tx, rx

  vector<int> verboseIndex;
  verboseIndex.push_back(3);
  verboseIndex.push_back(4);
  verboseIndex.push_back(5);
  BinaryClassifier classifier(400, 2, verboseIndex);
          
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
    pc.printf("%s", classifier(data) ? "true" : "false");
  };
}
