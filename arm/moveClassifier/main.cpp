
#include <vector>
#include "mbed.h"
#include "sensor.h"
#include "binaryClassifier.h"
#include "classifier.h"

int main() {

  // construct pc serial
  Serial pc(USBTX, USBRX); // tx, rx
  pc.baud(57600); 
  
  // construct binary classifier
  vector<int> verboseIndex;
  verboseIndex.push_back(3);
  verboseIndex.push_back(4);
  verboseIndex.push_back(5);
  BinaryClassifier binaryClassifier(50, 20, verboseIndex);
  // construct movement classifier
  MoveClassifier moveClassifier(900);
  // construct mpu9250
  connect_MPU9250(pc);

  // set up
  vector<vector<float> > target;
  while(1) {
    vector<float> data = read_data(pc);
    if( !binaryClassifier(data) ){
      pc.printf("stop\n");
    }
    else{
      target.push_back(data);
      string move = moveClassifier(target, pc);
      pc.printf("%s\n",move);
      if (move == "noOps"){
      }
      else {
        target.clear();
      };
    };
  };
}

