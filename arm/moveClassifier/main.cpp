
#include <vector>
#include "mbed.h"
#include "point.h"
#include "sensor.h"
#include "binaryClassifier.h"
#include "moveClassifier.h"
#include "speaker.h"
Serial pc(USBTX, USBRX);
AnalogOut DACout(A5);
SDBlockDeviceDISCOF469NI bd;
FATFileSystem fs("fs");
int main() {

  // construct pc serial
  pc.baud(57600); 
  
  // construct binary classifier
  vector<int> verboseIndex;
  verboseIndex.push_back(3);
  verboseIndex.push_back(4);
  verboseIndex.push_back(5);
  BinaryClassifier binaryClassifier(30, 5, verboseIndex);

  // construct mpu9250
  connect_MPU9250(pc);
  // construct speaker and SD card
  Speaker s(&DACout);
  bd.init();
  fs.mount(&bd);
  // construct movement classifier
  MoveClassifier moveClassifier(380);
  // set up
  vector<Point> target;
  string act, last_act;
  last_act = "noOps";
  unsigned short sizeThreshold = 25;
  while(1) {
    Point data = read_data(pc, 0);
    if( !binaryClassifier(data) ){
      pc.printf("stop\n");
      target.clear();
    }
    else{
      target.push_back(data);
      if(target.size() >= sizeThreshold) {
        act = ((moveClassifier(target)));
      }
      else act = "noOps";
      if (act == "noOps"){
        pc.printf("%s\n",act);
      }
      else {
        if(last_act != act) {
            pc.printf("%s\n",act);
            s(act.c_str());
            last_act = act;
        }
        target.clear();
      }
    };
  };
}





