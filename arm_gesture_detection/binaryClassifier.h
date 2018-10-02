#include "mbed.h"
#include "MPU9250.h"

class binaryClassifier{
  public:
    binaryClassifier();
    bool operator()(float data[16])
  private:
    float threshold = 40;
    int check_index = [3, 4, 5]
    uint32_t nStep = 8;
}
