
#define nData 16
#include "gesture.h"

void Gesture::operator() (const float& data1, const float& data2, 
      const float& data3, const float& data4, const float& data5, const float& data6,
      const float& data7, const float& data8, const float& data9, const float& data10,
      const float& data11, const float& data12, const float& data13, const float& data14,
      const float& data15, const float& data16){
  float* data = new float[nData];
  data[0] = data1;
  data[1] = data2;
  data[2] = data3;
  data[3] = data4;
  data[4] = data5;
  data[5] = data6;
  data[6] = data7;
  data[7] = data8;
  data[8] = data9;
  data[9] = data10;
  data[10] = data11;
  data[11] = data12;
  data[12] = data13;
  data[13] = data14;
  data[14] = data15;
  data[15] = data16;

  _data[_n] = data;
  ++_n;
};

float** Gesture::data(){
  return _data;
}
