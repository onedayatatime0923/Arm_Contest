
#define nData 16
#include "gesture.h"

void Gesture::operator() (const float& data1, const float& data2, 
      const float& data3, const float& data4, const float& data5, const float& data6,
      const float& data7, const float& data8, const float& data9, const float& data10,
      const float& data11, const float& data12, const float& data13, const float& data14,
      const float& data15, const float& data16){
  vector<float> data;
  data.push_back(data1);
  data.push_back(data2);
  data.push_back(data3);
  data.push_back(data4);
  data.push_back(data5);
  data.push_back(data6);
  data.push_back(data7);
  data.push_back(data8);
  data.push_back(data9);
  data.push_back(data10);
  data.push_back(data11);
  data.push_back(data12);
  data.push_back(data13);
  data.push_back(data14);
  data.push_back(data15);
  data.push_back(data16);
  _data.push_back(data);
};

vector<vector<float> > Gesture::data(){
  return _data;
}
