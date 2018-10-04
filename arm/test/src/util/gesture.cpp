
#define nData 16
#include "gesture.h"
#include "point.h"

void Gesture::operator() (const float& data1, const float& data2, 
      const float& data3, const float& data4, const float& data5, const float& data6,
      const float& data7, const float& data8, const float& data9, const float& data10,
      const float& data11, const float& data12, const float& data13, const float& data14,
      const float& data15, const float& data16){
  Point data(data1, data2, data3, data4, data5, data6,
      data7, data8, data9, data10, data11, data12,
      data13, data14, data15, data16);
  _data.push_back(data);
};

vector<Point>& Gesture::data(){
  return _data;
}

string& Gesture::action(){
  return _action;
}
