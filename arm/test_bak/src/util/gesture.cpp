
#define nData 16
#include <string>
#include <algorithm>
#include "gesture.h"
#include "point.h"

Gesture::Gesture(const string& name): _name(name), _threshold(0){
  string::const_iterator start = name.begin();
  string::const_iterator end = name.end();
  string::const_iterator next = find( start, end, '_' );
  _action = string(start, next);
};
void Gesture::operator() (const float& data1, const float& data2, 
      const float& data3, const float& data4, const float& data5, const float& data6,
      const float& data7, const float& data8, const float& data9, const float& data10,
      const float& data11, const float& data12, const float data13){
  Point data(data1, data2, data3, data4, data5, data6,
      data7, data8, data9, data10, data11, data12, data13);
  _data.push_back(data);
};

string& Gesture::name(){
  return _name;
}

string& Gesture::action(){
  return _action;
}

void Gesture::setThreshold(const float& threshold){
  _threshold = threshold;
}

float Gesture::getThreshold(){
  return _threshold;
}

vector<Point>& Gesture::data(){
  return _data;
}

