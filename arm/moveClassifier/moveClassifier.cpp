
#include <vector>
#include <cmath>
#include <float.h>
#include "moveClassifier.h"
#include "dtw.h"
#include "point.h"
#include "data.h"

using namespace std;

MoveClassifier::MoveClassifier(const float& threshold): _threshold(threshold){
  this->read();
}

string MoveClassifier::operator()(vector<Point>& target, Serial& pc){
  int result = 0;
  float loss = FLT_MAX;
  for(int i = 0;i < _data->size(); ++i){
    float value = _dtw(target, (*_data)[i].data());
    if( value < loss ){
      loss = value;
      result = i;
    };
  };

  if(loss <= _threshold){
    pc.printf(loss);
    return (*_data)[result].action();
  }
  else{
    return "noOps";
  };
};

void MoveClassifier::read(){
  Data data("./");
  data("text_1.txt");
  _data = data.data();
}
