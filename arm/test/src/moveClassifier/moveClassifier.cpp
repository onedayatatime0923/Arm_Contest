
#include <vector>
#include <cmath>
#include <float.h>
#include <stdio.h>
#include "moveClassifier.h"
#include "../util/dtw.h"
#include "../util/point.h"
#include "data.h"

using namespace std;

MoveClassifier::MoveClassifier(const float& threshold, const int& start, const float& lambda): _dtw(start, lambda), _lastState("dummy"), _threshold(threshold){
  this->read();
}

string MoveClassifier::operator()(vector<Point>& target){
  float diff = 0;
  int resultIndex = 0;
  float resultValue = 0;
  float resultThreshold;
  for(int i = 0;i < _data->size(); ++i){
    float threshold = this->getThreshold(i);
    float value = _dtw(target, (*_data)[i].data()) / ((*_data)[i].data().size());
    if( (threshold - value) > diff ){
      diff = threshold - value;
      resultIndex = i;
      resultValue = value;
      resultThreshold = threshold;
    };
  };
  printf("%f\n", resultThreshold);
  printf("%f\n", resultValue);
  if(diff > 0){
    _lastState = (*_data)[resultIndex].action();
    return (*_data)[resultIndex].action();
  }
  else{
    return "noOps";
  };
};

void MoveClassifier::read(){
  Data data("data/");
  _data = data.data();
}

float MoveClassifier::getThreshold(const int& i){
  float threshold = (*_data)[i].getThreshold();
  threshold = (threshold == 0)? _threshold: threshold;
  threshold += _hmm.query(_lastState, (*_data)[i].action());
  return threshold;
}
