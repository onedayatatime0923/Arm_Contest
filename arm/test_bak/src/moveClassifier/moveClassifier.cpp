
#include <vector>
#include <cmath>
#include <float.h>
#include <stdio.h>
#include "moveClassifier.h"
#include "../util/dtw.h"
#include "../util/point.h"
#include "data.h"

using namespace std;

MoveClassifier::MoveClassifier(const float& threshold, const int& start, const float& lambda): _threshold(threshold), _dtw(start, lambda){
  this->read();
}

string MoveClassifier::operator()(vector<Point>& target){
  int resultIndex = 0;
  float resultValue = 0;
  float diff = 0;
  for(int i = 0;i < _data->size(); ++i){
    float threshold = (*_data)[i].getThreshold();
    threshold = (threshold == 0)? _threshold: threshold;
    float value = _dtw(target, (*_data)[i].data()) / ((*_data)[i].data().size());
    if( (threshold - value) > diff ){
      diff = threshold - value;
      resultIndex = i;
      resultValue = value;
    };
  };

  printf("%f\n", resultValue);
  if(diff > 0){
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
