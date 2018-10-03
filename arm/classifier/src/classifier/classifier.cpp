
#include <iostream>
#include <vector>
#include <cmath>
#include "classifier.h"
#include "../dtw/dtw.h"

using namespace std;

Classifier::Classifier(const float& threshold): _threshold(threshold), _dtwEval(this->euclidean_distance){
  this->read();
}

string Classifier::operator()(vector<vector<float> > target){
  int result = 0;
  int loss = _threshold;
  for(int i = 0;i < _data.size(); ++i){
    float value = this->evaluateDtwCost(_data[i].data(), target);
    if( value < loss ){
      loss = value;
      result = i;
    };
  };

  if(loss != _threshold){
    return _data[result].action();
  }
  else{
    return "noOps";
  };
};

float Classifier::euclidean_distance(vector<float> P1, vector<float> P2) {
  float total = 0.0;
  for (unsigned int i = 0; i < P1.size(); i++) {
    total = total + pow((P1[i] - P2[i]), 2);
  }
  return sqrt(total);
}

float Classifier::evaluateDtwCost(vector<vector<float> > s1, vector<vector<float> > s2) {
   _dtwEval.Initialize(s1.size(), s2.size());
   return _dtwEval.EvaluateWarpingCost(s1, s2);
}
