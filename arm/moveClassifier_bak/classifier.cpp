
#include <iostream>
#include <vector>
#include <cmath>
#include "mbed.h"
#include "classifier.h"
#include "dtw.h"

using namespace std;

MoveClassifier::MoveClassifier(const float& threshold): _threshold(threshold), _dtwEval(this->euclidean_distance){
  this->read();
}

string MoveClassifier::operator() (vector<vector<float> >& target, Serial& pc){
  int result = 0;
  float loss = FLT_MAX;
  for(int i = 0;i < _data.size(); ++i){
    float value = this->evaluateDtwCost(_data[i].data(), target);
    if( value < loss ){
      loss = value;
      result = i;
    };
  };

  if(loss != FLT_MAX){
    pc.printf("%f\n", loss);
    return _data[result].action();
  }
  else{
    return "noOps";
  };
};

float MoveClassifier::euclidean_distance(vector<float> P1, vector<float> P2) {
  float total = 0.0;
  for (unsigned int i = 0; i < P1.size(); i++) {
    total = total + pow((P1[i] - P2[i]), 2);
  }
  return sqrt(total);
}

float MoveClassifier::evaluateDtwCost(vector<vector<float> > s1, vector<vector<float> > s2) {
   _dtwEval.Initialize(s1.size(), s2.size());
   return _dtwEval.EvaluateWarpingCost(s1, s2);
}
