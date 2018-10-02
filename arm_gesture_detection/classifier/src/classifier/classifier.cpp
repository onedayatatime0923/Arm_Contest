
#include <iostream>
#include <vector>
#include <cmath>
#include "classifier.h"
#include "../dtw/dtw.h"

using namespace std;

Classifier::Classifier(): _dtwEval(this->euclidean_distance){
  this->read();
}

char* Classifier::operator()(vector<vector<float> > target){
}

double Classifier::euclidean_distance(vector<float> P1, vector<float> P2) {
  double total = 0.0;
  for (unsigned int i = 0; i < P1.size(); i++) {
    total = total + pow((P1[i] - P2[i]), 2);
  }
  return sqrt(total);
}

double Classifier::evaluateDtwCost(vector<vector<float> > s1, vector<vector<float> > s2) {
   _dtwEval.Initialize(s1.size(), s2.size());
   cout<<_dtwEval.EvaluateWarpingCost(s1, s2)<< endl;
}
