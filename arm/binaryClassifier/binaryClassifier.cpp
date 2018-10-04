
#include <vector>
#include "binaryClassifier.h"

BinaryClassifier::BinaryClassifier(const float& threshold, const int& nStep, const vector<int>& index):
  _threshold(threshold), _nStep(nStep), _index(index), _data(nStep, vector<float>(nStep,0)){};

bool BinaryClassifier::operator() (Point& data) {
   _data.erase(_data.begin());
   vector<float> tmp_data;
   for(size_t i = 0; i < _index.size(); ++i) {
      tmp_data.push_back(data[_index[i]]);
   }
   _data.push_back(tmp_data);
   for(size_t i = 0; i < _nStep; ++i) {
      if(this->norm(_data[i]) > _threshold)
         return true;
   }
   return false;
}

float BinaryClassifier::norm(vector<float>& data) {
  float sum = 0;
  for(int i = 0; i <  _index.size(); ++i) {
    sum += data[i] * data[i];
  }
  return sqrt(sum);
}

