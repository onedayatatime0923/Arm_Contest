
#include <iostream>
#include <vector>
#include "binaryClassifier.h"

BinaryClassifier::BinaryClassifier(float threshold, int nStep, vector<int> index):
  _threshold(threshold), _nStep(nStep), _index(index), _data(nStep, vector<float>(16,0)){
    for(int i = 0; i < _nStep; ++i){
      for(int j = 0; j< 16; ++j){
        cout<< _data[i][j];
      };
      cout<< endl;
    };
};
bool BinaryClassifier::operator () (vector<float> data) {
   _data.erase(_data.begin());
   vector<float> tmp_data;
   for(size_t i = 0; i < _index.size(); ++i) {
      tmp_data.push_back(data[_index[i]]);
   }
   _data.push_back(tmp_data);
   for(size_t i = 0; i < _nStep; ++i) {
      if(norm(_data[i]) > _threshold)
         return true;
   }
   return false;
}
