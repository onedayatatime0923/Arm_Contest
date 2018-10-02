//#include "mbed.h"
#include <cmath>
using namespace std;

class binaryClassifier{
  public:
    binaryClassifier();
    binaryClassifier(float thres, uint32_t nStep, vector<int> index)  {
       _threshold = thres;
       _check_index = index;
       _nStep = nStep;
       for(size_t i = 0; i < _nStep; ++i) {
          _grp_data.push_back(new float[_check_index.size()]);
          for(size_t j = 0; j < _check_index.size(); ++j) {
             _grp_data[i][j] = 0;
          }
       }
    }
    bool operator () (float* data) {
       _grp_data.erase(_grp_data.begin());
       float* tmp_data = new float[_check_index.size()];
       for(size_t i = 0; i < _check_index.size(); ++i) {
          tmp_data[i] = data[_check_index[i]];
       }
       _grp_data.push_back(tmp_data);
       for(size_t i = 0; i < _nStep; ++i) {
          if(norm(_grp_data[i]) < _threshold)
             return false;
       }
       return true;
    }
  private:
    float norm(float* data) {
       float sum = 0;
       for(size_t i = 0; i <  _check_index.size(); ++i) {
          sum = sum + data[i] * data[i];
       }
       return sqrt(sum);
    }
    float _threshold;
    vector<int>_check_index;
    uint32_t _nStep;
    vector<float*> _grp_data;
};
