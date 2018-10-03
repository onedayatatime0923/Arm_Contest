//#include "mbed.h"
#include <cmath>
using namespace std;

class BinaryClassifier{
  public:
    BinaryClassifier(float threshold, int nStep, vector<int> index);

    bool operator () (vector<float> data);

  private:
    float norm(vector<float> data) {
       float sum = 0;
       for(int i = 0; i <  _index.size(); ++i) {
          sum += data[i] * data[i];
       }
       return sqrt(sum);
    }

    float _threshold;
    int _nStep;
    vector<int>_index;
    vector<vector<float> > _data;
};
