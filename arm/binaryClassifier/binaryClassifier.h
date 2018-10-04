#include <cmath>
#include <vector>
#include "point.h"

using namespace std;

class BinaryClassifier{
  public:
    BinaryClassifier(const float& threshold, const int& nStep, const vector<int>& index);

    bool operator() (Point& data);

  private:
    float norm(vector<float>& data);

    float _threshold;
    int _nStep;
    vector<int>_index;
    vector<vector<float> > _data;
};
