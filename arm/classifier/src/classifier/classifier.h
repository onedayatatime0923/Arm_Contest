
#ifndef _DEFINE_CLASSIFIER_
#define _DEFINE_CLASSIFIER_
#include <vector>
#include "../gesture/gesture.h"
#include "../dtw/dtw.h"

using namespace std;
using namespace DTW;

class Classifier{
  public:
    Classifier(const float& threshold);
    string operator()(vector<vector<float> > target);

    static float euclidean_distance(vector<float> P1, vector<float> P2);
    float evaluateDtwCost(vector<vector<float> > s1, vector<vector<float> > s2);
  private:
    void read();

    float _threshold;
    vector<Gesture> _data;
    SimpleDTW _dtwEval;
    
};
#endif
