
#ifndef _DEFINE_CLASSIFIER_
#define _DEFINE_CLASSIFIER_
#include <vector>
#include "../gesture/gesture.h"
#include "../dtw/dtw.h"

using namespace std;
using namespace DTW;

class Classifier{
  public:
    Classifier();
    char* operator()(vector<vector<float> > target);

    static double euclidean_distance(vector<float> P1, vector<float> P2);
    double evaluateDtwCost(vector<vector<float> > s1, vector<vector<float> > s2);
  private:
    void read();

    vector<Gesture> _data;
    SimpleDTW _dtwEval;
    
};
#endif
