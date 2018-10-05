
#ifndef _DEFINE_CLASSIFIER_
#define _DEFINE_CLASSIFIER_
#include <vector>
#include "mbed.h"
#include "gesture.h"
#include "dtw.h"

using namespace std;
using namespace DTW;

class MoveClassifier{
  public:
    MoveClassifier(const float& threshold);
    string operator() (vector<vector<float> >& target, Serial& pc);

    static float euclidean_distance(vector<float> P1, vector<float> P2);
    float evaluateDtwCost(vector<vector<float> > s1, vector<vector<float> > s2);
  private:
    void read();

    float _threshold;
    vector<Gesture> _data;
    SimpleDTW _dtwEval;
    
};
#endif
