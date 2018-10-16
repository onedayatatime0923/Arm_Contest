
#ifndef _DEFINE_CLASSIFIER_
#define _DEFINE_CLASSIFIER_
#include <vector>
#include "../hmm/hmm.h"
#include "../util/gesture.h"
#include "../util/dtw.h"

using namespace std;

class MoveClassifier{
  public:
    MoveClassifier(const float& threshold, const int& start, const float& lambda = 1);
    string operator()(vector<Point>& target);

  private:
    void read();
    float getThreshold(const int& i);

    DTW _dtw;
    Hmm _hmm;
    string _lastState;
    float _threshold;
    vector<Gesture>* _data;
};
#endif
