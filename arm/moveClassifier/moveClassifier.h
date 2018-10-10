
#ifndef _DEFINE_CLASSIFIER_
#define _DEFINE_CLASSIFIER_
#include <vector>
#include "gesture.h"
#include "dtw.h"

using namespace std;

class MoveClassifier{
  public:
    MoveClassifier(const float& threshold);
    string operator()(vector<Point>& target);

  private:
    void read();

    DTW _dtw;
    float _threshold;
    vector<Gesture>* _data;
    
};
#endif
