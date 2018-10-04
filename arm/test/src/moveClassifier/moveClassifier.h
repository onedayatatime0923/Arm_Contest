
#ifndef _DEFINE_CLASSIFIER_
#define _DEFINE_CLASSIFIER_
#include <vector>
#include "../util/gesture.h"
#include "../util/dtw.h"

using namespace std;

class Classifier{
  public:
    Classifier(const float& threshold);
    string operator()(vector<Point>& target);

  private:
    void read();

    float _threshold;
    vector<Gesture> _data;
    
};
#endif
