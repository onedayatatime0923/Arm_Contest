
#ifndef _DEFINE_CLASSIFIER_
#define _DEFINE_CLASSIFIER_
#include <vector>
#include "../gesture/gesture.h"

using namespace std;

class Classifier{
  public:
    Classifier();
    char* operator() (float** target);
  private:
    void read();

    vector<Gesture> _data;
};
#endif
