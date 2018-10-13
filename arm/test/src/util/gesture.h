
#ifndef _DEFINE_GESTURE_
#define _DEFINE_GESTURE_

#include <string>
#include <vector>
#include "point.h"

using namespace std;

class Gesture{
  public:
    Gesture(const string& action): _action(action), _threshold(0){};

    void operator() (const float& data1, const float& data2, 
        const float& data3, const float& data4, const float& data5, const float& data6,
        const float& data7, const float& data8, const float& data9, const float& data10,
        const float& data11, const float& data12, const float data13);

    string& action();
    void setThreshold(const float& threshold);
    float getThreshold();
    vector<Point>& data();

  private:
    string _action;
    float _threshold;
    vector<Point> _data;
};
#endif
