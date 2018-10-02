
#ifndef _DEFINE_GESTURE_
#define _DEFINE_GESTURE_

#include <string>

using namespace std;

class Gesture{
  public:
    Gesture(const string& action, int step):
      _action(action), _n(0), _step(step){
        _data = new float*[step];
    };

    void operator() (const float& data1, const float& data2, 
        const float& data3, const float& data4, const float& data5, const float& data6,
        const float& data7, const float& data8, const float& data9, const float& data10,
        const float& data11, const float& data12, const float& data13, const float& data14,
        const float& data15, const float& data16);

    float** data();

  private:
    int _n;
    int _step;
    string _action;
    float** _data;
};
#endif
