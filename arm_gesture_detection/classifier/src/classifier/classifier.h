
#include <vector>
#include <../gesture/gesture.h>

using namespace std;

class Classifier{
  public:
    Classifier();
    char* operator() (float** target);
  private:
    void read();

    vector<Gesture> _data;
};
