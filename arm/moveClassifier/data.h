
#include <string>
#include "mbed.h"
#include "gesture.h"

using namespace std;

class Data{
  public:
    Data( Serial* pc, const string& dir = "/fs/");

    void operator() (const string& file);
    inline vector<Gesture>* data(){
      return _data;
    }

  private:

    Serial* _pc;
    string _dir;
    vector<Gesture>* _data;
};


