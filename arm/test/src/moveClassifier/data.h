
#include <string>
#include "../util/gesture.h"

using namespace std;

class Data{
  public:
    Data(const string& dir = "/fs/");

    void operator() (const string& file);
    inline vector<Gesture>* data(){
      return _data;
    }

  private:

    string _dir;
    vector<Gesture>* _data;
};


