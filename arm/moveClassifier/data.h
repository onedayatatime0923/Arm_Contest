
#include <string>
#include "mbed.h"
#include "gesture.h"

using namespace std;

class Data{
  public:
    Data(Serial* pc, const string& dir = "/fs/references/");

    void operator() (const string& file);
    void read(const string& file);
    inline vector<Gesture>* data(){
      return _data;
    }

  private:
    bool endswith(std::string const & value, std::string const & ending = ".txt");

    Serial* _pc;
    string _dir;
    vector<Gesture>* _data;
};

inline bool Data::endswith(std::string const & value, std::string const & ending)
{
  if (ending.size() > value.size()) return false;
  return equal(ending.rbegin(), ending.rend(), value.rbegin());
}
