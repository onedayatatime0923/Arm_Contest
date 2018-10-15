
#include <string>
#include "../util/gesture.h"

using namespace std;

class Data{
  public:
    Data(const string& dir = "/fs/references/");

    void read();
    void setThreshold(const string& name, const float& threshold);
    inline vector<Gesture>* data(){
      return _data;
    }

  private:
    void read(const string& file);
    bool endswith(std::string const & value, std::string const & ending = ".txt");

    string _dir;
    vector<Gesture>* _data;
};

inline bool Data::endswith(std::string const & value, std::string const & ending)
{
  if (ending.size() > value.size()) return false;
  return equal(ending.rbegin(), ending.rend(), value.rbegin());
}
