
#include <vector>

class Classifier{
  public:
    Classifier();
    char* operater()(float** target);
  private:
    void read();

    vector<float**> data;
    vector<const char*> action;
}
