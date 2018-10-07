
#ifdef TEST_DATA
#include <iostream>
#include "../moveClassifier/data.h"

using namespace std;

int main(){
  Data data("./");
  Gesture g("move");
  data("text_1.txt");
}
#endif
