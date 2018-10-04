
#ifdef TEST_MOVECLASSIFIER
#include <iostream>
#include <stdio.h>
#include "../classifier/classifier.h"


using namespace std;

int main(int argc, char *argv[]){
  Classifier classifier(200);
  vector<vector<float> > target;
  for(int i = 0; i < 3; ++i){
    vector<float> tmp;
    for(int j = 0; j < 16; ++j){
      tmp.push_back(0.0);
    };
    target.push_back(tmp);
  };
  cout<< classifier(target)<<endl;

  return 0;
}
#endif
