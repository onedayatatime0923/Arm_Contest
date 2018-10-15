
#ifdef TEST_DATA
#include <iostream>
#include "../util/gesture.h"
#include "../moveClassifier/data.h"

using namespace std;

int main(){
  Data data("data/");
  vector<Gesture>* stack = data.data();
  for(int i = 0;i < stack->size();++i){
    cout<<(*stack)[i].name()<< endl;
    cout<<(*stack)[i].getThreshold()<< endl;
    for(int j = 0;j < (*stack)[i].data().size(); ++j){
      (*stack)[i].data()[j].print();
    }
  }
}
#endif
