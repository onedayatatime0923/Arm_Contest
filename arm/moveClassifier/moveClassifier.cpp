

#include <vector>
#include <cmath>
#include <float.h>
#include "moveClassifier.h"
#include "dtw.h"
#include "point.h"
#include "data.h"

using namespace std;

MoveClassifier::MoveClassifier(const float& threshold): _threshold(threshold){
  this->read();
}

string MoveClassifier::operator()(vector<Point>& target){
  int result = 0;
  float loss = FLT_MAX;
  for(int i = 0;i < _data->size(); ++i){
    float value = _dtw(target, (*_data)[i].data()) / (*_data)[i].data().size();
    if( value < loss ){
      loss = value;
      result = i;
    };
  };
  cout << loss << endl;
  if(loss <= _threshold){
    return (*_data)[result].action();
  }
  else{
    return "noOps";
  };
};

void MoveClassifier::read(){
  Data data("/fs/");
  _data = data.data();
}


