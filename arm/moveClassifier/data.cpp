
#include <stdio.h>
#include <string>
#include <vector>
#include <algorithm>
#include "data.h"
#include "../util/gesture.h"

using namespace std;

Data::Data(Serial* pc, const string& dir): _pc(pc), _dir(dir){
  _data = new vector<Gesture>;
}

void Data::operator() (const string& file){
  string::const_iterator start = file.begin();
  string::const_iterator end = file.end();
  string::const_iterator next = find( start, end, '_' );
  string gestureName = string(start, next);
  Gesture gesture(gestureName);
  // cout<< gestureName<< endl;
  float* result;
  result = new float[16];

  FILE *f = fopen((_dir + file).c_str(), "r");
	if(f) {
    char c = fscanf(f, "%f%f%f%f%f%f%f%f%f%f%f%f%f%f%f%f\n", &result[0], &result[1], &result[2], &result[3], &result[4], &result[5], &result[6], &result[7], &result[8], &result[9], &result[10], &result[11], &result[12], &result[13], &result[14], &result[15]); 
    for(; c != EOF;){
      gesture(result[0], result[1], result[2], result[3], result[4], result[5], result[6], result[7], result[8], result[9], result[10], result[11], result[12], result[13], result[14], result[15]);
      _pc->printf("%f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f\n", &result[0], &result[1], &result[2], &result[3], &result[4], &result[5], &result[6], &result[7], &result[8], &result[9], &result[10], &result[11], &result[12], &result[13], &result[14], &result[15]); 
      // cout<< result[0]<<' '<< result[1]<<' '<< result[2]<<' '<< result[3]<<' '<< result[4]<<' '<< result[5]<<' '<< result[6]<<' '<< result[7]<<' '<< result[8]<<' '<< result[9]<<' '<< result[10]<<' '<< result[11]<<' '<< result[12]<<' '<< result[13]<<' '<< result[14]<<' '<< result[15]<< endl;
      c = fscanf(f, "%f%f%f%f%f%f%f%f%f%f%f%f%f%f%f%f\n", &result[0], &result[1], &result[2], &result[3], &result[4], &result[5], &result[6], &result[7], &result[8], &result[9], &result[10], &result[11], &result[12], &result[13], &result[14], &result[15]); 
    }
    fclose(f);           
  }
  _data->push_back(gesture);
}

