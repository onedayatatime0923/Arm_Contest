
#include <stdio.h>
#include <string>
#include <vector>
#include <algorithm>
#include "data.h"
#include "gesture.h"

using namespace std;

Data::Data(Serial* pc, const string& dir): _pc(pc), _dir(dir){
  _data = new vector<Gesture>;
}

void Data::operator() (){
  DIR *dir;
  struct dirent *ent;
  if ((dir = opendir(_dir.c_str())) != NULL){
    /* print all the files and directories within directory */
    while ((ent = readdir (dir)) != NULL) {
      if(this->endswith(ent->d_name)){
        this->read(ent->d_name);
      }
    }
    closedir(dir);
  } else {
    /* could not open directory */
    cerr<< "Fail..."<< endl;
  }
}

void Data::read(const string& file){
  string::const_iterator start = file.begin();
  string::const_iterator end = file.end();
  string::const_iterator next = find( start, end, '_' );
  string gestureName = string(start, next);
  Gesture gesture(gestureName);
  _pc->printf("%s\n", gestureName);
  // cout<< gestureName << endl;
  float* result = new float[12];
  FILE *f = fopen((_dir + file).c_str(), "r");
	if(f) {
    int c = int(fscanf(f, "%f%f%f%f%f%f%f%f%f%f%f%f\n", &result[0], &result[1], &result[2], &result[3], &result[4], &result[5], &result[6], &result[7], &result[8], &result[9], &result[10], &result[11]));
    for(; c != -1;){
      gesture(result[0], result[1], result[2], result[3], result[4], result[5], result[6], result[7], result[8], result[9], result[10], result[11]);
      // cout<<result[0]<< result[1]<< result[2]<< result[3]<< result[4]<< result[5]<< result[6]<< result[7]<< result[8]<< result[9]<< result[10]<< result[11]<< endl;
      // _pc->printf("%f%f%f%f%f%f%f%f%f%f%f%f\n", &result[0], &result[1], &result[2], &result[3], &result[4], &result[5], &result[6], &result[7], &result[8], &result[9], &result[10], &result[11]);
      c = int(fscanf(f, "%f%f%f%f%f%f%f%f%f%f%f%f\n", &result[0], &result[1], &result[2], &result[3], &result[4], &result[5], &result[6], &result[7], &result[8], &result[9], &result[10], &result[11]));
    }
    fclose(f);           
  }
  else {
    cout<< "Fail..." << endl;
  }
  _data->push_back(gesture);
}


