
#include <stdio.h>
#include <string>
#include <vector>
#include <algorithm>
#include <dirent.h>
#include "data.h"
#include "../util/gesture.h"

using namespace std;

Data::Data(const string& dir): _dir(dir){
  _data = new vector<Gesture>;
  this->read();
}

void Data::read(){
  DIR *dir;
  struct dirent *ent;
  printf("> reading from %s\n", _dir.c_str());
  if ((dir = opendir(_dir.c_str())) != NULL){
    /* print all the files and directories within directory */
    while ((ent = readdir (dir)) != NULL) {
      if(this->endswith(ent->d_name)){
        printf("Reading %s\n", ent->d_name);
        this->read(ent->d_name);
      }
    }
    closedir(dir);
    printf("> reading complete.\n");
  } else {
    /* could not open directory */
    printf("> reading fail...\n");
  }
}

void Data::read(const string& file){
  string::const_iterator start = file.begin();
  string::const_iterator end = file.end();
  string::const_iterator next = find( start, end, '_' );
  string gestureName = string(start, next);
  Gesture gesture(gestureName);
  float threshold;
  float* result = new float[13];
  int c = 0;
  FILE *f = fopen((_dir + file).c_str(), "r");
	if(f) {
    c = int(fscanf(f, "threshold %f\n", &threshold));
    if(c == 1){
      gesture.setThreshold(threshold);
    }
    c = int(fscanf(f, "%f%f%f%f%f%f%f%f%f%f%f%f%f\n", &result[0], &result[1], &result[2], &result[3], &result[4], &result[5], &result[6], &result[7], &result[8], &result[9], &result[10], &result[11], &result[12]));
    for(; c != -1;){
      gesture(result[0], result[1], result[2], result[3], result[4], result[5], result[6], result[7], result[8], result[9], result[10], result[11], result[12]);
      // printf("%.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f \n", result[0], result[1], result[2], result[3], result[4], result[5], result[6], result[7], result[8], result[9], result[10], result[11]);
      c = int(fscanf(f, "%f%f%f%f%f%f%f%f%f%f%f%f%f\n", &result[0], &result[1], &result[2], &result[3], &result[4], &result[5], &result[6], &result[7], &result[8], &result[9], &result[10], &result[11], &result[12]));
    }
    fclose(f);           
  }
  else {
    cout<< "File open fail..." << endl;
  }
  _data->push_back(gesture);
}


