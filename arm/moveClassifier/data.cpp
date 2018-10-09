
#include <stdio.h>
#include <string>
#include <vector>
#include <algorithm>
#include "data.h"
#include "gesture.h"

using namespace std;

Data::Data(const string& dir): _dir(dir){
  _data = new vector<Gesture>;
}

void Data::operator() (const string& file){

	int size = 1024;
	int c, pos;
	char *buffer = (char *)malloc(size);

  string::const_iterator start = file.begin();
  string::const_iterator end = file.end();
  string::const_iterator next = find( start, end, '_' );
  Gesture gesture(string(start, next));
  // cout<<(string(start, next));
  vector<float> result;
  result.reserve(16);

  FILE *f = fopen((_dir + file).c_str(), "r");
	if(f) {
		while(c != -1){ // read all lines in file
      for(int wordCount = 0;wordCount < 16; ++wordCount){
        pos = 0;
        for(c = fgetc(f);(c != -1)&&(c != int('\n'))&&(c != int(' '));c = fgetc(f)){ // read one word
          buffer[pos++] = (char)c;
          if(pos >= size - 1) { // increase buffer length - leave room for 0
            size *=2;
            buffer = (char*)realloc(buffer, size);
          }
        }
        if(c != -1){
          buffer[pos] = 0;
          result.push_back(atof(buffer));
          // cout<< (atof(buffer))<< " ";
        }
      }
      if(c != -1){
        gesture(result[0], result[1], result[2], result[3], result[4], result[5], result[6], result[7], result[8], result[9], result[10], result[11], result[12], result[13], result[14], result[15]);
        // cout<< result[0]<<' '<< result[1]<<' '<< result[2]<<' '<< result[3]<<' '<< result[4]<<' '<< result[5]<<' '<< result[6]<<' '<< result[7]<<' '<< result[8]<<' '<< result[9]<<' '<< result[10]<<' '<< result[11]<<' '<< result[12]<<' '<< result[13]<<' '<< result[14]<<' '<< result[15]<< endl;
        result.clear();
      }
    }
    fclose(f);           
  }
  free(buffer);
  _data->push_back(gesture);
}

