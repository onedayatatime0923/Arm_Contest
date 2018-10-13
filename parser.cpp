
#include <stdio.h>
#include <iostream>

using namespace std;

float * parser(const char*& data){
  float* fData = new float[13];
  sscanf(data,"%f %f %f %f %f %f %f %f %f %f %f %f %f\n", fData, fData + 1, fData + 2, fData + 3, fData + 4, fData + 5, fData + 6, fData + 7, fData + 8, fData + 9, fData + 10, fData + 11, fData + 12);
  return fData;
}

int main(){
  const char* data = "1.2 2.3 1.2 2.3 1.2 2.3 1.2 2.3 1.2 2.3 1.2 2.3 3.4\n";
  float* fData = parser(data);
  for(int i = 0;i < 13;++i){
    printf("%f ", fData[i]);
  }
}
