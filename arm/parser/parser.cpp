
#include <iostream>
#include <string>
#include <stdio.h>
#include <sstream>

using namespace std;

float* parse(const char* data){
  ss  
  float* fData;
  int iDummy;
  int test;
  char* sDummy;
  int flag = sscanf(data, "+IPD,0,%i:%f %f %f %f %f %f %f %f %f %f %f %f %f%n\n", &iDummy, fData, fData + 1,
      fData + 2, fData + 3, fData + 4, fData + 5, fData + 6, fData + 7, fData + 8, fData + 9, fData + 10,
      
  cout<< test<< endl;
  flag = sscanf(data, "%s", sDummy);
  cout<< sDummy<< endl;
  flag = sscanf(data, "%s", sDummy);
  cout<< sDummy<< endl;
  // printf("%f %f %f %f %f %f %f %f %f %f %f %f %f\n", fdata[0], fdata[1], fdata[2], fdata[3], fdata[4], fdata[5], fdata[6], fdata[7], fdata[8], fdata[9], fdata[10], fdata[11], fdata[12]);
  // flag = sscanf(data, "+IPD,0,%i:%f %f %f %f %f %f %f %f %f %f %f %f %f\n", &dummy, fdata, fdata + 1,
  //     fdata + 2, fdata + 3, fdata + 4, fdata + 5, fdata + 6, fdata + 7, fdata + 8, fdata + 9, fdata + 10,
  //     fdata + 11, fdata + 12);
  // printf("%f %f %f %f %f %f %f %f %f %f %f %f %f\n", fdata[0], fdata[1], fdata[2], fdata[3], fdata[4], fdata[5], fdata[6], fdata[7], fdata[8], fdata[9], fdata[10], fdata[11], fdata[12]);
  return fData;
}

int main(){
  const char* data = "+IPD,0,125:.-1.00 -0.07 0.09 2.71 1.57 -2.32 -18.75 3.81 -33.69 1.50 -1.28 1 3\n+IPD,0,125:-1.00 -0.07 0.09 2.71 1.57 -2.32 -18.75 3.81 -33.69 1.50 -1.28 1 3\n";
  float* fdata = parse(data);
  printf("%f %f %f %f %f %f %f %f %f %f %f %f %f\n", fdata[0], fdata[1], fdata[2], fdata[3], fdata[4], fdata[5], fdata[6], fdata[7], fdata[8], fdata[9], fdata[10], fdata[11], fdata[12]);
  

}
  
  
