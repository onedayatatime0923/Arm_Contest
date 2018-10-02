#include <iostream>
#include <vector>
#include <cmath>
#include "binaryClassifier.h"

using namespace std;

int main() {
   /* vector<float*> ptr;
   for(size_t i = 0; i < 4; ++i) {
      ptr.push_back(new float[3]);
      for(size_t j = 0; j < 3; ++j) {
         ptr[i][j] = i;
         cout << ptr[i][j] << " ";
      }
      cout << endl;
   }
   ptr.erase(ptr.begin());
   cout << "Size: " << ptr.size() << endl;
   for(size_t i = 0; i < ptr.size(); ++i) {
      for(size_t j = 0; j < 3; ++j) {
         cout << sqrt(ptr[i][j]) << " ";
      }
      cout << endl;
   }
   */
   vector<int> index;
   index.push_back(3);
   index.push_back(4);
   index.push_back(5);
   binaryClassifier a =  binaryClassifier(40, 8, index);
   float* data;
   data = new float[16];
   for(size_t i = 0; i < 16; ++i) data[i] = 40;
   cout << a(data) << endl;
   cout << a(data) << endl;
   cout << a(data) << endl;
   cout << a(data) << endl;
   cout << a(data) << endl;
   cout << a(data) << endl;
   cout << a(data) << endl;
   cout << a(data) << endl;
   cout << a(data) << endl;
   cout << a(data) << endl;
   cout << a(data) << endl;
   cout << a(data) << endl;
   cout << a(data) << endl;

}
