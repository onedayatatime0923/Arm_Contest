#include <iostream>
#include <vector>
#include <cmath>
#include "binaryClassifier.h"

using namespace std;

int main() {
   vector<int> index;
   index.push_back(3);
   index.push_back(4);
   index.push_back(5);
   BinaryClassifier classifier(40, 8, index);

   vector<float> data(16,50);
   cout << classifier(data) << endl;
   cout << classifier(data) << endl;
   cout << classifier(data) << endl;
   cout << classifier(data) << endl;
   cout << classifier(data) << endl;
   cout << classifier(data) << endl;
   cout << classifier(data) << endl;
   cout << classifier(data) << endl;
   cout << classifier(data) << endl;
   cout << classifier(data) << endl;
   cout << classifier(data) << endl;
   cout << classifier(data) << endl;
   cout << classifier(data) << endl;
   cout << "finish"<< endl;

   data = vector<float> (16,0);
   cout << classifier(data) << endl;
   cout << classifier(data) << endl;
   cout << classifier(data) << endl;
   cout << classifier(data) << endl;
   cout << classifier(data) << endl;
   cout << classifier(data) << endl;
   cout << classifier(data) << endl;
   cout << classifier(data) << endl;
   cout << classifier(data) << endl;
   cout << classifier(data) << endl;
   cout << "finish"<< endl;
};
