#include <iostream>
#include <vector>
#include <cmath>
#include "binaryClassifier.h"
#include "point.h"

using namespace std;

int main() {
   vector<int> index;
   index.push_back(3);
   index.push_back(4);
   index.push_back(5);
   BinaryClassifier classifier(40, 8, index);

   Point data(40);
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

   data = Point(3);
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
   data.print();
};
