
#include <iostream>
#include <vector>
#include "../util/point.h"
#include "../binaryClassifier/binaryClassifier.h"

using namespace std;

int main()
{
  vector<int> index;
  index.push_back(3);
  index.push_back(4);
  index.push_back(5);
  BinaryClassifier classifier(50, 8, index);

  Point move(50);

  cout<< classifier(move)<< endl;
  cout<< classifier(move)<< endl;
  cout<< classifier(move)<< endl;
  cout<< classifier(move)<< endl;
  cout<< classifier(move)<< endl;
  cout<< classifier(move)<< endl;
  cout<< classifier(move)<< endl;
  cout<< classifier(move)<< endl;
  cout<< classifier(move)<< endl;
  cout<< classifier(move)<< endl;
  cout<< classifier(move)<< endl;
  cout<< "finish"<< endl;

  move = Point(0);
		
  cout<< classifier(move)<< endl;
  cout<< classifier(move)<< endl;
  cout<< classifier(move)<< endl;
  cout<< classifier(move)<< endl;
  cout<< classifier(move)<< endl;
  cout<< classifier(move)<< endl;
  cout<< classifier(move)<< endl;
  cout<< classifier(move)<< endl;
  cout<< classifier(move)<< endl;
  cout<< classifier(move)<< endl;
	return 0;
}
