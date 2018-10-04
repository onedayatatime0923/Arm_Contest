#include <iostream>
#include "vectordtw.h"
#include <vector>

using namespace std;

int main()
{
	cout << "Vector DTW Test" << endl;
	vector<Point> mainVec;
	vector<Point> testVec;
	Point p1(1, 2, 3, 4, 5, 6, 4, 5, 6, 4, 5, 6, 4, 5, 6, 8);
	Point p2(1, 2, 3, 4, 5, 6, 4, 5, 6, 4, 5, 6, 4, 5, 6, 8);
	Point p3(1, 2, 3, 4, 5, 6, 4, 5, 6, 4, 5, 6, 4, 5, 6, 8);
	Point p4(1, 2, 3, 4, 5, 6, 4, 5, 6, 4, 5, 6, 4, 5, 6, 8);

	mainVec.push_back(p1);
	mainVec.push_back(p2);
	testVec.push_back(p3);
	testVec.push_back(p4);

	VectorDTW dtw1(mainVec.size(), 0.3);

	double dist = dtw1.fastdynamic(mainVec, testVec);
		
	cout << "Distance: " << dist << endl;
		
	return 0;
}
