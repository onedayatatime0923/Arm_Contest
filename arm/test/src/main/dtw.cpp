
#ifdef TEST_DTW
#include <iostream>
#include <vector>
#include "../util/dtw.h"
#include "../util/gesture.h"

using namespace std;

int main()
{
	cout << "DTW Test" << endl;
  Gesture target("target");
  for(int i = 0; i< 10; ++i){
    // target(1.0, 3.0, 5.0, 7.0, 7.0, 17.0, 1.0, 47.0, 1.0, 3.0, 5.0, 7.0, 7.0, 13.0, 17.0, 47.0);
    target(9.0, 9.0, 2.0, 8.0, 7.2, -9.0, 2.0, 8.0, 37.2, 2.0, 3.0, 5.0);
  }

  Gesture ref("ref");
  for(int i = 0; i< 20; ++i){
    // ref(8.0, 8.0, 2.0, 8.0, 7.2, -9.0, 2.0, 8.0, 37.2, 2.0, 3.0, 5.0);
    ref(8.0, 8.0, 2.0, 8.0, 7.2, -9.0, 2.0, 8.0, 37.2, 2.0, 3.0, 5.0);
  };
  cout<< "target data size: "<< target.data().size()<< endl;
  cout<< "ref data size: "<< ref.data().size()<< endl;


	DTW dtw(10);

  double dist;
  for(int i = 0; i<10000; ++i){
    dist = dtw(target.data(), ref.data());
  };
		
  cout << "Distance: " << dist << endl;
		
	return 0;
}
#endif
