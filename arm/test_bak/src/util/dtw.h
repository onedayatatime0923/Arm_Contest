/**
* (c) Daniel Lemire, 2008
* (c) Earlence Fernandes, Vrije Universiteit Amsterdam 2011
*
* This C++ library implements dynamic time warping (DTW). 
* This library includes the dynamic programming solution for vectored input signals represented
* by the class Point. Currently, it has 3 dimensions - x, y, z. More can easily be added to this class.
* No change would be required to the DTW class. Only keep in mind that the distance code has to be updated
* to accomodate more dimensions.
*  
* Time series are represented using STL vectors.
*/

#ifndef _DEFINE_DTW_
#define _DEFINE_DTW_

#include <vector>
#include <cmath>
#include <assert.h>
#include <limits.h>
#include "point.h"

using namespace std;

//DTW code

class DTW 
{
  private:
    int _start;
    float _lambda;
    vector<vector<float> > _gamma;
    int _h, _w;
  public:
    enum { INF = INT_MAX }; //some big number
        

	/**
	* n is the length of the time series 
	*
	* constraint is the maximum warping distance.
	* Typically: constraint = n/10.
	* If you set constraint = n, things will be slower.
	*
	*/
    DTW(const int& start, const float& lambda = 1);
    
	/**
	* This currently uses euclidean distance. You can change it to whatever is needed for your application
	*/
	float operator() (vector<Point> &lhs, vector<Point> &rhs);
};

#endif
