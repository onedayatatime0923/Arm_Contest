
#ifndef _DEFINE_POINT_
#define _DEFINE_POINT_
#include <cmath>
#include <iostream>

using namespace std;

class Point 
{
public:
	float _v0, _v1, _v2, _v3, _v4, _v5, _v6, _v7, _v8, _v9, _v10, _v11, _v12;

	Point(): _v0(0), _v1(0), _v2(0), _v3(0), _v4(0), _v5(0), _v6(0), _v7(0), _v8(0), _v9(0), _v10(0), _v11(0), _v12(0){};

	Point(const int& i): _v0(i), _v1(i), _v2(i), _v3(i), _v4(i), _v5(i), _v6(i), _v7(i), _v8(i), _v9(i), _v10(i), _v11(i), _v12(i){};

	Point(const float& v0, const float& v1, const float& v2, const float& v3, const float& v4, const float& v5, const float& v6, const float& v7, const float& v8, const float& v9, const float& v10, const float& v11, const float& v12): _v0(v0), _v1(v1), _v2(v2), _v3(v3), _v4(v4), _v5(v5), _v6(v6), _v7(v7), _v8(v8), _v9(v9), _v10(v10), _v11(v11), _v12(v12){};
	
	void operator() (const float& v0, const float& v1, const float& v2, const float& v3, const float& v4, const float& v5, const float& v6, const float& v7, const float& v8, const float& v9, const float& v10, const float& v11, const float& v12){
    _v0 = v0;
    _v1 = v1;
    _v2 = v2;
    _v3 = v3;
    _v4 = v4;
    _v5 = v5;
    _v6 = v6;
    _v7 = v7;
    _v8 = v8;
    _v9 = v9;
    _v10 = v10;
    _v11 = v11;
    _v12 = v12;
  }
  float& operator[] (int& x) {
    if(x == 0) return _v0;
    else if(x == 1) return _v1;
    else if(x == 2) return _v2;
    else if(x == 3) return _v3;
    else if(x == 4) return _v4;
    else if(x == 5) return _v5;
    else if(x == 6) return _v6;
    else if(x == 7) return _v7;
    else if(x == 8) return _v8;
    else if(x == 9) return _v9;
    else if(x == 10) return _v10;
    else if(x == 11) return _v11;
    else if(x == 12) return _v12;
  }
	//computes the l1 distance with another point
	float l1_distance(const Point &p) 
	{
		return fabs(_v0 - p._v0) + fabs(_v1 - p._v1) + fabs(_v2 - p._v2) + fabs(_v3 - p._v3) + 
           fabs(_v4 - p._v4) + fabs(_v5 - p._v5) + fabs(_v6 - p._v6) + fabs(_v7 - p._v7) + 
           fabs(_v8 - p._v8) + fabs(_v9 - p._v9) + fabs(_v10 - p._v10) + fabs(_v11 - p._v11) +
           fabs(_v12 - p._v12);
	}

	//euclidean distance
	float euclid_distance(const Point &p) 
	{
		return sqrt((_v0 - p._v0) * (_v0 - p._v0) + (_v1 - p._v1) * (_v1 - p._v1) + 
                (_v2 - p._v2) * (_v2 - p._v2) + (_v3 - p._v3) * (_v3 - p._v3) + 
                (_v4 - p._v4) * (_v4 - p._v4) + (_v5 - p._v5) * (_v5 - p._v5) + 
                (_v6 - p._v6) * (_v6 - p._v6) + (_v7 - p._v7) * (_v7 - p._v7) + 
                (_v8 - p._v8) * (_v8 - p._v8) + (_v9 - p._v9) * (_v9 - p._v9) + 
                (_v10 - p._v10) * (_v10 - p._v10) + (_v11 - p._v11) * (_v11 - p._v11) +
                (_v12 - p._v12) * (_v12 - p._v12));
	}
	//euclidean distance
	float norm()
	{
		return sqrt((_v0 - _v0) * (_v0 - _v0) + (_v1 - _v1) * (_v1 - _v1) + 
                (_v2 - _v2) * (_v2 - _v2) + (_v3 - _v3) * (_v3 - _v3) + 
                (_v4 - _v4) * (_v4 - _v4) + (_v5 - _v5) * (_v5 - _v5) + 
                (_v6 - _v6) * (_v6 - _v6) + (_v7 - _v7) * (_v7 - _v7) + 
                (_v8 - _v8) * (_v8 - _v8) + (_v9 - _v9) * (_v9 - _v9) + 
                (_v10 - _v10) * (_v10 - _v10) + (_v11 - _v11) * (_v11 - _v11) +
                (_v11 - _v11) * (_v11 - _v11));
	}

	void print(){
	  cout<< _v0 << ' ' << _v1 << ' ' << _v2 << ' ' << _v3 << ' ' << _v4 << ' ' << _v5 << ' ' << _v6 << ' ' << _v7 << ' ' << _v8 << ' ' << _v9 << ' ' << _v10 << ' ' << _v11 << ' ' << _v12 << endl;
  };
};
#endif
