
#ifndef _DEFINE_POINT_
#define _DEFINE_POINT_
#include <cmath>
#include <iostream>

using namespace std;

class Point 
{
public:
	float v0, v1, v2, v3, v4, v5, v6, v7, v8, v9, v10, v11, v12, v13, v14, v15;

	Point(): v0(0), v1(0), v2(0), v3(0), v4(0), v5(0), v6(0), v7(0), v8(0), v9(0), v10(0), v11(0), v12(0), v13(0), v14(0), v15(0){};

	Point(const int& i): v0(i), v1(i), v2(i), v3(i), v4(i), v5(i), v6(i), v7(i), v8(i), v9(i), v10(i), v11(i), v12(i), v13(i), v14(i), v15(i){};

	Point(const float& v0, const float& v1, const float& v2, const float& v3, const float& v4, const float& v5, const float& v6, const float& v7, const float& v8, const float& v9, const float& v10, const float& v11, const float& v12, const float& v13, const float& v14, const float& v15): v0(v0), v1(v1), v2(v2), v3(v3), v4(v4), v5(v5), v6(v6), v7(v7), v8(v8), v9(v9), v10(v10), v11(v11), v12(v12), v13(v13), v14(v14), v15(v15){};
	
  float& operator[] (int& x) {
    if(x == 0) return v0;
    else if(x == 1) return v1;
    else if(x == 2) return v2;
    else if(x == 3) return v3;
    else if(x == 4) return v4;
    else if(x == 5) return v5;
    else if(x == 6) return v6;
    else if(x == 7) return v7;
    else if(x == 8) return v8;
    else if(x == 9) return v9;
    else if(x == 10) return v10;
    else if(x == 11) return v11;
    else if(x == 12) return v12;
    else if(x == 13) return v13;
    else if(x == 14) return v14;
    else return v15;
  }
	//computes the l1 distance with another point
	float l1_distance(const Point &p) 
	{
		return fabs(v0 - p.v0) + fabs(v1 - p.v1) + fabs(v2 - p.v2) + fabs(v3 - p.v3) + 
           fabs(v4 - p.v4) + fabs(v5 - p.v5) + fabs(v6 - p.v6) + fabs(v7 - p.v7) + 
           fabs(v8 - p.v8) + fabs(v9 - p.v9) + fabs(v10 - p.v10) + fabs(v11 - p.v11) + 
           fabs(v12 - p.v12) + fabs(v13 - p.v13) + fabs(v14 - p.v14) + fabs(v15 - p.v15);
	}

	//euclidean distance
	float euclid_distance(const Point &p) 
	{
		return sqrt((v0 - p.v0) * (v0 - p.v0) + (v1 - p.v1) * (v1 - p.v1) + 
                (v2 - p.v2) * (v2 - p.v2) + (v3 - p.v3) * (v3 - p.v3) + 
                (v4 - p.v4) * (v4 - p.v4) + (v5 - p.v5) * (v5 - p.v5) + 
                (v6 - p.v6) * (v6 - p.v6) + (v7 - p.v7) * (v7 - p.v7) + 
                (v8 - p.v8) * (v8 - p.v8) + (v9 - p.v9) * (v9 - p.v9) + 
                (v10 - p.v10) * (v10 - p.v10) + (v11 - p.v11) * (v11 - p.v11) + 
                (v12 - p.v12) * (v12 - p.v12) + (v13 - p.v13) * (v13 - p.v13) + 
                (v14 - p.v14) * (v14 - p.v14) + (v15 - p.v15) * (v15 - p.v15));
	}
	//euclidean distance
	float norm()
	{
		return sqrt((v0 - v0) * (v0 - v0) + (v1 - v1) * (v1 - v1) + 
                (v2 - v2) * (v2 - v2) + (v3 - v3) * (v3 - v3) + 
                (v4 - v4) * (v4 - v4) + (v5 - v5) * (v5 - v5) + 
                (v6 - v6) * (v6 - v6) + (v7 - v7) * (v7 - v7) + 
                (v8 - v8) * (v8 - v8) + (v9 - v9) * (v9 - v9) + 
                (v10 - v10) * (v10 - v10) + (v11 - v11) * (v11 - v11) + 
                (v12 - v12) * (v12 - v12) + (v13 - v13) * (v13 - v13) + 
                (v14 - v14) * (v14 - v14) + (v15 - v15) * (v15 - v15));
	}

	void print(){
	  cout<< v0 << ' ' << v1 << ' ' << v2 << ' ' << v3 << ' ' << v4 << ' ' << v5 << ' ' << v6 << ' ' << v7 << ' ' << v8 << ' ' << v9 << ' ' << v10 << ' ' << v11 << ' ' << v12 << ' ' << v13 << ' ' << v14 << ' ' << v15 << endl;
  };
};
#endif
