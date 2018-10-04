
#include <cmath>
class Point 
{
public:
	float v0, v1, v2, v3, v4, v5, v6, v7, v8, v9, v10, v11, v12, v13, v14, v15;

	Point(const float& v0, const float& v1, const float& v2, const float& v3, const float& v4, const float& v5, const float& v6, const float& v7, const float& v8, const float& v9, const float& v10, const float& v11, const float& v12, const float& v13, const float& v14, const float& v15): v0(v0), v1(v1), v2(v2), v3(v3), v4(v4), v5(v5), v6(v6), v7(v7), v8(v8), v9(v9), v10(v10), v11(v11), v12(v12), v13(v13), v14(v14), v15(v15){};
	
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
};
