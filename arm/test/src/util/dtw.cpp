
#include "dtw.h"

float DTW::fastdynamic(vector<Point> &lhs, vector<Point> &rhs){
  _h = lhs.size();
  _w = rhs.size();
  _gamma = vector<vector<float> >(_h, vector<float>(_w, INF));

  float Best(INF);
  for (int i = 0; i < _h; ++i) {
    for(int j = max(float(0), i - (_w * _constraint)); j < min(float(_w), i + (_w * _constraint) + 1); ++j){
      Best = INF;
      if(i > 0) 
        Best = _gamma[i - 1][j];
      if(j > 0) 
        Best = min(Best, _gamma[i][j - 1]);
      if((i > 0) && (j > 0))
        Best = min(Best, _gamma[i - 1][j - 1]);
      if((i == 0) && (j == 0))
        _gamma[i][j] = lhs[i].euclid_distance(rhs[j]);
      else 
        _gamma[i][j] = Best + lhs[i].euclid_distance(rhs[j]);                   
    }
  }
  return _gamma[_h-1][_w-1];
}
