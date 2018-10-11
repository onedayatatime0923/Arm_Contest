
#include "dtw.h"
#include <iostream>

using namespace std;

DTW::DTW(const int& start, const float& lambda): _start(start), _lambda(lambda){};

float DTW::operator() (vector<Point> &lhs, vector<Point> &rhs){
  _h = lhs.size();
  _w = rhs.size();
  if(_h > _start && _w > _start){
    _gamma = vector<vector<float> >(_h - _start, vector<float>(_w - _start, INF));

    float Best(INF);
    for (int i = 0; i < _h - _start; ++i) {
      for(int j = 0; j < _w - _start; ++j){
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
          _gamma[i][j] = Best * _lambda + lhs[i].euclid_distance(rhs[j]);                   
      }
    }
    return _gamma[_h - _start - 1][_w - _start - 1];
  }
  else return INF;
}
