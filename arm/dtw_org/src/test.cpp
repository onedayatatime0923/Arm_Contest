#include <iostream>
#include <vector>
#include <cmath>
#include "dtw.h"

using namespace std;
double euclidean_distance(vector<double> P1, vector<double> P2) {
    double total = 0.0;
    for (unsigned int i = 0; i < P1.size(); i++) {
        total = total + pow((P1[i] - P2[i]), 2);
    }
    return sqrt(total);
}

double evaluateDtwCost(DTW::SimpleDTW dtw, vector<vector<double> > s1, vector<vector<double> > s2) {
   dtw.Initialize(s1.size(), s2.size());
   return dtw.EvaluateWarpingCost(s1, s2);
}
/*
double printSeq(vector<vector<double> > s) {
   for(size_t i = 0; i < s.size(); ++i) {
      for(size_t j = 0; j < s[i].size(); ++j) {
         cout << s[i][j] << " ";
      }
      cout << "|";
   }
   cout << endl;
}

int main() {
   size_t traj_length = 10;
   size_t iterations = 10;
   vector<vector<double> > test_vec_1;
   for (size_t i = 0; i < traj_length; i++) {
      vector<double> state;
      state.push_back((double)rand());
      state.push_back((double)rand());
      state.push_back((double)rand());
      test_vec_1.push_back(state);
   }
   
   vector<vector<vector<double> > > test_vec_2;
   for (int i = 0; i < iterations; i++) {
      vector<vector<double> > traj;
      for (int j = 0; j < traj_length+1; j++) {
         vector<double> state2;
         state2.push_back((double)rand());
         state2.push_back((double)rand());
         state2.push_back((double)rand());
         traj.push_back(state2);
      }
      test_vec_2.push_back(traj);
   }
   DTW::SimpleDTW my_eval = DTW::SimpleDTW(euclidean_distance);
   for(size_t i = 0; i < iterations; ++i) {
      double cost;
      cout << "Seq1: ";
      printSeq(test_vec_1);
      cout << "Seq2: ";
      printSeq(test_vec_2[i]);
      cost = evaluateDtwCost(my_eval, test_vec_1, test_vec_2[i]);
      cout << cost / double(100000000) << endl;
   }
}
*/
