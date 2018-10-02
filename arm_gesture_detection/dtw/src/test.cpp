#include "dtw.h"
#include <iostream>
#include <vector>
#include <cmath>

using namespace std;
double euclidean_distance(vector<double> P1, vector<double> P2)
{
    double total = 0.0;
    for (unsigned int i = 0; i < P1.size(); i++)
    {
        total = total + pow((P1[i] - P2[i]), 2);
    }
    return sqrt(total);
}

int main() {
   printf("123");
   size_t traj_length = 10;
   size_t iterations = 100;
   vector<vector<double> > test_vec_1;
    for (size_t i = 0; i < traj_length; i++) {
        vector<double> state;
        state.push_back(0.0);
        state.push_back(0.0);
        state.push_back(0.0);
        test_vec_1.push_back(state);
    }
    
    vector<vector<vector<double> > > test_vec_2;
    for (int i = 0; i < iterations; i++)
    {
        vector<vector<double> > traj;
        for (int j = 0; j < traj_length; j++)
        {
            vector<double> state2;
            state2.push_back((double)rand());
            state2.push_back((double)rand());
            state2.push_back((double)rand());
            traj.push_back(state2);
        }
        test_vec_2.push_back(traj);
    }
    DTW::SimpleDTW my_eval = DTW::SimpleDTW(traj_length, traj_length, euclidean_distance);
    double cost;
    cost = my_eval.EvaluateWarpingCost(test_vec_1, test_vec_2[0]);
    cout << cout << endl;
}
