#include <stdlib.h>
#include <stdio.h>
#include <vector>
using namespace std;

class SimpleDTW
{
private:

    double (*distance_fn_)(vector<double> p1, vector<double> p2);
    vector<double> data_;
    size_t x_dim_;
    size_t y_dim_;
    bool initialized_;

    void Initialize(size_t x_size, size_t y_size);

    inline size_t GetDataIndex(size_t x, size_t y)
    {
        return (x * y_dim_) + y;
    }

    inline double GetFromDTWMatrix(size_t x, size_t y)
    {
        return data_[GetDataIndex(x, y)];
    }

    inline void SetInDTWMatrix(size_t x, size_t y, double val)
    {
        data_[GetDataIndex(x, y)] = val;
    }

public:

    SimpleDTW(size_t x_dim, size_t y_dim, double (*distance_fn)(vector<double> p1, vector<double> p2));

    SimpleDTW(double (*distance_fn)(vector<double> p1, vector<double> p2));

    SimpleDTW();

    ~SimpleDTW() {}

    double EvaluateWarpingCost(vector<vector<double> > sequence_1, vector<vector<double> > sequence_2);

};
