#include <stdlib.h>
#include <stdio.h>
#include <vector>

#ifndef DTW_H
#define DTW_H

namespace DTW
{

class SimpleDTW
{
private:

    double (*distance_fn_)(std::vector<double> p1, std::vector<double> p2);
    std::vector<double> data_;
    size_t x_dim_;
    size_t y_dim_;
    bool initialized_;


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

    SimpleDTW(size_t x_dim, size_t y_dim, double (*distance_fn)(std::vector<double> p1, std::vector<double> p2));

    SimpleDTW(double (*distance_fn)(std::vector<double> p1, std::vector<double> p2));

    SimpleDTW();

    ~SimpleDTW() {}

    double EvaluateWarpingCost(std::vector<std::vector<double> > sequence_1, std::vector<std::vector<double> > sequence_2);
    void Initialize(size_t x_size, size_t y_size);

};

}

#endif // DTW_H
