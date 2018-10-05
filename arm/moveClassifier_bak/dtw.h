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

    float (*distance_fn_)(std::vector<float> p1, std::vector<float> p2);
    std::vector<float> data_;
    size_t x_dim_;
    size_t y_dim_;
    bool initialized_;


    inline size_t GetDataIndex(size_t x, size_t y)
    {
        return (x * y_dim_) + y;
    }

    inline float GetFromDTWMatrix(size_t x, size_t y)
    {
        return data_[GetDataIndex(x, y)];
    }

    inline void SetInDTWMatrix(size_t x, size_t y, float val)
    {
        data_[GetDataIndex(x, y)] = val;
    }

public:

    SimpleDTW(size_t x_dim, size_t y_dim, float (*distance_fn)(std::vector<float> p1, std::vector<float> p2));

    SimpleDTW(float (*distance_fn)(std::vector<float> p1, std::vector<float> p2));

    SimpleDTW();

    ~SimpleDTW() {}

    float EvaluateWarpingCost(std::vector<std::vector<float> >& sequence_1, std::vector<std::vector<float> >& sequence_2);
    void Initialize(size_t x_size, size_t y_size);

};

}

#endif // DTW_H
