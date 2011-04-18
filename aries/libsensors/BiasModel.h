#include <math.h>
#include <algorithm>
#include <cutils/log.h>

#pragma GCC optimize ("O3", "fast-math", "tree-vectorize", "inline-limit=600") 

#include <Eigen/Dense>
using namespace Eigen;
using namespace std;

#define PLOGD(format, ...) LOGD("%s(%d): " format, __PRETTY_FUNCTION__, __LINE__, ## __VA_ARGS__)

template <class dType, int size, int interval = size / 2>
class BiasModel {
public:
    typedef Matrix<dType, 1, 4> Point;
    BiasModel(dType minError, dType maxError, dType power);
    bool updateBias(dType *input);
    void filterPoint(dType *point);

private:
    inline void addPoint(Point &coord);
    int mState, mCount;
    dType mPower, mFitOffset, mFitScale;
    Point mCenter, mCenter1, mCenter2;
    Matrix<dType, Dynamic, 4, RowMajor, size> mData;
    Matrix<dType, Dynamic, 4, RowMajor, size> mCoeff;
    Matrix<dType, Dynamic, 1, 0, size> mVec;
    JacobiSVD<Matrix<dType, Dynamic, 4, 0, size> > mSolver;
};

template <class dType, int size, int interval>
BiasModel<dType, size, interval>::BiasModel(dType minError, dType maxError, dType power):
    mState(0),
    mCount(0),
    mPower(power),
    mCenter(Point::Zero()),
    mCenter1(Point::Zero()),
    mCenter2(Point::Zero()),
    mData(interval, 4),
    mCoeff(),
    mVec(),
    mSolver(size, 4)
{
    EIGEN_STATIC_ASSERT(!(size % interval),MODEL_SIZE_MUST_BE_MULTIPLE_OF_UPDATE_INTERVAL);
    mFitScale = 1 / (maxError - minError);
    mFitOffset = 1 + minError * mFitScale;
}

template <class dType, int size, int interval>
void BiasModel<dType, size, interval>::addPoint(Point &coord) {
    mData.row(mCount) = coord;
    mCount = (mCount + 1) % size;
}

template <class dType, int size, int interval>
bool BiasModel<dType, size, interval>::updateBias(dType *input) {
    size_t i, j;
    bool ret = false;
    Point coord(input[0], input[1], input[2], 0);
    addPoint(coord);
    if (!(mCount % interval)) {
        dType radius;
        mCenter = mCenter1 = mCenter2;
        mCenter2 = mData.colwise().sum() / mData.rows();
        mCoeff = mData.rowwise() - mCenter2;
        mVec = mCoeff.rowwise().squaredNorm();
        mCoeff += mCoeff;
        mCoeff.col(3).setConstant(1);
        mSolver.compute(mCoeff, ComputeThinU|ComputeThinV);
        Point result = mSolver.solve(mVec);
        mCenter2 += result;
        mCenter2(3) = 0;
        radius = sqrt(result(3) + result.head(3).squaredNorm());
        if (!mState) {
            mState = 1;
            mCenter = mCenter1 = mCenter2;
        } else {
            dType error = sqrt((((mData.rowwise() - mCenter2).rowwise().norm().array() - dType(radius)) / radius).matrix().squaredNorm() / mData.rows());
            dType quality = mFitOffset - error * mFitScale;
            quality = min(dType(1), max(dType(0), quality));
            quality = pow(quality, mPower);
            PLOGD("fit sphere <%2.1f,%2.1f,%2.1f><%2.1f> error %.2f quality %.2f", mCenter2[0], mCenter2[1], mCenter2[2], radius, error, quality);
            mCenter2 = mCenter1 + (mCenter2 - mCenter1) * quality;
        }
        if (mData.rows() < (mCount ? mCount : size))
            mData.conservativeResize(mCount ? mCount : size, NoChange);
        ret = true;
    } else {
        dType step = (mCount % interval) / dType(interval);
        mCenter = mCenter1 * (1 - step) + mCenter2 * step;
    }
    return ret;
}

template <class dType, int size, int interval>
void BiasModel<dType, size, interval>::filterPoint(dType *input) {
    int i;
    updateBias(input);
    PLOGD("state: %d count: %d center: <%2.1f,%2.1f,%2.1f>",
        mState, mCount,
        mCenter[0], mCenter[1], mCenter[2]);
    for (i = 0; i < 3; i++)
        input[i] -= mCenter[i];
}

#pragma GCC reset_options
