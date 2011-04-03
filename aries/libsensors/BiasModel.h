#include <math.h>
#include <cutils/log.h>

#pragma GCC optimize ("O3", "fast-math", "tree-vectorize", "inline-limit=600") 

#include <Eigen/Dense>
using namespace Eigen;

#define PLOGD(format, ...) LOGD("%s(%d): " format, __PRETTY_FUNCTION__, __LINE__, ## __VA_ARGS__)

template <class dType, int size>
class BiasModel {
public:
    typedef Matrix<dType, 1, 4> Point;
    BiasModel(dType mSmoothFactor);
    bool updateBias(dType *input);
    void filterPoint(dType *point);

private:
    bool addPoint(Point &coord);
    inline void judgePoints();
    void replaceNeighbors(Point &coord);
    dType mSmoothFactor, mMinDistanceSq, mRadiusSq, mRadius, mErrorAccum;
    int mMode;
    Point mCenter;
    Matrix<dType, Dynamic, 4, RowMajor, size> mData;
    Matrix<dType, Dynamic, 4, RowMajor, size> mCoeff;
    Matrix<dType, Dynamic, 1, 0, size> mVec;
    JacobiSVD<Matrix<dType, Dynamic, 4, 0, size> > mSolver;
};

template <class dType, int size>
BiasModel<dType, size>::BiasModel(dType smoothFactor):
    mMinDistanceSq(0), mRadiusSq(0), mMode(0),
    mCenter(Point::Zero()),
    mData(0,4),
    mCoeff(), mVec(), mSolver(size, 4)
{
    mSmoothFactor = 1 - smoothFactor;
}

template <class dType, int size>
void BiasModel<dType, size>::replaceNeighbors(Point &coord) {
    size_t i, points = mData.rows();
    int minIndex = -1;
    bool replaced = false;
    dType minRadiusSq = 0;
    mVec = mData.rowwise().squaredNorm();
    for (i = 0; i < points;) {
        dType radiusSq = mVec(i);
        if (radiusSq < mMinDistanceSq) {
            if (!replaced) {
                replaced = 1;
                mData.row(i) = coord;
            }
            else if (i < --points) {
                mData.row(i) = mData.row(points);
                mVec(i) = mVec(points);
                continue;
            }
        } else if (radiusSq < minRadiusSq || minIndex == -1) {
            minIndex = i;
            minRadiusSq = radiusSq;
        }
        ++i;
    }
    if (!replaced) {
        if(points < size) {
            mData.conservativeResize(points + 1, NoChange);
            mData.row(points) = coord;
        } else if (minIndex != 1) {
            mData.row(minIndex) = coord;
        } else
            PLOGD("points table full, but no minimum match found");
    } else
        mData.conservativeResize(points, NoChange);
}

template <class dType, int size>
inline void BiasModel<dType, size>::judgePoints() {
    if (mData.rows() < 16 || mMode)
        return;
    int points = mData.rows();
    dType rms(0);
    Point centroid = mData.colwise().sum() / points;
    mVec = (mData.rowwise() - centroid).rowwise().squaredNorm();
    rms = mVec.sum() / (mData.rows() * mData.rows());
    if (!mMode) {
        size_t x, y;
        mVec.minCoeff(&x, &y);
        if (x)
            mData.row(0) = mData.row(x);
        mMode = 1;
        mMinDistanceSq = rms * 4;
        mData.conservativeResize(1, NoChange);
    }
}

template <class dType, int size>
bool BiasModel<dType, size>::addPoint(Point &coord) {
    int minIndex, nearCount;
    switch (mMode) {
      case 0:
        if (mData.rows() < size) {
            int rows = mData.rows();
            mData.conservativeResize(rows + 1, NoChange);
            mData.row(rows) = coord;
        }
        judgePoints();
        return false;
      case 1:
        replaceNeighbors(coord);
    }
    return mData.rows() > 16;
}

template <class dType, int size>
bool BiasModel<dType, size>::updateBias(dType *input) {
    size_t i, j;
    Point coord(input[0], input[1], input[2], 0);
    if (addPoint(coord)) {
        Point newOffset = mData.colwise().sum() / mData.rows();
        mCoeff = mData.rowwise() - newOffset;
        mVec = mCoeff.rowwise().squaredNorm();
        mCoeff += mCoeff;
        mCoeff.col(3).setConstant(1);
        mSolver.compute(mCoeff, ComputeThinU|ComputeThinV);
        Point result = mSolver.solve(mVec);
        newOffset += result;
        newOffset(3) = 0;
        result(3) += result.head(3).squaredNorm();
        mRadiusSq = result(3);
        mRadius = sqrt(result(3));
        mMinDistanceSq = mRadiusSq * 4 / (size);
        result(3) = mRadius;
        newOffset -= mCenter;
        newOffset *= mSmoothFactor;
        mCenter += newOffset * mSmoothFactor;
        dType error = (coord - mCenter).norm() - mRadius;
        mErrorAccum += error * error;
        return true;
    }
    return false;
}

template <class dType, int size>
void BiasModel<dType, size>::filterPoint(dType *input) {
    int i;
    updateBias(input);
    PLOGD("in: <%2.1f,%2.1f,%2.1f> count: %d center: <%2.1f,%2.1f,%2.1f> radius: %2.1f out: <%2.1f,%2.1f,%2.1f>",
        input[0], input[1], input[2], (int)mData.rows(),
        mCenter[0], mCenter[1], mCenter[2], mRadius, 
        input[0]-mCenter[0], input[1]-mCenter[1], input[2]-mCenter[2]);
    for (i = 0; i < 3; i++)
        input[i] -= mCenter[i];
}

#pragma GCC reset_options
