#include <math.h>
#include <algorithm>
#include <cutils/log.h>

#pragma GCC optimize ("O3", "fast-math", "tree-vectorize", "inline-limit=600") 

#include <Eigen/Dense>
using namespace Eigen;
using namespace std;

#define PLOGD(format, ...) LOGD("%s(%d): " format, __PRETTY_FUNCTION__, __LINE__, ## __VA_ARGS__)

template <class dType, int size>
class BiasModel {
public:
    typedef Matrix<dType, 1, 4> Point;
    BiasModel(dType mSmoothFactor);
    bool updateBias(dType *input);
    void filterPoint(dType *point);

private:
    inline void addPoint(Point &coord);
    inline void judgePoints();
    inline void replaceNeighbors(Point &coord);
    int mState, mCount;
    dType mSmoothFactor, mMinDistanceSq, mRadius, mErrorAccum;
    Point mCenter, mCenter2;
    Matrix<dType, Dynamic, 4, RowMajor, size> mData;
    Matrix<dType, Dynamic, 4, RowMajor, size> mCoeff;
    Matrix<dType, Dynamic, 1, 0, size> mVec;
    JacobiSVD<Matrix<dType, Dynamic, 4, 0, size> > mSolver;
};

template <class dType, int size>
BiasModel<dType, size>::BiasModel(dType smoothFactor):
    mState(0),
    mCount(0),
    mMinDistanceSq(0),
    mRadius(0),
    mErrorAccum(0),
    mCenter(Point::Zero()),
    mCenter2(Point::Zero()),
    mData(0,4),
    mCoeff(),
    mVec(),
    mSolver(size, 4)
{
    mSmoothFactor = 1 - smoothFactor;
}

template <class dType, int size>
void BiasModel<dType, size>::replaceNeighbors(Point &coord) {
    size_t i, points = mData.rows();
    bool replaced = false;
    mVec = mData.rowwise().squaredNorm();
    for (i = 0; i < points;) {
        dType radiusSq = mVec(i);
        if (radiusSq < mMinDistanceSq) {
            if (!replaced) {
                replaced = true;
                mData.row(i) = coord;
            }
            else if (i < --points) {
                mData.row(i) = mData.row(points);
                mVec(i) = mVec(points);
                continue;
            }
        }
        ++i;
    }
    if (!replaced) {
        if(points < size) {
            mData.conservativeResize(points + 1, NoChange);
            mData.row(points) = coord;
        } else {
            size_t j;
            mVec.minCoeff(&i, &j);
            LOGD("replacing nearest neighbor at %d", i);
            mData.row(i) = coord;
        }
    } else
        mData.conservativeResize(points, NoChange);
}

template <class dType, int size>
inline void BiasModel<dType, size>::judgePoints() {
    if (mData.rows() < 16 || mState)
        return;
    int points = mData.rows();
    dType rms(0);
    Point centroid = mData.colwise().sum() / points;
    mVec = (mData.rowwise() - centroid).rowwise().squaredNorm();
    rms = mVec.sum() / (mData.rows() * mData.rows());
    if (!mState) {
        mState = 1;
        mMinDistanceSq = rms * 4;
        mData.conservativeResize(0, NoChange);
    }
}

template <class dType, int size>
void BiasModel<dType, size>::addPoint(Point &coord) {
    int minIndex, nearCount;
    switch (mState) {
        dType error;
        case 0:
            if (mData.rows() < size) {
                int rows = mData.rows();
                mData.conservativeResize(rows + 1, NoChange);
                mData.row(rows) = coord;
            }
            judgePoints();
            break;
        case 3:
            /* The selected values equate to accepting up to 10% rejection over
               the last 100 points. */
            error = (coord - mCenter).norm() / mRadius - 1;
            error *= error;
            mErrorAccum *= 0.988;
            if (error > .3) {
                mErrorAccum += 1;
                if (mErrorAccum > 8.79) {
                    mData.resize(0, NoChange);
                    mState = 2;
                    mErrorAccum = 0;
                    mCount = 0;
                }
                break;
            }
        case 1:
        case 2:
            ++mCount;
            replaceNeighbors(coord);
    }
}

template <class dType, int size>
bool BiasModel<dType, size>::updateBias(dType *input) {
    size_t i, j;
    bool ret = false;
    Point coord(input[0], input[1], input[2], 0);
    addPoint(coord);
    if (mCount >= 4 &&
            mData.rows() >= max(size / 16, 8)) {
        mCount = 0;
        mCenter2 = mData.colwise().sum() / mData.rows();
        mCoeff = mData.rowwise() - mCenter2;
        mVec = mCoeff.rowwise().squaredNorm();
        mCoeff += mCoeff;
        mCoeff.col(3).setConstant(1);
        mSolver.compute(mCoeff, ComputeThinU|ComputeThinV);
        Point result = mSolver.solve(mVec);
        mCenter2 += result;
        mCenter2(3) = 0;
        result(3) += result.head(3).squaredNorm();
        mMinDistanceSq = result(3) * 4 / size;
        mRadius = sqrt(result(3));
        switch (mState) {
            case 0:
            case 1:
                mCenter = mCenter2;
                mState = 2;
                break;
            case 2:
                if (mData.rows() > size / 2) {
                    mState = 3;
                    mErrorAccum = 0;
                }
            default:
                break;
        }
        ret =true;
    }
    mCenter += (mCenter2 - mCenter) * mSmoothFactor;
    return ret;
}

template <class dType, int size>
void BiasModel<dType, size>::filterPoint(dType *input) {
    int i;
    updateBias(input);
    PLOGD("state: %d count: %d center: <%2.1f,%2.1f,%2.1f> radius: %2.1f err: %2.1f accum: %.2f",
        mState, (int)mData.rows(),
        mCenter[0], mCenter[1], mCenter[2], mRadius, 
        (Point(input[0], input[1], input[2], 0) - mCenter).norm() - mRadius,
        mErrorAccum);
    for (i = 0; i < 3; i++)
        input[i] -= mCenter[i];
}

#pragma GCC reset_options
