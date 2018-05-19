#ifndef ONE_PASS_MORPHOLOGY_H
#define ONE_PASS_MORPHOLOGY_H

#include <opencv2/opencv.hpp>


class OnePassMorphology
{
public:
    static bool dilate(const cv::Mat1b& image, cv::Mat1b& result, const cv::Size& kernel);

    // Intrinsics implementation
    static bool dilateIntr(const cv::Mat1b& image, cv::Mat1b& result, const cv::Size& kernel);
private:
    static bool dilateRow(const uchar* rowInp, uchar* rowOut, int kernelWidth, int rowSize);
    static bool dilateColumn(const uchar* colInp, uchar* colOut, int kernelHeight, int colSize, int stride);

    // Intrinsics implementation
    static bool dilateColumnsIntr(const cv::Mat1b& image, cv::Mat1b& result, int kernelHeight);
};

#endif // ONE_PASS_MORPHOLOGY_H