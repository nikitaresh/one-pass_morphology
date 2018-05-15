#ifndef ONE_PASS_MORPHOLOGY_H
#define ONE_PASS_MORPHOLOGY_H

#include <opencv2/opencv.hpp>


class OnePassMorphology
{
public:
    static void dilate(const cv::Mat1b& image, cv::Mat1b& result, const cv::Size& kernel);

private:
    static void dilateRow(const uchar* rowInp, uchar* rowOut, int kernelWidth, int rowSize);
    static void dilateColumn(const uchar* colInp, uchar* colOut, int kernelHeight, int colSize, int stride);

};

#endif // ONE_PASS_MORPHOLOGY_H