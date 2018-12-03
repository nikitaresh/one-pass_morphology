#ifndef ONE_PASS_MORPHOLOGY_H
#define ONE_PASS_MORPHOLOGY_H

#include <opencv2/opencv.hpp>


class OnePassMorphology
{
public:


    /**
    \brief  Calculation gray image dilatation with given kernel
    
    \param  image - input gray image
    \param  result - output dilatation
    \param  kernel - filter kernel
    
    \retval boolean value of execution, false if the image is empty or the kernel has
            even side or the result has non-empty size different from the image size
    */
    static bool dilate(const cv::Mat1b& image, cv::Mat1b& result,
                       const cv::Size& kernel);

    /**
    \brief  Intrinsics implementation of calculation gray image dilatation with given kernel
    
    \param  image - input gray image
    \param  result - output dilatation
    \param  kernel - filter kernel
    
    \retval boolean value of execution, false if the image is empty or the kernel has
            even side or the result has non-empty size different from the image size
    */
    static bool dilateIntr(const cv::Mat1b& image, cv::Mat1b& result,
                           const cv::Size& kernel);
private:
    static bool dilateRow(const uchar* rowInp, uchar* rowOut,
                          int kernelWidth, int rowSize);
    static bool dilateColumn(const uchar* colInp, uchar* colOut,
                             int kernelHeight, int colSize, int stride);

    // Intrinsics implementation
    static bool dilateColumnsIntr(const cv::Mat1b& image, cv::Mat1b& result,
                                  int kernelHeight);
};

#endif // ONE_PASS_MORPHOLOGY_H