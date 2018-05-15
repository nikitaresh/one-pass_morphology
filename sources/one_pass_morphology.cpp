
#include <one_pass_morphology.h>
#include <array>
// #include <omp.h>

void OnePassMorphology::dilate(const cv::Mat1b& image, cv::Mat1b& result, const cv::Size& kernel)
{
    if( result.empty() ) {
        result = cv::Mat1b(image.rows, image.cols);
    }

    if( result.rows != image.rows || result.cols != image.cols ) {
        return;
    }

    if( kernel.width % 2 == 0 || kernel.height % 2 == 0 ) {
        return;
    }

    // #pragma omp parallel for
    for( int rowIndex = 0; rowIndex < result.rows; ++rowIndex ) {
        dilateRow( image.ptr(rowIndex), result.ptr(rowIndex), kernel.width, result.cols );
    }

    // transpose image -> dilateRow() -> transpose image
/*
    result = result.t();
    // #pragma omp parallel for
    for(int rowIndex = 0; rowIndex < result.rows; ++rowIndex)
    {
        dilateRow(result.ptr(rowIndex), result.ptr(rowIndex), kernel.height, result.cols);
    }
    result = result.t();
*/

    // #pragma omp parallel for
    for(int colIndex = 0; colIndex < image.cols; ++colIndex)
    {
        dilateColumn(result.ptr() + colIndex, result.ptr() + colIndex, 
                     kernel.height, image.rows, image.cols);
    }
}


void OnePassMorphology::dilateRow(const uchar* rowInp, uchar* rowOut, int kernelWidth, int rowSize)
{
    if( rowInp == nullptr || rowOut == nullptr || kernelWidth % 2 == 0 ) {
        return;
    }

    std::vector<uchar> maxBefore(rowSize), maxAfter(rowSize);

    for( int row = 0; row < rowSize; ++row )
    {
        maxAfter[row] = (row % kernelWidth) == 0 ? 
                         rowInp[row] : std::max(rowInp[row], maxAfter[row - 1]);

        int curIndex = rowSize - row;
        maxBefore[curIndex - 1] = (row == 0 || ((curIndex) % kernelWidth) == 0) ? 
            rowInp[curIndex - 1] : std::max(rowInp[curIndex - 1], maxBefore[curIndex]);
    }

    int kernelWidthHalf = kernelWidth / 2;
    for( int row = 0; row < rowSize; ++row )
    {
        int indexBefore = std::max(0, row - kernelWidthHalf);
        int indexAfter = std::min(rowSize - 1, row + kernelWidthHalf);

        if( row < kernelWidthHalf ) {
            rowOut[row] = maxAfter[indexAfter];
        }
        else if( (row + kernelWidthHalf > rowSize - 1) && 
                 (rowSize % kernelWidth == 0 || (rowSize % kernelWidth > kernelWidthHalf 
                                                 && indexBefore > kernelWidth * (rowSize / kernelWidth))) )
        {
            rowOut[row] = maxBefore[indexBefore];
        }
        else {
            rowOut[row] = std::max(maxBefore[indexBefore], maxAfter[indexAfter]);
        }
    }
}

void OnePassMorphology::dilateColumn(const uchar* colInp, uchar* colOut, int kernelHeight, 
                                     int colSize, int stride)
{
    if( colInp == nullptr || colOut == nullptr || kernelHeight % 2 == 0 ) {
        return;
    }

    std::vector<uchar> maxBefore(colSize), maxAfter(colSize);

    for( int col = 0; col < colSize; ++col )
    {
        maxAfter[col] = (col % kernelHeight) == 0 ? 
            colInp[col * stride] : std::max(colInp[col * stride], maxAfter[col - 1]);

        int currIndex = colSize - col;
        maxBefore[currIndex - 1] = (col == 0 || (currIndex) % kernelHeight == 0) ? 
            colInp[(currIndex - 1) * stride] : 
            std::max(colInp[(currIndex - 1) * stride], maxBefore[currIndex]);
    }

    int kernelHeightHalf = kernelHeight / 2;
    for( int col = 0; col < colSize; ++col )
    {
        int indexBefore = std::max(0, col - kernelHeightHalf);
        int indexAfter = std::min(colSize - 1, col + kernelHeightHalf);
        if( col < kernelHeightHalf ) {
            colOut[col * stride] = maxAfter[indexAfter];
        }
        else if( (col + kernelHeightHalf > colSize - 1) && 
                 (colSize % kernelHeight == 0 || (colSize % kernelHeight > kernelHeightHalf 
                                                  && indexBefore > kernelHeight * (colSize / kernelHeight))) )
        {
            colOut[col * stride] = maxBefore[indexBefore];
        }
        else {
            colOut[col * stride] = std::max(maxBefore[indexBefore], maxAfter[indexAfter]);
        }
    }
}
