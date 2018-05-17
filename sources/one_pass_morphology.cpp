
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
    if( rowInp == nullptr || rowOut == nullptr || kernelWidth % 2 == 0 || kernelWidth < 2 ) {
        return;
    }

    std::vector<uchar> maxBefore(rowSize), maxAfter(rowSize);
    for( int row = kernelWidth; row < rowSize; row += kernelWidth )
    {
        maxAfter[row] = rowInp[row];
        maxBefore[row - 1] = rowInp[row - 1];
    }
    maxAfter[0] = rowInp[0];
    maxBefore[rowSize - 1] = rowInp[rowSize - 1];

    for( int step = 0; step < rowSize / kernelWidth; ++step )
    {
        for( int row = 1; row < kernelWidth; ++row )
        {
            maxAfter[step * kernelWidth + row] = std::max(rowInp[step * kernelWidth + row], maxAfter[step * kernelWidth + row - 1]);
            maxBefore[(step + 1) * kernelWidth - row - 1] = 
                std::max(rowInp[(step + 1) * kernelWidth - row - 1], maxBefore[(step + 1) * kernelWidth - row]);
        }
    }

    const int wholeKernelsSize = kernelWidth * (rowSize / kernelWidth);
    for( int row = wholeKernelsSize; row < rowSize - 1; ++row )
    {
        maxAfter[row + 1] = std::max(rowInp[row + 1], maxAfter[row]);

        int curIndex = rowSize - row + wholeKernelsSize - 1;
        maxBefore[curIndex - 1] = std::max(rowInp[curIndex - 1], maxBefore[curIndex]);
    }

    const int kernelWidthHalf = kernelWidth / 2;

    const int kernelWidthHalfEnd = std::min(kernelWidthHalf, rowSize - 1);
    for( int row = 0; row < kernelWidthHalfEnd; ++row )
    {
        int index = std::min(row + kernelWidthHalf, rowSize - 1);
        rowOut[row] = maxAfter[index];
    }

    for( int row = kernelWidthHalf; row < rowSize - kernelWidthHalf; ++row ) {
        rowOut[row] = std::max(maxBefore[row - kernelWidthHalf], maxAfter[row + kernelWidthHalf]);
    }

    const int lastPartStart = std::max(rowSize - kernelWidthHalf, 0);
    for( int row = lastPartStart; row < rowSize; ++row )
    {
        int indexBefore = std::max(0, row - kernelWidthHalf);
        int indexAfter = std::min(rowSize - 1, row + kernelWidthHalf);
        if( indexBefore > wholeKernelsSize ) {
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
    if( colInp == nullptr || colOut == nullptr || kernelHeight % 2 == 0 || kernelHeight < 2 ) {
        return;
    }

    std::vector<uchar> maxBefore(colSize), maxAfter(colSize);

    for( int col = kernelHeight; col < colSize; col += kernelHeight )
    {
        maxAfter[col] = colInp[col * stride];
        maxBefore[col - 1] = colInp[(col - 1) * stride];
    }
    maxAfter[0] = colInp[0];
    maxBefore[colSize - 1] = colInp[(colSize - 1) * stride];

    for( int step = 0; step < colSize / kernelHeight; ++step )
    {
        for( int col = 1; col < kernelHeight; ++col )
        {
            maxAfter[step * kernelHeight + col] = std::max(colInp[(step * kernelHeight + col) * stride], maxAfter[step * kernelHeight + col - 1]);
            maxBefore[(step + 1) * kernelHeight - col - 1] = 
                std::max(colInp[((step + 1) * kernelHeight - col - 1) * stride], maxBefore[(step + 1) * kernelHeight - col]);
        }
    }

    const int wholeKernelsSize = kernelHeight * (colSize / kernelHeight);
    for( int col = wholeKernelsSize; col < colSize - 1; ++col )
    {
        maxAfter[col + 1] = std::max(colInp[(col + 1) * stride], maxAfter[col]);

        int curIndex = colSize - col + wholeKernelsSize - 1;
        maxBefore[curIndex - 1] = std::max(colInp[(curIndex - 1) * stride], maxBefore[curIndex]);
    }

    const int kernelHeightHalf = kernelHeight / 2;

    const int kernelHeightHalfEnd = std::min(kernelHeightHalf, colSize - 1);
    for( int col = 0; col < kernelHeightHalfEnd; ++col )
    {
        int index = std::min(col + kernelHeightHalf, colSize - 1);
        colOut[col * stride] = maxAfter[index];
    }

    for( int col = kernelHeightHalf; col < colSize - kernelHeightHalf; ++col ) {
        colOut[col * stride] = std::max(maxBefore[col - kernelHeightHalf], maxAfter[col + kernelHeightHalf]);
    }

    const int lastPartStart = std::max(colSize - kernelHeightHalf, 0);
    for( int col = lastPartStart; col < colSize; ++col )
    {
        int indexBefore = std::max(0, col - kernelHeightHalf);
        int indexAfter = std::min(colSize - 1, col + kernelHeightHalf);
        if( indexBefore > wholeKernelsSize ) {
            colOut[col * stride] = maxBefore[indexBefore];
        }
        else {
            colOut[col * stride] = std::max(maxBefore[indexBefore], maxAfter[indexAfter]);
        }
    }

}

