
#include <one_pass_morphology.h>
#include <array>
#include <smmintrin.h>


bool OnePassMorphology::dilate(const cv::Mat1b& image, cv::Mat1b& result, const cv::Size& kernel)
{
    if( image.empty() ) {
        return false;
    }

    if( result.empty() ) {
        result = cv::Mat1b(image.rows, image.cols);
    }

    if( result.rows != image.rows || result.cols != image.cols ) {
        return false;
    }

    if( kernel.width % 2 == 0 || kernel.height % 2 == 0 ) {
        return false;
    }

    for( int rowIndex = 0; rowIndex < result.rows; ++rowIndex ) {
        dilateRow(image.ptr(rowIndex), result.ptr(rowIndex),
                  kernel.width, result.cols );
    }

    for(int colIndex = 0; colIndex < image.cols; ++colIndex) {
        dilateColumn(result.ptr() + colIndex, result.ptr() + colIndex,
                     kernel.height, image.rows, image.cols);
    }

    return true;
}


bool OnePassMorphology::dilateRow(const uchar* rowInp, uchar* rowOut,
                                  int kernelWidth, int rowSize)
{
    if( rowInp == nullptr || rowOut == nullptr ||
        kernelWidth % 2 == 0 || kernelWidth < 2 )
    {
        return false;
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
            maxAfter[step * kernelWidth + row] = 
                std::max(rowInp[step * kernelWidth + row],
                         maxAfter[step * kernelWidth + row - 1]);
            maxBefore[(step + 1) * kernelWidth - row - 1] = 
                std::max(rowInp[(step + 1) * kernelWidth - row - 1],
                         maxBefore[(step + 1) * kernelWidth - row]);
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
        rowOut[row] = std::max(maxBefore[row - kernelWidthHalf],
                               maxAfter[row + kernelWidthHalf]);
    }

    const int prevKernel = (wholeKernelsSize == rowSize) ? 
        (wholeKernelsSize - kernelWidth) : wholeKernelsSize;
    const int lastPartStart = std::max(rowSize - kernelWidthHalf, 0);
    for( int row = lastPartStart; row < rowSize; ++row )
    {
        int indexBefore = std::max(0, row - kernelWidthHalf);
        int indexAfter = std::min(rowSize - 1, row + kernelWidthHalf);
        if( indexBefore > prevKernel ) {
            rowOut[row] = maxBefore[indexBefore];
        }
        else {
            rowOut[row] = std::max(maxBefore[indexBefore], maxAfter[indexAfter]);
        }
    }

    return true;
}

bool OnePassMorphology::dilateColumn(const uchar* colInp, uchar* colOut,
                                     int kernelHeight, int colSize, int stride)
{
    if( colInp == nullptr || colOut == nullptr ||
        kernelHeight % 2 == 0 || kernelHeight < 2 )
    {
        return false;
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
            maxAfter[step * kernelHeight + col] = 
                std::max(colInp[(step * kernelHeight + col) * stride],
                         maxAfter[step * kernelHeight + col - 1]);
            maxBefore[(step + 1) * kernelHeight - col - 1] = 
                std::max(colInp[((step + 1) * kernelHeight - col - 1) * stride],
                         maxBefore[(step + 1) * kernelHeight - col]);
        }
    }

    const int wholeKernelsSize = kernelHeight * (colSize / kernelHeight);
    for( int col = wholeKernelsSize; col < colSize - 1; ++col )
    {
        maxAfter[col + 1] = std::max(colInp[(col + 1) * stride], maxAfter[col]);

        int curIndex = colSize - col + wholeKernelsSize - 1;
        maxBefore[curIndex - 1] =
            std::max(colInp[(curIndex - 1) * stride], maxBefore[curIndex]);
    }

    const int kernelHeightHalf = kernelHeight / 2;

    const int kernelHeightHalfEnd = std::min(kernelHeightHalf, colSize - 1);
    for( int col = 0; col < kernelHeightHalfEnd; ++col )
    {
        int index = std::min(col + kernelHeightHalf, colSize - 1);
        colOut[col * stride] = maxAfter[index];
    }

    for( int col = kernelHeightHalf; col < colSize - kernelHeightHalf; ++col ) {
        colOut[col * stride] = 
            std::max(maxBefore[col - kernelHeightHalf],
                     maxAfter[col + kernelHeightHalf]);
    }

    const int prevKernel = (wholeKernelsSize == colSize) ? 
        (wholeKernelsSize - kernelHeight) : wholeKernelsSize;
    const int lastPartStart = std::max(colSize - kernelHeightHalf, 0);
    for( int col = lastPartStart; col < colSize; ++col )
    {
        int indexBefore = std::max(0, col - kernelHeightHalf);
        int indexAfter = std::min(colSize - 1, col + kernelHeightHalf);
        if( indexBefore > prevKernel ) {
            colOut[col * stride] = maxBefore[indexBefore];
        }
        else {
            colOut[col * stride] =
                std::max( maxBefore[indexBefore], maxAfter[indexAfter] );
        }
    }

    return true;
}



// Intrinsics implementation
//////////////////////////////////////////////////////////////////////////

bool OnePassMorphology::dilateIntr(const cv::Mat1b& image, cv::Mat1b& result,
                                   const cv::Size& kernel)
{
    if( image.empty() ) {
        return false;
    }

    if( result.empty() ) {
        result = cv::Mat1b(image.rows, image.cols);
    }

    if( result.rows != image.rows || result.cols != image.cols ) {
        return false;
    }

    if( kernel.width % 2 == 0 || kernel.height % 2 == 0 ) {
        return false;
    }

    dilateColumnsIntr(image, result, kernel.height);

    // transpose result -> dilateColumnIntr() -> transpose result
    result = result.t();
    dilateColumnsIntr(result, result, kernel.width);
    result = result.t();

    return true;
}

bool OnePassMorphology::dilateColumnsIntr(const cv::Mat1b& image, cv::Mat1b& result,
                                          int kernelHeight)
{
    if( image.rows != result.rows || image.cols != result.cols ||
        image.step.p[0] != result.step.p[0] )
    {
        return false;
    }

    cv::Mat1b maxAfterMat(image.rows, image.cols);
    cv::Mat1b maxBeforeMat(image.rows, image.cols);

    const int imgWidth = image.cols;
    const int imgHeight = image.rows;
    const int stride = image.cols;
    const uchar* imagePtr = image.ptr();
    uchar* resultPtr = result.ptr();
    uchar* maxAfterPtr = maxAfterMat.ptr();
    uchar* maxBeforePtr = maxBeforeMat.ptr();

    for( int row = kernelHeight; row < imgHeight; row += kernelHeight )
    {
        memcpy( maxAfterPtr + row * stride, imagePtr + row * stride, stride );
        memcpy( maxBeforePtr + (row - 1) * stride,
                imagePtr + (row - 1) * stride, stride );
    }
    memcpy( maxAfterPtr, imagePtr, stride );
    memcpy( maxBeforePtr + (imgHeight - 1) * stride,
            imagePtr + (imgHeight - 1) * stride, stride );

    const int numKernelSteps = imgHeight / kernelHeight;
    const int numWidthSteps = imgWidth / 16;
    for( int kernelStep = 0; kernelStep < numKernelSteps; ++kernelStep )
    {
        for( int row = 1; row < kernelHeight; ++row )
        {
            const int dataShiftAfter = (kernelStep * kernelHeight + row) * stride;
            const int dataShiftBefore = ((kernelStep + 1) * kernelHeight - row) * stride;
            for( int col = 0; col < numWidthSteps; ++col )
            {
                __m128i imgDataAft = _mm_loadu_si128((const __m128i*)(imagePtr + dataShiftAfter + 16 * col));
                __m128i maxAftPreData = _mm_loadu_si128((const __m128i*)(maxAfterPtr + dataShiftAfter - stride + 16 * col));
                __m128i maxDataAfter = _mm_max_epu8( imgDataAft, maxAftPreData );
                _mm_storeu_si128((__m128i*)(maxAfterPtr + dataShiftAfter + 16 * col), maxDataAfter);

                __m128i imgDataBef = _mm_loadu_si128((const __m128i*)(imagePtr + dataShiftBefore - stride + 16 * col));
                __m128i maxBefNextData = _mm_loadu_si128((const __m128i*)(maxBeforePtr + dataShiftBefore + 16 * col));
                __m128i maxDataBef = _mm_max_epu8( imgDataBef, maxBefNextData );
                _mm_storeu_si128((__m128i*)(maxBeforePtr + dataShiftBefore - stride + 16 * col), maxDataBef);
            }
        }
    }

    const int wholeKernelsSize = kernelHeight * numKernelSteps;
    for( int row = wholeKernelsSize; row < imgHeight - 1; ++row )
    {
        const int dataShiftAfter = (row + 1) * stride;
        const int dataShiftBefore = (imgHeight - 1 + wholeKernelsSize - row) * stride;
        for( int col = 0; col < numWidthSteps; ++col )
        {
            __m128i imgDataAft = _mm_loadu_si128((const __m128i*)(imagePtr + dataShiftAfter + 16 * col));
            __m128i maxAftPreData = _mm_loadu_si128((const __m128i*)(maxAfterPtr + dataShiftAfter - stride + 16 * col));
            __m128i maxDataAfter = _mm_max_epu8( imgDataAft, maxAftPreData );
            _mm_storeu_si128((__m128i*)(maxAfterPtr + dataShiftAfter + 16 * col), maxDataAfter);

            __m128i imgDataBef = _mm_loadu_si128((const __m128i*)(imagePtr + dataShiftBefore - stride + 16 * col));
            __m128i maxBefNextData = _mm_loadu_si128((const __m128i*)(maxBeforePtr + dataShiftBefore + 16 * col));
            __m128i maxDataBef = _mm_max_epu8( imgDataBef, maxBefNextData );
            _mm_storeu_si128((__m128i*)(maxBeforePtr + dataShiftBefore - stride + 16 * col), maxDataBef);
        }
    }

    const int widthStepStart = 16 * numWidthSteps;
    for( int col = widthStepStart; col < imgWidth; ++col )
    {
        for( int kernelStep = 0; kernelStep < numKernelSteps; ++kernelStep )
        {
            for( int row = 1; row < kernelHeight; ++row )
            {
                const int dataShiftAfter = (kernelStep * kernelHeight + row) * stride;
                const int dataShiftBefore = ((kernelStep + 1) * kernelHeight - row) * stride;
                maxAfterPtr[dataShiftAfter + col] = 
                    std::max( imagePtr[dataShiftAfter + col], maxAfterPtr[dataShiftAfter - stride + col] );
                maxBeforePtr[dataShiftBefore - stride + col] = 
                    std::max( imagePtr[dataShiftBefore - stride + col], maxBeforePtr[dataShiftBefore + col] );
            }
        }

        for( int row = wholeKernelsSize; row < imgHeight - 1; ++row )
        {
            maxAfterPtr[(row + 1) * stride + col] = 
                std::max(imagePtr[(row + 1) * stride + col], maxAfterPtr[row * stride + col]);

            int curIndex = (imgHeight - row + wholeKernelsSize - 1) * stride;
            maxBeforePtr[curIndex - stride + col] = 
                std::max(imagePtr[(curIndex - stride) + col], maxBeforePtr[curIndex + col]);
        }
    }



    const int kernelHeightHalf = kernelHeight / 2;

    const int kernelHeightHalfEnd = std::min(kernelHeightHalf, imgHeight - 1);
    for( int row = 0; row < kernelHeightHalfEnd; ++row )
    {
        int index = std::min( row + kernelHeightHalf, imgHeight - 1 );
        memcpy( resultPtr + row * stride, maxAfterPtr + index * stride, stride );
    }

    for( int row = kernelHeightHalf; row < imgHeight - kernelHeightHalf; ++row )
    {
        for( int col = 0; col < numWidthSteps; ++col )
        {
            const uchar* beforePtr = maxBeforePtr + (row - kernelHeightHalf) * stride + 16 * col;
            const uchar* afterPtr = maxAfterPtr + (row + kernelHeightHalf) * stride + 16 * col;
            uchar* resPtr = resultPtr + row * stride + 16 * col;
            __m128i maxDataBef = _mm_loadu_si128((const __m128i*)(beforePtr));
            __m128i maxDataAft = _mm_loadu_si128((const __m128i*)(afterPtr));
            __m128i maxResData = _mm_max_epu8( maxDataBef, maxDataAft );
            _mm_storeu_si128((__m128i*)(resPtr), maxResData);
        }

        for( int col = widthStepStart; col < imgWidth; ++col )
        {
            resultPtr[row * stride + col] = 
                std::max(maxBeforePtr[(row - kernelHeightHalf) * stride + col], 
                         maxAfterPtr[(row + kernelHeightHalf) * stride + col]);
        }
    }

    const int prevKernel = (wholeKernelsSize == imgHeight) ? 
        (wholeKernelsSize - kernelHeight) : wholeKernelsSize;
    const int lastPartStart = std::max(imgHeight - kernelHeightHalf, 0);
    for( int row = lastPartStart; row < imgHeight; ++row )
    {
        int indexBefore = std::max(0, row - kernelHeightHalf);
        int indexAfter = std::min(imgHeight - 1, row + kernelHeightHalf);
        if( indexBefore > prevKernel ) {
            memcpy( resultPtr + row * stride, maxBeforePtr + indexBefore * stride, stride );
        }
        else
        {
            for( int col = 0; col < numWidthSteps; ++col )
            {
                const uchar* beforePtr = maxBeforePtr + indexBefore * stride + 16 * col;
                const uchar* afterPtr = maxAfterPtr + indexAfter * stride + 16 * col;
                uchar* resPtr = resultPtr + row * stride + 16 * col;
                __m128i maxDataBef = _mm_loadu_si128((const __m128i*)(beforePtr));
                __m128i maxDataAft = _mm_loadu_si128((const __m128i*)(afterPtr));
                __m128i maxResData = _mm_max_epu8( maxDataBef, maxDataAft );
                _mm_storeu_si128((__m128i*)(resPtr), maxResData);
            }

            for( int col = widthStepStart; col < imgWidth; ++col )
            {
                resultPtr[row * stride + col] = 
                    std::max( maxBeforePtr[indexBefore * stride + col], maxAfterPtr[indexAfter * stride + col] );
            }
        }
    }

    return true;
}

