
#include <one_pass_morphology.h>
#include <chrono>

const int numCycles = 10;

static bool isImageEqual(const cv::Mat1b& img1, const cv::Mat1b& img2)
{
    cv::Mat1b lr = img1 - img2;
    cv::Mat1b rl = img2 - img1;
    int lrNonZero = cv::countNonZero(lr);
    int rlNonZero = cv::countNonZero(rl);
    return ( lrNonZero == 0) && ( rlNonZero == 0);
}

int main(int argc, char *argv[])
{
    if( argc < 2 ) {
        std::cout << "Usage: one_pass_morphology[.exe] image_path" << std::endl;
        return 1;
    }

    cv::Mat srcImage = cv::imread( argv[1] );
    cv::Mat1b grayImage;
    cv::cvtColor(srcImage, grayImage, cv::COLOR_BGR2GRAY );
    cv::Size kernel(301, 301);
    cv::Mat1b erodeImage;

    auto startOnePass = std::chrono::system_clock::now();
    for( int index = 0; index < numCycles; ++index ) {
        OnePassMorphology::dilate(grayImage, erodeImage, kernel);
    }
    auto endOnePass = std::chrono::system_clock::now();


    cv::Mat kernelElement = cv::getStructuringElement(cv::MORPH_RECT, kernel);
    cv::Mat1b erodeCVImage;

    auto startCV = std::chrono::system_clock::now();
    for( int index = 0; index < numCycles; ++index ) {
       cv::dilate(grayImage, erodeCVImage, kernelElement);
    }
    auto endCV = std::chrono::system_clock::now();

    bool isEqualREsult = isImageEqual(erodeImage, erodeCVImage);

    std::chrono::duration<double> durationOnePass = endOnePass - startOnePass;
    std::chrono::duration<double> durationCV = endCV - startCV;
    std::cout << "durationOnePass: " << durationOnePass.count() 
              << "  durationCV: " << durationCV.count() << std::endl;
    std::cout << "is equal : " << (isEqualREsult ? "true" : "FALSE") << std::endl;

    return 0;
}