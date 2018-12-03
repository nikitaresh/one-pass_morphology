
#include <one_pass_morphology.h>
#include <gtest/gtest.h>
#include <opencv2/opencv.hpp>
#include <cstdlib>
#include <functional>

typedef std::function<bool(const cv::Mat1b&, cv::Mat1b&, const cv::Size&)> DilateFunction;

class OnePassDilateTest : public ::testing::TestWithParam<DilateFunction> {};


TEST_P(OnePassDilateTest, ArgumentsTest)
{
    const DilateFunction dailate = GetParam();

    cv::Mat1b image, result;
    cv::Size kernel(1, 1);

    // An empty image
    EXPECT_FALSE(dailate(image, result, kernel));

    image = cv::Mat1b(100, 100, 128);

    const int numOfCicles = 100;
    for( int index = 0; index < numOfCicles; ++index )
    {
        const int width = std::rand();
        const int height = std::rand();
        kernel = cv::Size(width, height);

        if( width % 2 == 0 || height % 2 == 0 ) {
            EXPECT_FALSE(dailate(image, result, kernel));
        }
        else {
            EXPECT_TRUE(dailate(image, result, kernel));
        }
    }

    result = cv::Mat1b(image.rows, image.cols);
    kernel = cv::Size(11, 11);
    // The result has the same size as the image
    EXPECT_TRUE(dailate(image, result, kernel));

    result = cv::Mat1b(image.rows + 1, image.cols + 1);
    // The result and the image have different sizes
    EXPECT_FALSE(dailate(image, result, kernel));
}


static bool isImagesEqual(const cv::Mat1b& image1, const cv::Mat1b& image2)
{
    if( image1.rows != image2.rows || image1.cols != image1.cols ) {
        return false;
    }

    cv::Mat cmpResult;
    cv::compare(image1, image2, cmpResult, cv::CMP_NE);
    const int numNonZero = cv::countNonZero(cmpResult);
    return numNonZero == 0;
}

TEST_P(OnePassDilateTest, HomogeneousImages)
{
    const DilateFunction dailate = GetParam();
    const cv::Size kernel(3, 3);

    for( int grayLevel = 0; grayLevel <= 255; grayLevel += 5 )
    {
        const cv::Mat1b image = cv::Mat1b(100, 100, grayLevel);

        cv::Mat1b result;
        EXPECT_TRUE(dailate(image, result, kernel));
        EXPECT_TRUE(isImagesEqual(image, result));
    }
}

TEST_P(OnePassDilateTest, SingleSplashImages)
{
    const DilateFunction dailate = GetParam();
    const cv::Size kernel(3, 3);

    const uchar backgroundValue = 128;
    const cv::Mat1b templateImage(100, 100, backgroundValue);

    const int numTests = 20;
    for( int index = 0; index < numTests; ++index )
    {
        cv::Mat1b image, result;
        image = templateImage.clone();

        const int x = std::rand() % image.cols;
        const int y = std::rand() % image.rows;
        image.at<uchar>(y, x) = backgroundValue / 2;

        EXPECT_TRUE(dailate(image, result, kernel));
        EXPECT_TRUE(isImagesEqual(templateImage, result));
    }
}

TEST_P(OnePassDilateTest, UseCaseTests)
{
    const DilateFunction dailate = GetParam();

    const std::string filePrefix = "../data/tests/";
    const cv::Size kernel(5, 5);

    const int numTests = 6;
    for( int index = 0; index < numTests; ++index )
    {
        const std::string strIndex = std::to_string(index + 1);
        std::cout << "UseCaseTest " << strIndex << std::endl;

        const std::string testPath = filePrefix + "test" + strIndex + ".bmp";
        const std::string answerPath = filePrefix + "answer" + strIndex + ".bmp";
        const cv::Mat1b testImage = cv::imread(testPath, cv::IMREAD_GRAYSCALE);
        EXPECT_FALSE(testImage.empty());
        const cv::Mat1b answerImage = cv::imread(answerPath, cv::IMREAD_GRAYSCALE);
        EXPECT_FALSE(answerImage.empty());

        cv::Mat1b tempAnswer;
        EXPECT_TRUE(dailate(testImage, tempAnswer, kernel));
        EXPECT_TRUE(isImagesEqual(tempAnswer, answerImage));
    }
}


INSTANTIATE_TEST_CASE_P(BothImplementations, OnePassDilateTest,
    ::testing::Values(OnePassMorphology::dilate, OnePassMorphology::dilateIntr));

int main( int argc, char *argv[] )
{
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
