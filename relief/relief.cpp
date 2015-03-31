#include "relief.h"
#include "opencv2/opencv.hpp"
using namespace cv;
bool relief::doRelief(std::string imageSrc,std::string output, int level)
{
	Mat src = imread(imageSrc);
	Mat m1 = (Mat_<char>(2, 2) << level, 0, 0, -1*level);
	Mat m2 = (Mat_<char>(2, 2) << -1*level, 0, 0, level);

	Mat dst1, dst2;
	cv::filter2D(src, dst1, src.depth(), m1, Point(0, 0));
	cv::filter2D(src, dst2, src.depth(), m2, Point(0, 0));
	dst1 += 128;
	dst2 += 128;
	cvtColor(dst1,dst1, CV_RGB2GRAY);
	dst1 = dst1 + 56;
	//dst1.convertTo(dst1,CV_8UC1);
	imwrite(output, dst1);
	return 1;
}