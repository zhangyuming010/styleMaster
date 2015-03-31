/*
convert an image to stick style, based on edge detection

*/
#pragma once
#include "opencv2/opencv.hpp"

class styleStick
{
public:
	// level is set to 0-1
	bool Stick(std::string imageSrc,std::string output,float level = 0.5);

private:
	void sharpen(const cv::Mat& img, cv::Mat& result);
};

