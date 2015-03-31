/*
convert an image to relief style. Images are filtered by [1 -1] and [1; -1]
*/
#pragma once
#include "opencv2/opencv.hpp"

class relief
{
public:
	// level is an integer larger than 1
	bool doRelief(std::string imageSrc,std::string output, int level = 2);
};
