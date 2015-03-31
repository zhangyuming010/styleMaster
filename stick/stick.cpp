#include "stick.h"
using namespace cv;
bool styleStick::Stick(std::string imageSrc,std::string output,float level)
{
	if (level <=0 || level >1)
	{
		return 0;

	}
	Mat _inputImg,_smoothImg,_sharpImg, _edgeImg;
	_inputImg = imread(imageSrc);
	cvtColor(_inputImg,_inputImg,CV_RGB2GRAY);
	int _row = _inputImg.rows;
	// Blur
	int _gaussWin = int(_row / 100 * level);
	if (_gaussWin%2 == 0)	
		_gaussWin = _gaussWin + 1;
	
	int _gaussSigma = int(_row / 120 * level);
	if (_gaussSigma < 2)
		_gaussSigma = 2;
	
	GaussianBlur(_inputImg, _smoothImg, Size(_gaussWin,_gaussWin), _gaussSigma);
	//Sharp
	sharpen(_smoothImg,_sharpImg);

	//Edge
	int thresh1 = int(100 * level);
	int thresh2 = int(200 * level);
	Canny(_smoothImg, _edgeImg, thresh1, thresh2);
	_edgeImg = 255 - _edgeImg;
	//threshold(_edgeImg, _edgeImg, 200, 1, THRESH_BINARY);
	imwrite(output, _edgeImg);
	
	return 1;
}

void styleStick::sharpen(const cv::Mat& img, cv::Mat& result)
{    
	result.create(img.size(), img.type());
	// first handle internal pixels
	for (int row = 1; row < img.rows-1; row++)
	{
		//previous row
		const uchar* previous = img.ptr<const uchar>(row-1);
		//current row
		const uchar* current = img.ptr<const uchar>(row);
		//next row
		const uchar* next = img.ptr<const uchar>(row+1);
		uchar *output = result.ptr<uchar>(row);
		int ch = img.channels();
		int starts = ch;
		int ends = (img.cols - 1) * ch;
		for (int col = starts; col < ends; col++)
		{
			
			*output++ = saturate_cast<uchar>(5 * current[col] - current[col-ch] - current[col+ch] - previous[col] - next[col]);
		}
	} //end loop
	// handle boundary
	result.row(0).setTo(Scalar::all(0));
	result.row(result.rows-1).setTo(Scalar::all(0));
	result.col(0).setTo(Scalar::all(0));
	result.col(result.cols-1).setTo(Scalar::all(0));
}