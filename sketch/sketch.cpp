#include "sketch.h"


bool StyleSumiao::MyLoadImg(string imageSrc, string imageOut, int r, float parameter)
{
	
	originalImageGray = imread(imageSrc, CV_LOAD_IMAGE_GRAYSCALE);  //Ä¬ÈÏÎª8bit´æ´¢
	
	if( originalImageGray.empty() )
	{
		cout << "Cannot read image file:" << imageSrc << endl;
		return false;
	}
	InverseColor(reverseOriginalImage);
	DFTImage(reverseOriginalImage, dftReverseImage);
	DFTShift(dftReverseImage);
	GaussianFilter(r);
	InverseDFTShift(invDftReverseImage);
	MixImages(parameter);
	resultImageMix.convertTo(resultImageMix,CV_8UC1,255,0);
	//equalizeHist(resultImageMix, resultImageMix);
	imwrite(imageOut, resultImageMix);
	//imshow("resultImage", resultImageMix);
	//waitKey(0);
	return true;
}


bool StyleSumiao::InverseColor(Mat &reverseOriginalImage)
{
	Mat image255;
	image255.create(Size(GetWidth(), GetHeight()), CV_8UC1);
	image255 = 255;
	reverseOriginalImage = image255 - originalImageGray;
	return true;
}

bool StyleSumiao::DFTImage(Mat reverseOriginalImageTemp, Mat &dftReverseImage)
{
	Mat padded;                         
	reverseOriginalImage.copyTo(padded);
	planes[0] = Mat_<float>(padded);
	planes[1] = Mat::zeros(padded.size(), CV_32FC1);
	merge(planes, 2, dftReverseImage);   
	dft(dftReverseImage, dftReverseImage); 

	return true;
}

bool StyleSumiao::DFTShift(Mat dftReverseImage)
{
	split(dftReverseImage, planes);          // planes[0] = Re(DFT(I), planes[1] = Im(DFT(I))
	int halfWidth = GetWidth()/2;
	int halfHeight = GetHeight()/2;
	//real part
	Mat q0(planes[0], Rect(0, 0, halfWidth, halfHeight));   // Top-Left - Create a ROI per quadrant
	Mat q1(planes[0], Rect(halfWidth, 0, halfWidth, halfHeight));  // Top-Right
	Mat q2(planes[0], Rect(0, halfHeight, halfWidth, halfHeight));  // Bottom-Left
	Mat q3(planes[0], Rect(halfWidth, halfHeight, halfWidth, halfHeight)); // Bottom-Right
	// swap quadrants (Top-Left with Bottom-Right)
	Mat tmp;                           
	q0.copyTo(tmp);
	q3.copyTo(q0);
	tmp.copyTo(q3);
	 // swap quadrant (Top-Right with Bottom-Left)
	q1.copyTo(tmp);                   
	q2.copyTo(q1);
	tmp.copyTo(q2);
	//imaginary part
	Mat q_1_0(planes[1], Rect(0, 0, halfWidth, halfHeight));   // Top-Left - Create a ROI per quadrant
	Mat q_1_1(planes[1], Rect(halfWidth, 0, halfWidth, halfHeight));  // Top-Right
	Mat q_1_2(planes[1], Rect(0, halfHeight, halfWidth, halfHeight));  // Bottom-Left
	Mat q_1_3(planes[1], Rect(halfWidth, halfHeight, halfWidth, halfHeight)); // Bottom-Right
	// swap quadrants (Top-Left with Bottom-Right)
	q_1_0.copyTo(tmp);
	q_1_3.copyTo(q_1_0);
	tmp.copyTo(q_1_3);
	// swap quadrant (Top-Right with Bottom-Left)
	q_1_1.copyTo(tmp);                    
	q_1_2.copyTo(q_1_1);
	tmp.copyTo(q_1_2);
	return true;
}

bool StyleSumiao::GaussianFilter(int r)
{
	float tempD;
	Mat tempH;
	int tempWidth = GetWidth(), tempHeight = GetHeight();
	tempH.create(Size(tempWidth, tempHeight), CV_32FC1);
	for ( int i=0; i<GetHeight(); ++i )
	{
		for ( int j=0; j<GetWidth(); ++j )
		{
			tempD = ((i-tempHeight/2)*(i-tempHeight/2)+(j-tempWidth/2)*(j-tempWidth/2));
			tempH.at<float>(i,j)=exp(-1*tempD/(2*r*r));
		}
	}
	planes[0]=planes[0].mul(tempH);
	planes[1]=planes[1].mul(tempH);
	return true;
}

bool StyleSumiao::InverseDFTShift(Mat &invDftReverseImage)
{
	int halfWidth = GetWidth()/2;
	int halfHeight = GetHeight()/2;
	Mat q0(planes[0], Rect(0, 0, halfWidth, halfHeight));   // Top-Left - Create a ROI per quadrant
	Mat q1(planes[0], Rect(halfWidth, 0, halfWidth, halfHeight));  // Top-Right
	Mat q2(planes[0], Rect(0, halfHeight, halfWidth, halfHeight));  // Bottom-Left
	Mat q3(planes[0], Rect(halfWidth, halfHeight, halfWidth, halfHeight)); // Bottom-Right
	Mat q_1_0(planes[1], Rect(0, 0, halfWidth, halfHeight));   // Top-Left - Create a ROI per quadrant
	Mat q_1_1(planes[1], Rect(halfWidth, 0, halfWidth, halfHeight));  // Top-Right
	Mat q_1_2(planes[1], Rect(0, halfHeight, halfWidth, halfHeight));  // Bottom-Left
	Mat q_1_3(planes[1], Rect(halfWidth, halfHeight, halfWidth, halfHeight)); // Bottom-Right
	Mat tmp;
	q0.copyTo(tmp);
	q3.copyTo(q0);
	tmp.copyTo(q3);
	 // swap quadrant (Top-Right with Bottom-Left)
	q1.copyTo(tmp);                   
	q2.copyTo(q1);
	tmp.copyTo(q2);
	q_1_0.copyTo(tmp);
	q_1_3.copyTo(q_1_0);
	tmp.copyTo(q_1_3);
	 // swap quadrant (Top-Right with Bottom-Left)
	q_1_1.copyTo(tmp);                   
	q_1_2.copyTo(q_1_1);
	tmp.copyTo(q_1_2);
	merge(planes, 2, invDftReverseImage);  
	dft(invDftReverseImage, invDftReverseImage, DFT_INVERSE+DFT_REAL_OUTPUT); //DFT_SCALE|DFT_INVERSE
	
	normalize(invDftReverseImage, invDftReverseImage, 0, 1, CV_MINMAX);
	
	return true;
}

bool StyleSumiao::MixImages(float parameter)
{
	Mat originalImageGrayFloat;
	float tempFloat;

	originalImageGray.convertTo(originalImageGrayFloat,CV_32FC1,1.0/255,0);
	resultImageMix.create(Size(GetWidth(), GetHeight()), CV_32FC1);
	for ( int i=0; i<GetHeight(); ++i )
	{
		for ( int j=0; j<GetWidth(); ++j )
		{

			tempFloat = originalImageGrayFloat.at<float>(i,j)/(parameter-invDftReverseImage.at<float>(i,j));
			if (tempFloat > 1)
			{
				tempFloat = 1;
			}
			resultImageMix.at<float>(i,j) = tempFloat;
		}
	}
	return true;
}