#pragma once
#include <memory>
#include <iostream>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/stitching.hpp>
#include "opencv2/calib3d/calib3d.hpp"


#include <tuple>

class CoinDetector
{

public:

	// input image
	cv::Mat input_img;

	
public:

	// constructor 
	// input_img: image to be filtered

	CoinDetector(cv::Mat src);

	// Preprocessing function
	void preProcessing(cv::Mat *input_img);

	// Computing the Hough circles
	std::vector<cv::Vec3f> computeHoughCircles(cv::Mat src);

	// Cropping the detected coins and saving them
	void cropAndSave(std::vector<cv::Vec3f> *circles, cv::Mat *input_img, int inputImg_idx);

	// Drawing the circles on the image
	void drawCircles(std::vector<cv::Vec3f> *circles, int idx);

};