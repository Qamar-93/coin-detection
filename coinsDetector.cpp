#include "CoinDetectorHeader.h"

//	// constructor
CoinDetector::CoinDetector(cv::Mat src) {
	input_img = src.clone();
}


// Preprocessing function
void CoinDetector::preProcessing(cv::Mat *input_img)
{
	/// Convert it to gray
	cv::cvtColor(*input_img, *input_img, CV_BGR2GRAY);

	/// Reduce the noise so we avoid false circle detection
	GaussianBlur(*input_img, *input_img, cv::Size(11, 11), 2, 2);

}

// Computing the Hough circles
std::vector<cv::Vec3f> CoinDetector::computeHoughCircles(cv::Mat src) {

	std::vector<cv::Vec3f> circles, finalCircles;
	int threshold_value = 0;
	cv::Scalar threshold = cv::mean(src);
	float thresholdVal = threshold.val[0];
	size_t s = 0;
	int minRadius = cv::min(src.rows, src.cols);

	for (int i = 1; i*1.75 < (minRadius - 1); i += 2) {
			/// Apply the Hough Transform to find the circles
		HoughCircles(src, circles, CV_HOUGH_GRADIENT, 1, i, thresholdVal * 0.5, 60, i, i*1.75);
		/// get the highest number of cycles
		std::cout << "Computing\n";
		if (circles.size() > s) {
			s = circles.size();
			finalCircles = circles;
		}
	}

	return finalCircles;
}

// Cropping the detected coins and saving them
void CoinDetector::cropAndSave(std::vector<cv::Vec3f> *circles, cv::Mat *input_img, int inputImg_idx){
	std::vector<cv::Vec3f> tempCircles = *circles;
	cv::Mat src = *input_img;

	for (int i = 0; i < tempCircles.size(); i++)
	{
		cv::Point center(cvRound(tempCircles[i][0]), cvRound(tempCircles[i][1]));
		int radius = cvRound(tempCircles[i][2]);

		const cv::Vec3f& circ = tempCircles[i];
		// Draw the mask: white circle on black background
		cv::Mat1b mask(src.size(), uchar(0));
		circle(mask, cv::Point(circ[0], circ[1]), circ[2], cv::Scalar(255), CV_FILLED);

		// Compute the bounding box
		cv::Rect myROI(abs(circ[0] - circ[2]), abs(circ[1] - circ[2]), 2 * radius, 2 * radius);


		if (0 <= myROI.x
			&& 0 <= myROI.width
			&& myROI.x + myROI.width <= src.cols
			&& 0 <= myROI.y
			&& 0 <= myROI.height
			&& myROI.y + myROI.height <= src.rows) {
			// box within the image plane
			cv::Mat res;
			CoinDetector::input_img.copyTo(res, mask);
			res = res(myROI);

			cv::resize(res, res, cv::Size(150, 150));
			cv::String name = ".//coin-ml//validate//" + std::to_string(inputImg_idx) + std::to_string(i) + ".png";

			// Save the image
			std::cout << "Saving image\n";
			imwrite(name, res);
		}

	}
}

// Drawing the circles on the image
void CoinDetector::drawCircles(std::vector<cv::Vec3f> *circles, int idx) {
	std::vector<cv::Vec3f> tempCircles = *circles;
	for (size_t i = 0; i < tempCircles.size(); i++)
	{
		cv::Point center(cvRound(tempCircles[i][0]), cvRound(tempCircles[i][1]));
		int radius = cvRound(tempCircles[i][2]);
		circle(CoinDetector::input_img, center, radius, cv::Scalar(0, 0, 255), 3, 8, 0);

	}
	std::string name = "result_img_" + std::to_string(idx);
	cv::namedWindow(name, CV_WINDOW_NORMAL);
	cv::imshow(name, CoinDetector::input_img);
}
