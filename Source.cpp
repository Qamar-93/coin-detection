#include "coinDetectorHeader.h"


using namespace cv;
// Namespace where all the C++ OpenCV functionality resides.

using namespace std;

int main(int argc, char** argv)
{
	cout << "starting\n";
	vector<cv::String> fn;
	glob(".//T2//*.jpg", fn, false);


	vector<Mat> images;
	size_t count = fn.size(); 
	for (size_t i = 0; i < count; i++) {
		Mat src = imread(fn[i]);
		images.push_back(imread(fn[i]));
	}
	imshow("q", images[0]);

	for (int i = 3; i < 4; i++){

		CoinDetector temp = CoinDetector(images[i]);

		Mat processedImg = images[i].clone();

		temp.preProcessing(&processedImg);


		std::vector<cv::Vec3f> circles = temp.computeHoughCircles(processedImg);

		imshow("qq", temp.input_img);
		//temp.cropAndSave(&circles, &processedImg, i);

		temp.drawCircles(&circles, i);


	}

	waitKey(0);
	return 0;
}