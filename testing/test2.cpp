#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#define OPENCV_TRAITS_ENABLE_DEPRECATED
using namespace std;
using namespace cv;

int main(int argc, char **argv){
	Mat left = imread("testing/images/left.jpeg");
    Mat right = imread("testing/images/right.jpeg");
    vector<KeyPoint> keyPointVector1, keyPointVector2;
    Mat descriptors1,descriptors2;
    Ptr<FeatureDetector> detector = ORB::create();
    Ptr<DescriptorExtractor> descriptor = ORB::create();
    Ptr<DescriptorMatcher> matcherBrute = DescriptorMatcher::create("BruteForce-Hamming");
    // Ptr<DescriptorMatcher> matcherFlann = DescriptorMatcher::create("FlannBased");
    // chrono::steady_clock::time_point t1 = chrono::steady_clock::now();
    detector->detect(left, keyPointVector1);
    detector->detect(right, keyPointVector2);

    descriptor->compute(left, keyPointVector1, descriptors1);
    descriptor->compute(right, keyPointVector2, descriptors2);
	vector<DMatch> matchesBrute;
	matcherBrute->match(descriptors1, descriptors2, matchesBrute);

	vector<Point2f> keyPointVector1Matched, keyPointVector2Matched;

	for(int i = 0; i < matchesBrute.size(); i++) {
		int idx1 = matchesBrute[i].trainIdx; 
    	int idx2 = matchesBrute[i].queryIdx;
		keyPointVector1Matched.push_back(keyPointVector1[idx1].pt);
		keyPointVector2Matched.push_back(keyPointVector2[idx2].pt);
	}
// For sequence 1 VO dataset
	float left_projection_matrix[3][4] = {
		{718.856, 0.00, 607.1928, 45.38225},
		{0.00, 718.856, 185.2157, -0.1130887},
		{0.00, 0.00, 1.00, 0.003779761}
	};

	float right_projection_matrix[3][4] = {
		{718.856, 0.00, 607.1928, -337.2877},
		{0.00, 718.856, 185.2157, 2.369057},
		{0.00, 0.00, 1.00, 0.004915215}
	};

	Mat lpm = cv::Mat(3, 4, CV_64FC2, left_projection_matrix);

	Mat rpm = cv::Mat(3, 4, CV_64FC2, right_projection_matrix);


	// vector<Point3d> Hom3D;
	cv::Mat pnts3D;
	Mat nonhom3D;
	Mat K = (Mat_<double>(3,3) << 520.9, 0, 325.1, 0, 521, 249.7, 0, 0, 1);
	cv::triangulatePoints(lpm, rpm, keyPointVector1Matched, keyPointVector2Matched, pnts3D);
	cout << pnts3D.size << endl;
	Mat temp = pnts3D.t();
	cout << temp.size << endl;
	imshow("Test", temp);
	// for(int i = 0; i < pnts3D.cols; i++) {
	// 	cout << pnts3D.col(i) << endl;
	// }
	// pnts3D = pnts3D.reshape(4, 1);
	// cout << pnts3D.size << endl;
	
	// convertPointsFromHomogeneous(pnts3D, nonhom3D);
	// for (int j=0; j<pnts3D.cols; j++){
	// 	pnts3D.at<float>(0,j) = pnts3D.at<float>(0,j)/pnts3D.at<float>(3,j);
	// 	pnts3D.at<float>(1,j) = pnts3D.at<float>(1,j)/pnts3D.at<float>(3,j);
	// 	pnts3D.at<float>(2,j) = pnts3D.at<float>(2,j)/pnts3D.at<float>(3,j);
	// }

	// cout << nonhom3D.size << endl;
	// nonhom3D = nonhom3D.reshape(1, 3);
	// cout << nonhom3D.size << endl;
	// cout << nonhom3D.rows << endl;
	// cout << nonhom3D.cols << endl;
	// for(int i = 0; i < nonhom3D.cols; i++) {
	// 	cout << nonhom3D.col(i) << endl;
	// }
	
	waitKey(0);
}
