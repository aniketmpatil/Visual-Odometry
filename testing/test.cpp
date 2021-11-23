#include "opencv2/opencv.hpp"
#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <eigen3/Eigen/Eigen>
#include <chrono>

using namespace cv;
using namespace std;

void poseEstimation2d2d(vector<KeyPoint> features1, vector<KeyPoint> features2, vector<DMatch> matches, Mat &R, Mat &t) {
    
    vector<Point2f> pt1, pt2;

    for(int i = 0; i < (int)matches.size(); i++) {
            pt1.push_back(features1[matches[i].queryIdx].pt);
            pt2.push_back(features2[matches[i].trainIdx].pt);
    }

    Mat fundmentalMat;
    fundmentalMat = findFundamentalMat(pt1, pt2, cv::FM_8POINT);
    cout << "Fundamental Matrix : " << fundmentalMat << endl;

    Point2d principalPt(325.1, 249.7);
    double focal_length = 521;
    Mat essentialMatrix;
    essentialMatrix = findEssentialMat(pt1, pt2, focal_length, principalPt);
    cout << "Essential Matrix : " << essentialMatrix << endl;

    recoverPose(essentialMatrix, pt1, pt2, R, t, focal_length, principalPt);
    cout << "R: " << R << endl;
    cout << "t: " << t << endl;

}

void generateDisparity(Mat left, Mat right) {
    int blockSize = 9;
    Ptr<StereoSGBM> sgbm = StereoSGBM::create(0, 96, blockSize, 8*blockSize*blockSize, 32*blockSize*blockSize, 1, 63, 10, 100, 2);
    Mat disparity_sgbm, disparity;
    sgbm -> compute(left, right, disparity_sgbm);
    disparity_sgbm.convertTo(disparity, CV_32F, 1.0 / 16.0f);
    imshow("Disparity", disparity / 96.0);
    waitKey(0);
}

void getMatchesFlan(Mat descriptors1, Mat descriptors2, vector<DMatch> &goodMatchFlann) {

    vector<DMatch> matchesFlann;
    FlannBasedMatcher matcherFlann = FlannBasedMatcher(makePtr<flann::LshIndexParams>(12, 20, 2));
    matcherFlann.match(descriptors1, descriptors2, matchesFlann);
    auto min_maxFlann = minmax_element(matchesFlann.begin(), matchesFlann.end(), [](const DMatch &m1, const DMatch &m2) { return m1.distance < m2.distance; });
    double min_dist_flann = min_maxFlann.first->distance;
    double max_dist_flann = min_maxFlann.second->distance;
    for(int i = 0; i < matchesFlann.size(); i++) {
        if(matchesFlann[i].distance <= max(2 * min_dist_flann, 30.0)) {
            goodMatchFlann.push_back(matchesFlann[i]);
        }
    }
    // vector<DMatch> matchesKnnFlann;
    // matcherFlann.knnMatch(descriptors1, descriptors2, matchesKnnFlann, 1);
}

void getMatchesBrute(Mat descriptors1, Mat descriptors2, vector<DMatch> &goodMatchBrute) {

    vector<DMatch> matchesBrute;
    Ptr<DescriptorMatcher> matcherBrute = DescriptorMatcher::create("BruteForce-Hamming");    
    matcherBrute->match(descriptors1, descriptors2, matchesBrute);
    auto min_maxBrute = minmax_element(matchesBrute.begin(), matchesBrute.end(), [](const DMatch &m1, const DMatch &m2) { return m1.distance < m2.distance; });
    double min_dist_brute = min_maxBrute.first->distance;
    double max_dist_brute = min_maxBrute.second->distance;
    printf("−− Max dist Brute: %f \n", max_dist_brute);
    printf("−− Min dist Brute: %f \n", min_dist_brute);
    for(int i = 0; i < descriptors1.rows; i++) {
        if(matchesBrute[i].distance <= max(2 * min_dist_brute, 30.0)) {
            goodMatchBrute.push_back(matchesBrute[i]);
        }
    }
}

void triangulate(vector<KeyPoint> KeyPoints1, vector<KeyPoint> KeyPoints2, vector<DMatch> matches, Mat R, Mat t) {

    Mat T1 = (Mat_<float>(3, 4) <<
        1, 0, 0, 0,
        0, 1, 0, 0,
        0, 0, 1, 0
    );

    Mat T2 = (Mat_<float>(3, 4) <<
        R.at<double>(0, 0), R.at<double>(0, 1), R.at<double>(0, 2), t.at<double>(0, 0),
        R.at<double>(1, 0), R.at<double>(1, 1), R.at<double>(1, 2), t.at<double>(1, 0),
        R.at<double>(2, 0), R.at<double>(2, 1), R.at<double>(2, 2), t.at<double>(2, 0)
    );

    Mat K = (Mat_<double>(3,3) << 520.9, 0, 325.1, 0, 521, 249.7, 0, 0, 1);
    vector<Point2f> pts1, pts2;
    for(DMatch m: matches) {
        // pts1.push_back(pixel2cam())
    }
}

int main(int argc, char **argv) {

    Mat left = imread(argv[1]);
    Mat right = imread(argv[2]);

    // cout << left.size() << endl;
    // cout << right.size() << endl;
    // Rect crop(0,0,right.cols,right.rows);
    // left = left(crop);

    // resize(left, left, Size(left.cols/4, left.rows/4)); 
    // resize(right, right, Size(right.cols/4, right.rows/4));

    // double fx = 718.856, fy = 718.856, cx = 607.1928, cy = 185.2157;
    // double b = 0.573;
    
    generateDisparity(left, right);
    
    // pcl::PointCloud<PointXYZRGB> pc;
    // for(int i = 0; i < left.rows; i++) {
    //     for(int j = 0; j < left.cols; j++) {
    //         if(disparity.at<float>(i, j) <= 10.0 || disparity.at<float>(i, j) >= 96.0) {
    //             continue;
    //         }
    //         PointXYZRGB pt;
    //         pt.rgb = left.at<uchar>(i, j) / 255.0;
    //         // Eigen::Vector4d pt(0, 0, 0, left.at<uchar>(i, j) / 255.0);
    //         double x = (i - cx) / fx;
    //         double y = (j - cy) / fy;
    //         double depth = fx * b / (disparity.at<float>(i, j));
    //         pt.x = x * depth;
    //         pt.y = y *depth;
    //         pt.z = depth;
    //         pc.push_back(pt);
    //     }
    // }

    vector<KeyPoint> keyPointVector1, keyPointVector2;
    Mat descriptors1,descriptors2;
    Ptr<FeatureDetector> detector = ORB::create();
    Ptr<DescriptorExtractor> descriptor = ORB::create();
    detector->detect(left, keyPointVector1);
    detector->detect(right, keyPointVector2);
    descriptor->compute(left, keyPointVector1, descriptors1);
    descriptor->compute(right, keyPointVector2, descriptors2);

    Mat outimg1;
    drawKeypoints(left, keyPointVector1, outimg1, Scalar::all(-1), DrawMatchesFlags::DEFAULT);
    imshow("ORB features 1", outimg1);

    Mat outimg2;
    drawKeypoints(right, keyPointVector2, outimg2, Scalar::all(-1), DrawMatchesFlags::DEFAULT);
    imshow("ORB features 2", outimg2);

    vector<DMatch> goodMatchBrute, goodMatchFlann;
    getMatchesBrute(descriptors1, descriptors2, goodMatchBrute);
    getMatchesFlan(descriptors1, descriptors2, goodMatchFlann);

    Mat img_goodmatchBrute, img_goodmatchFlann;
    drawMatches(left, keyPointVector1, right, keyPointVector2, goodMatchBrute, img_goodmatchBrute);
    imshow("good matches brute", img_goodmatchBrute);

    drawMatches(left, keyPointVector1, right, keyPointVector2, goodMatchFlann, img_goodmatchFlann);
    imshow("good matches flann", img_goodmatchFlann);
    cout << "vsbf" << endl;

    Mat R, t;
    poseEstimation2d2d(keyPointVector1, keyPointVector2, goodMatchFlann, R, t);


    triangulate(keyPointVector1, keyPointVector2, goodMatchFlann, R, t);


    waitKey(0);
    return 0;
}