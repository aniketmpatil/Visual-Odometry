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
    Mat K = (Mat_<double>(3,3) << 520.9, 0, 325.1, 0, 521, 249.7, 0, 0, 1);
    vector<Point2f> pt1, pt2;

    for(int i = 0; i < (int)matches.size(); i++) {
            pt1.push_back(features1[matches[i].queryIdx].pt);
            pt2.push_back(features2[matches[i].trainIdx].pt);
    }

    Mat fundmentalMat;
    fundmentalMat = findFundamentalMat(pt1, pt2, cv::FM_8POINT);
    cout << fundmentalMat << endl;

    Point2d principalPt(325.1, 249.7);
    double focal_length = 521;
    Mat essentialMatrix;
    
}

int main(int argc, char **argv) {

    Mat left = imread("testing/images/left.jpeg");
    Mat right = imread("testing/images/right.jpeg");

    double fx = 718.856, fy = 718.856, cx = 607.1928, cy = 185.2157;
    double b = 0.573;

    Ptr<StereoSGBM> sgbm = StereoSGBM::create(0, 96, 9, 8*9*9, 32*9*9, 1, 63, 10, 100, 32);
    Mat disparity_sgbm, disparity;
    sgbm -> compute(left, right, disparity_sgbm);
    disparity_sgbm.convertTo(disparity, CV_32F, 1.0 / 16.0f);
    
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

    imshow("Disparity", disparity / 96.0);
    waitKey(0);

    // resize(left, left, Size(left.cols/2, left.rows/2));
    // resize(right, right, Size(right.cols/2, right.rows/2));

    vector<KeyPoint> keyPointVector1, keyPointVector2;
    Mat descriptors1,descriptors2;
    Ptr<FeatureDetector> detector = ORB::create();
    Ptr<DescriptorExtractor> descriptor = ORB::create();
    Ptr<DescriptorMatcher> matcherBrute = DescriptorMatcher::create("BruteForce-Hamming");
    FlannBasedMatcher   matcherFlann = FlannBasedMatcher(makePtr<flann::LshIndexParams>(12, 20, 2));
    chrono::steady_clock::time_point t1 = chrono::steady_clock::now();
    detector->detect(left, keyPointVector1);
    detector->detect(right, keyPointVector2);

    descriptor->compute(left, keyPointVector1, descriptors1);
    descriptor->compute(right, keyPointVector2, descriptors2);
    chrono::steady_clock::time_point t2 = chrono::steady_clock::now();
    chrono::duration<double> time_used = chrono::duration_cast<chrono::duration<double> > (t2 - t1);
    cout << "extract ORB cost = " << time_used.count() << " seconds. " << endl;

    Mat outimg1;
    drawKeypoints(left, keyPointVector1, outimg1, Scalar::all(-1), DrawMatchesFlags::DEFAULT);
    imshow("ORB features 1", outimg1);

    Mat outimg2;
    drawKeypoints(right, keyPointVector2, outimg2, Scalar::all(-1), DrawMatchesFlags::DEFAULT);
    imshow("ORB features 2", outimg2);

    vector<DMatch> matchesBrute, matchesFlann;
    t1 = chrono::steady_clock::now();
    matcherBrute->match(descriptors1, descriptors2, matchesBrute);
    t2 = chrono::steady_clock::now();
    time_used = chrono::duration_cast<chrono::duration<double> > (t2 - t1);
    cout << "Match Brute cost = " << time_used.count() << " seconds. " << endl;
    cout << "vsbf" << endl;
    vector<vector<DMatch>> matchesKnnFlann;
    // vector<DMatch> matchesKnnFlann;
    t1 = chrono::steady_clock::now();
    // matcherFlann.knnMatch(descriptors1, descriptors2, matchesKnnFlann, 1);
    matcherFlann.match(descriptors1, descriptors2, matchesFlann);

    t2 = chrono::steady_clock::now();
    time_used = chrono::duration_cast<chrono::duration<double> > (t2 - t1);
    cout << "Match Flann cost = " << time_used.count() << " seconds. " << endl;
    cout << "vsbf" << endl;
    auto min_maxBrute = minmax_element(matchesBrute.begin(), matchesBrute.end(), [](const DMatch &m1, const DMatch &m2) { return m1.distance < m2.distance; });
    double min_dist_brute = min_maxBrute.first->distance;
    double max_dist_brute = min_maxBrute.second->distance;

    printf("−− Max dist Brute: %f \n", max_dist_brute);
    printf("−− Min dist Brute: %f \n", min_dist_brute);

    auto min_maxFlann = minmax_element(matchesFlann.begin(), matchesFlann.end(), [](const DMatch &m1, const DMatch &m2) { return m1.distance < m2.distance; });
    double min_dist_flann = min_maxFlann.first->distance;
    double max_dist_flann = min_maxFlann.second->distance;

    // printf("−− Max dist Flann: %f \n", max_dist_flann);
    // printf("−− Min dist Flann: %f \n", min_dist_flann);

    vector<DMatch> goodMatchBrute, goodMatchFlann;

    for(int i = 0; i < descriptors1.rows; i++) {
        if(matchesBrute[i].distance <= max(2 * min_dist_brute, 30.0)) {
            goodMatchBrute.push_back(matchesBrute[i]);
        }
    }
    // cout << matchesKnnFlann.size() << endl;
    // size_t size = matchesKnnFlann.size();
    // for(auto i = 0; i < size; i++) {
    //     cout << "a";
    //     if(matchesKnnFlann[i][0].distance < 0.7f * matchesKnnFlann[i][1].distance) {
    //         cout << endl;
    //         cout << "a" << endl;
    //         goodMatchFlann.push_back(matchesKnnFlann[i][0]);
    //         cout << endl;
    //         cout << "a" << endl;
    //     }
    // }
    for(int i = 0; i < matchesFlann.size(); i++) {
        if(matchesFlann[i].distance <= max(2 * min_dist_flann, 30.0)) {
            goodMatchFlann.push_back(matchesFlann[i]);
        }
    }
    // for(auto i = 0; i < matchesKnnFlann.size() - 1; i++) {
    //     if(matchesKnnFlann[i].distance < 0.5f * matchesKnnFlann[i+1].distance) {
    //         goodMatchFlann.push_back(matchesKnnFlann[i]);
    //     }
    // }
    // cout << "vsbf" << endl;
    
    Mat img_matchBrute, img_matchFlann;
    Mat img_goodmatchBrute, img_goodmatchFlann;
    drawMatches(left, keyPointVector1, right, keyPointVector2, matchesBrute, img_matchBrute);
    drawMatches(left, keyPointVector1, right, keyPointVector2, goodMatchBrute, img_goodmatchBrute);
    imshow("all matches brute", img_matchBrute);
    imshow("good matches brute", img_goodmatchBrute);

    // drawMatches(left, keyPointVector1, right, keyPointVector2, matchesKnnFlann, img_matchFlann);
    drawMatches(left, keyPointVector1, right, keyPointVector2, goodMatchFlann, img_goodmatchFlann);
    // imshow("all matches flann", img_matchFlann);
    imshow("good matches flann", img_goodmatchFlann);
    cout << "vsbf" << endl;





    waitKey(0);
    return 0;
}