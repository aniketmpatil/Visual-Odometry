#include "opencv2/opencv.hpp"
#include <iostream>
#include <eigen3/Eigen/Eigen>
#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/core/eigen.hpp>
#include <chrono>

using namespace std;
using namespace cv;
using namespace Eigen;

void getMatchesFlan(Mat descriptors1, Mat descriptors2, vector<DMatch> &goodMatchFlann);

float left_projection_matrix[3][4] = {
    {718.856,   0.00,       607.1928,   0},
    {0.00,      718.856,    185.2157,  0},
    {0.00,      0.00,       1.00,       0}
};

float right_projection_matrix[3][4] = {
    {718.856, 0.00, 607.1928, -386.1448},
    {0.00, 718.856, 185.2157, 0},
    {0.00, 0.00, 1.00, 0}
};

Mat lpm = cv::Mat(3, 4, CV_32F, left_projection_matrix);
Mat rpm = cv::Mat(3, 4, CV_32F, right_projection_matrix);

int main(int argc, char** argv) {

    string left_images_path = "dataset/sequences/00/image_0/*.png";
    string right_images_path = "dataset/sequences/00/image_1/*.png";
    vector<string> left_images_names;
    vector<string> right_images_names;
    glob(left_images_path, left_images_names);
    glob(right_images_path, right_images_names);

    for(int i = 0; i < left_images_names.size(); i++) {
        Mat left_image = imread(left_images_names[i], IMREAD_GRAYSCALE);
        Mat right_image = imread(right_images_names[i], IMREAD_GRAYSCALE);
        // imshow("Left", left_image);
        // imshow("Right", right_image);

        vector<KeyPoint> keyPointVector1, keyPointVector2;
        Mat descriptors1,descriptors2;
        Ptr<FeatureDetector> detector = ORB::create();
        Ptr<DescriptorExtractor> descriptor = ORB::create();
        detector->detect(left_image, keyPointVector1);
        detector->detect(right_image, keyPointVector2);
        descriptor->compute(left_image, keyPointVector1, descriptors1);
        descriptor->compute(right_image, keyPointVector2, descriptors2);

        vector<DMatch> goodMatchFlann;
        getMatchesFlan(descriptors1, descriptors2, goodMatchFlann);
        Mat img_goodmatchFlann;
        drawMatches(left_image, keyPointVector1, right_image, keyPointVector2, goodMatchFlann, img_goodmatchFlann);
        imshow("good matches flann", img_goodmatchFlann);

        

        waitKey(33);
    }
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
}