#include "opencv2/opencv.hpp"
#include <iostream>
#include <eigen3/Eigen/Eigen>
#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/core/eigen.hpp>
#include <chrono>
#include <iostream>
#include <fstream>

// #include <pcl/point_cloud.h>  
// #include <pcl/visualization/pcl_visualizer.h>  

using namespace cv;
using namespace std;
using namespace Eigen;


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
    cout << "Matches: " << matchesFlann.size() << endl;
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
    // printf("−− Max dist Brute: %f \n", max_dist_brute);
    // printf("−− Min dist Brute: %f \n", min_dist_brute);
    for(int i = 0; i < descriptors1.rows; i++) {
        if(matchesBrute[i].distance <= max(2 * min_dist_brute, 30.0)) {
            goodMatchBrute.push_back(matchesBrute[i]);
        }
    }
}

int main(int argc, char **argv) {

    Mat left = imread(argv[1], IMREAD_GRAYSCALE);
    Mat right = imread(argv[2], IMREAD_GRAYSCALE);

    // double fx = 718.856, fy = 718.856, cx = 607.1928, cy = 185.2157;
    // double b = 0.573; 
    //
    // generateDisparity(left, right);
    //
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

    // Mat outimg1;
    // drawKeypoints(left, keyPointVector1, outimg1, Scalar::all(-1), DrawMatchesFlags::DEFAULT);
    // imshow("ORB features 1", outimg1);
    //
    // Mat outimg2;
    // drawKeypoints(right, keyPointVector2, outimg2, Scalar::all(-1), DrawMatchesFlags::DEFAULT);
    // imshow("ORB features 2", outimg2);

    vector<DMatch> goodMatchBrute, goodMatchFlann;
    // getMatchesBrute(descriptors1, descriptors2, goodMatchBrute);
    getMatchesFlan(descriptors1, descriptors2, goodMatchFlann);
    cout << "Good Matches: " << goodMatchFlann.size() << endl;

    Mat img_goodmatchBrute, img_goodmatchFlann;
    // drawMatches(left, keyPointVector1, right, keyPointVector2, goodMatchBrute, img_goodmatchBrute);
    // imshow("good matches brute", img_goodmatchBrute);
    //
    // drawMatches(left, keyPointVector1, right, keyPointVector2, goodMatchFlann, img_goodmatchFlann);
    // imshow("good matches flann", img_goodmatchFlann);
    // cout << "vsbf" << endl;
    //
    // float left_projection_matrix[3][4] = {
	// 	{718.856,   0.00,       607.1928,   45.38225},
	// 	{0.00,      718.856,    185.2157,  -0.1130887},
	// 	{0.00,      0.00,       1.00,       0.003779761}
	// };

    float left_projection_matrix[3][4] = {
		{718.856,   0.00,       607.1928,   0},
		{0.00,      718.856,    185.2157,  0},
		{0.00,      0.00,       1.00,       0}
	};


	Mat lpm = cv::Mat(3, 4, CV_32F, left_projection_matrix);
    // cout << "lpm:\n" << lpm << endl;
	// float right_projection_matrix[3][4] = {
	// 	{718.856, 0.00, 607.1928, -337.2877},
	// 	{0.00, 718.856, 185.2157, 2.369057},
	// 	{0.00, 0.00, 1.00, 0.004915215}
	// };

    float right_projection_matrix[3][4] = {
		{718.856, 0.00, 607.1928, -386.1448},
		{0.00, 718.856, 185.2157, 0},
		{0.00, 0.00, 1.00, 0}
	};

    Mat rpm = cv::Mat(3, 4, CV_32F, right_projection_matrix);

    Mat left1 = imread("testing/images/left01.png", IMREAD_GRAYSCALE);
    vector<KeyPoint> keyPointVector3;
    vector<Point2d> matchedKeyPointVector;
    Mat descriptors3;

    detector->detect(left1, keyPointVector3);
    descriptor->compute(left1, keyPointVector3, descriptors3);
    vector<DMatch> goodMatchFlann1;
    getMatchesFlan(descriptors1, descriptors3, goodMatchFlann1);
    // cout << "Left 0-1 Matches: " << goodMatchFlann1.size() << endl;

    Mat R, t;
    poseEstimation2d2d(keyPointVector1, keyPointVector3, goodMatchFlann1, R, t);
    cout << "=============2D 2D===============" << endl;
    cout << "R:" << endl;
    cout << R << endl;
    cout << "T:" << endl;
    cout << t << endl;
    vector<Point3d> Hom3D;
	// cv::Mat pnts3D(1,keyPointVector1Matched.size(),CV_64FC4);
    Mat pnts3D;

    vector<Point2d> pt1, pt2;
    vector<KeyPoint> tempKeyPoints;

    for(int i = 0; i < (int)goodMatchFlann.size(); i++) {
            pt1.push_back(keyPointVector1[goodMatchFlann[i].queryIdx].pt);
            pt2.push_back(keyPointVector2[goodMatchFlann[i].trainIdx].pt);
            tempKeyPoints.push_back(keyPointVector1[goodMatchFlann[i].queryIdx]);
    }
    // Mat outimgTemp;
    // drawKeypoints(left, tempKeyPoints, outimgTemp, Scalar::all(-1), DrawMatchesFlags::DEFAULT);
    // imshow("ORB features 1", outimgTemp);

    // triangulate(keyPointVector1, keyPointVector2, goodMatchFlann, R, t);
    triangulatePoints(lpm, rpm, pt1, pt2, pnts3D);
    // cout << "3D Points: " << pnts3D.size() << endl;
    
    for(int j=0; j<pnts3D.cols;j++)
    {
        // cout << "Old: " << pnts3D.col(j) << endl;
        pnts3D.at<double>(0,j)=pnts3D.at<double>(0,j)/pnts3D.at<double>(3,j);
        pnts3D.at<double>(1,j)=pnts3D.at<double>(1,j)/pnts3D.at<double>(3,j);
        pnts3D.at<double>(2,j)=pnts3D.at<double>(2,j)/pnts3D.at<double>(3,j);
        // cout << "New: " << pnts3D.col(j) << endl;
    }

    ofstream file;
    file.open("points.csv");

    for(int i = 0; i < pnts3D.cols; i++) {
        // cout << pnts3D.col(i) << endl;
        file << pnts3D.at<double>(0,i) << "," << pnts3D.at<double>(1,i) << "," << pnts3D.at<double>(2,i) << "\n";
    }
    file.close();
    
    Eigen::Matrix<double, 4, 4> H;
    H << 1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1;

    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> pnts3DEigen;
    cv2eigen(pnts3D, pnts3DEigen);

    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> transPnts3dEigen = H * pnts3DEigen;

    Mat transPnts3D;
    eigen2cv(transPnts3dEigen, transPnts3D);



    // cout << transPnts3D.size() << endl;

    
    Mat matched3Dpnts;
    // vector<DMatch> tempMatches;
    bool flag = true;
    for(int i = 0; i < goodMatchFlann.size(); i++) {
        for(int j = 0; j < goodMatchFlann1.size(); j++) {
            if(keyPointVector1[goodMatchFlann[i].queryIdx].pt.x == keyPointVector1[goodMatchFlann1[j].queryIdx].pt.x &&
                keyPointVector1[goodMatchFlann[i].queryIdx].pt.y == keyPointVector1[goodMatchFlann1[j].queryIdx].pt.y) {
                    matchedKeyPointVector.push_back(keyPointVector3[goodMatchFlann1[j].trainIdx].pt);
                    if(flag) {
                        transPnts3D(Range(0, transPnts3D.rows-1), Range(i, i+1)).copyTo(matched3Dpnts);
                        flag = false;
                    } else {
                        Mat temp;
                        transPnts3D(Range(0, transPnts3D.rows-1), Range(i, i+1)).copyTo(temp);
                        hconcat(matched3Dpnts, temp, matched3Dpnts);
                    }
            }
        }
    }
    Mat points2D = Mat(matchedKeyPointVector);
    // cout << points2D.size() << endl;
    // cout << matched3Dpnts.t().size() << endl;
    Mat cameraMatrix = (Mat_<double>(3,3) << 520.9, 0, 325.1, 0, 521, 249.7, 0, 0, 1);
    Mat Rvec, t1;
    Mat DistorMat, rotMat, c;
    Mat D3 = matched3Dpnts.t();
    cout << D3.size() << endl;
    Mat D2 = points2D;
    Mat TDPoints,keymatches_img3,m,n;
    int fflag = 0;
    for(int i=0; i<D3.rows;i++)
    { 
        // cout<<"Z 3D POINTS: "<<D3.at<float>(i,2)<<endl;
        if(abs(D3.at<float>(i,0))<100000 && abs(D3.at<float>(i,1))<100000 && abs(D3.at<float>(i,2))<100000 && D3.at<float>(i,2)>0)
        {  

            if(fflag==0)
            {
                D3(Range(i,i+1),Range(0,D3.cols)).copyTo(TDPoints);
                D2(Range(i,i+1),Range(0,D2.cols)).copyTo(keymatches_img3);
                fflag=1;
            }
            else{

                    D3(Range(i,i+1),Range(0,D3.cols)).copyTo(m);
                    vconcat(TDPoints,m,TDPoints);
                    D2(Range(i,i+1),Range(0,D2.cols)).copyTo(n);
                    vconcat(keymatches_img3,n,keymatches_img3);

            }
            //keypcont.push_back(keypointsfwd[i]);
        
        }
    }
    // cout << "gfhdgnfxm" <<endl;
    // cout << lpm.size() << endl;
    // cout << lpm << endl;
    // lpm.convertTo(lpm, CV_64FC1);
    // for(int i = 0; i < lpm.rows; i++) {
    //     for(int j = 0; j < lpm.cols; j++) {
    //         cout << lpm.at<double>(i,j) << "; ";
    //     }
    //     cout << endl;
    // }
    decomposeProjectionMatrix(lpm, c, rotMat, t1);
    // cout << c << "\n\n" << rotMat << "\n\n" << t << endl;
    // cout << c << endl;

    Mat dist_coeffs = Mat::zeros(4,1,CV_64F);
    Mat abcd = TDPoints;
    cout << abcd.rows << endl;
    cout << keymatches_img3.rows << endl;
    
    solvePnP(abcd, keymatches_img3, c, dist_coeffs, Rvec, t1);

    // cout << Rvec << endl;
    // cout << t << endl;
    Mat R1;
    Rodrigues(Rvec, R1);
    // cout << R << endl;

    cout << "=============3D 2D===============" << endl;
    cout << "R:" << endl;
    cout << R1 << endl;
    cout << "T:" << endl;
    cout << t1 << endl;


    // Mat temp;
    // eigen2cv(H, temp);
    // cout << temp(Range(0,2DPo), Range(0, 2)) << endl;
    // Mat img_gooradmatchFlann1;
    // drawMatches(left, keyPointVector1, left1, keyPointVector3, goodMatchFlann1, img_goodmatchFlann1);
    // imshow("good matches flann", img_goodmatchFlann1);
    // cout << "vsbf" << endl;

    waitKey(0);
    return 0;
}