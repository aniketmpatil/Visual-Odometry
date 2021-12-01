#include "opencv2/opencv.hpp"
#include <iostream>
#include <fstream>
#include <eigen3/Eigen/Eigen>
#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/core/eigen.hpp>
#include <chrono>

using namespace std;
using namespace cv;
using namespace Eigen;

void getMatchesFlan(Mat prev_left_descriptors, Mat prev_right_descriptors, vector<DMatch> &stereo_matches);
void getMatchesBrute(Mat descriptors1, Mat descriptors2, vector<DMatch> &goodMatchBrute);
Mat convertToHomogeneousMat(Mat R, Mat T);
vector<string> split (string s, string delimiter);
void getProjectionMatrices(string path);

Mat lpm = cv::Mat(3, 4, CV_32F);
Mat rpm = cv::Mat(3, 4, CV_32F);
Mat camera_matrix, rot_matrix, trans_vect, distortion_mat;

Mat trajectory = Mat(1000, 1000, CV_8UC3);

Mat prev_3d_points;
Eigen::Matrix<float, 4, 4> prev_H;

int main(int argc, char** argv) {

    prev_H << 1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1;
    string seq = argv[1];
    string path = "dataset/sequences/"+seq;
    string left_images_path = path+"/image_0/*.png";
    string right_images_path = path+"/image_1/*.png";
    getProjectionMatrices(path);
    waitKey(0);
    vector<string> left_images_names;
    vector<string> right_images_names;
    glob(left_images_path, left_images_names);
    glob(right_images_path, right_images_names);
    
    
    decomposeProjectionMatrix(lpm, camera_matrix, rot_matrix, trans_vect);
    distortion_mat = Mat::zeros(4,1,CV_64F);

    vector<KeyPoint> prev_left_keypoints, prev_right_keypoints, cur_left_keypoints;
    Mat prev_left_descriptors,prev_right_descriptors, cur_left_descriptors;
    Ptr<FeatureDetector> detector = ORB::create(1000);
    Ptr<DescriptorExtractor> descriptor = ORB::create(1000);

    for(int image_seq = 1; image_seq < left_images_names.size(); image_seq++) {
        Mat prev_left_image = imread(left_images_names[image_seq-1], IMREAD_GRAYSCALE);
        Mat prev_right_image = imread(right_images_names[image_seq-1], IMREAD_GRAYSCALE);
        Mat cur_left_image = imread(left_images_names[image_seq], IMREAD_GRAYSCALE);
        
        

        detector->detect(prev_left_image, prev_left_keypoints);
        detector->detect(prev_right_image, prev_right_keypoints);
        // detector->detect(cur_left_image, cur_left_keypoints);
        detector->detectAndCompute(prev_left_image, Mat(), prev_left_keypoints, prev_left_descriptors, true);
        detector->detectAndCompute(prev_right_image, Mat(), prev_right_keypoints, prev_right_descriptors, true);
        // descriptor->compute(cur_left_image, cur_left_keypoints, cur_left_descriptors);
        

        vector<DMatch> stereo_matches;
        getMatchesFlan(prev_left_descriptors, prev_right_descriptors, stereo_matches);
        // cout << "Matches : " << stereo_matches.size() << endl;
        // vector<DMatch> temporal_matches;
        // getMatchesFlan(prev_left_descriptors, cur_left_descriptors, temporal_matches);

        vector<Point2f> stereoPointL, stereoPointR, stereoPointL1;
        for(int i = 0; i < stereo_matches.size(); i++) {
            stereoPointL.push_back(prev_left_keypoints[stereo_matches[i].queryIdx].pt);
            stereoPointR.push_back(prev_right_keypoints[stereo_matches[i].trainIdx].pt);
        }
        // cout << stereoPointL.size() << ", " << stereoPointR.size() << endl;
        triangulatePoints(lpm, rpm, stereoPointL, stereoPointR, prev_3d_points);
        // cout << prev_3d_points.size() << endl;
        // cout << prev_3d_points.type() << endl;
        for(int i = 0; i < prev_3d_points.cols; i++) {
            prev_3d_points.at<float>(0,i) = prev_3d_points.at<float>(0,i) / prev_3d_points.at<float>(3,i);
            prev_3d_points.at<float>(1,i) = prev_3d_points.at<float>(1,i) / prev_3d_points.at<float>(3,i);
            prev_3d_points.at<float>(2,i) = prev_3d_points.at<float>(2,i) / prev_3d_points.at<float>(3,i);
        }

        // cout << lpm << endl << rpm << endl;
        // cout << prev_3d_points.col(0) << endl;
        
        Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic> eigen_3d_points;
        cv2eigen(prev_3d_points, eigen_3d_points);
        Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic> transformed_pt;
        transformed_pt = prev_H*eigen_3d_points;
        eigen2cv(transformed_pt, prev_3d_points);
        // cout << prev_H << endl;
        // cout << "H: " << endl << prev_H << endl;
        // cout << "3D_pre: " << endl << eigen_3d_points.col(0) << endl;
        // cout << "3D_post: " << endl << transformed_pt.col(0) << endl;

        vector<Point2f> matched_keypoint;
        Mat matched_3d_point;
        vector<KeyPoint> tempVec, tempVec2, tempVec3;
        
        Mat status1, err1;
        calcOpticalFlowPyrLK(prev_left_image, cur_left_image, stereoPointL, stereoPointL1, status1, err1);

        status1.convertTo(status1,CV_32F);
        bool flag = true;
        for(int i = 0; i < status1.rows; i++) {
            if(status1.at<float>(i,1) != 0) {
                matched_keypoint.push_back(stereoPointL1[i]);
                if(flag) {
                    flag = false;
                    prev_3d_points(Range(0, prev_3d_points.rows - 1), Range(i, i+1)).copyTo(matched_3d_point);
                } else {
                    Mat temp;
                    prev_3d_points(Range(0, prev_3d_points.rows - 1), Range(i, i+1)).copyTo(temp);
                    hconcat(matched_3d_point, temp, matched_3d_point);
                }
            }
        }
        // cout << matched_keypoint.size() << endl;
        // for(int i = 0; i < stereo_matches.size(); i++) {
        //     for(int j = 0; j < temporal_matches.size(); j++) {
        //         if((prev_left_keypoints[stereo_matches[i].queryIdx].pt.x == prev_left_keypoints[temporal_matches[j].queryIdx].pt.x) &&
        //         (prev_left_keypoints[stereo_matches[i].queryIdx].pt.y == prev_left_keypoints[temporal_matches[j].queryIdx].pt.y)) {
        //             matched_keypoint.push_back(cur_left_keypoints[temporal_matches[j].trainIdx].pt);
        //             tempVec.push_back(cur_left_keypoints[temporal_matches[j].trainIdx]);
        //             tempVec3.push_back(prev_left_keypoints[temporal_matches[j].queryIdx]);
        //             if(matched_3d_point.cols == 0) {
        //                 prev_3d_points(Range(0, prev_3d_points.rows - 1), Range(i, i+1)).copyTo(matched_3d_point);
        //             } else {
        //                 Mat temp;
        //                 prev_3d_points(Range(0, prev_3d_points.rows - 1), Range(i, i+1)).copyTo(temp);
        //                 hconcat(matched_3d_point, temp, matched_3d_point);
        //             }
        //         }
        //     }
        // }
        // Mat outimgTemp;
        // drawKeypoints(cur_left_image, tempVec, outimgTemp, Scalar::all(-1), DrawMatchesFlags::DEFAULT);
        // imshow("Test 1", outimgTemp);
        // Mat outimgTemp2;
        // drawKeypoints(prev_left_image, tempVec3, outimgTemp2, Scalar::all(-1), DrawMatchesFlags::DEFAULT);
        // imshow("Test 0", outimgTemp2);
        Mat matched_keypoints_mat = Mat(matched_keypoint);
        Mat matched_3d_points_t = matched_3d_point.t();
        Mat points_3d;
        Mat points_2d;
        flag = true;
        for(int i = 0; i < matched_3d_points_t.rows; i++) {
            // cout << matched_3d_points_t.row(i) << endl;
            if((abs(matched_3d_points_t.at<float>(i, 0)) < 100000) &&
            (abs(matched_3d_points_t.at<float>(i, 1)) < 100000) &&
            (abs(matched_3d_points_t.at<float>(i, 2)) < 100000) &&
            (matched_3d_points_t.at<float>(i, 2) > 0)) {
                // tempVec2.push_back(tempVec[i]);
                if(flag) {
                    flag=false;
                    matched_3d_points_t(Range(i, i+1), Range(0, matched_3d_points_t.cols)).copyTo(points_3d);
                    matched_keypoints_mat(Range(i, i+1), Range(0, matched_keypoints_mat.cols)).copyTo(points_2d);
                } else {
                    Mat t1, t2;
                    matched_3d_points_t(Range(i, i+1), Range(0, matched_3d_points_t.cols)).copyTo(t1);
                    matched_keypoints_mat(Range(i, i+1), Range(0, matched_keypoints_mat.cols)).copyTo(t2);
                    vconcat(points_3d, t1, points_3d);
                    vconcat(points_2d, t2, points_2d);

                }
            }
        }
        
        Mat outimgTemp1;
        drawKeypoints(cur_left_image, tempVec2, outimgTemp1, Scalar::all(-1), DrawMatchesFlags::DEFAULT);
        imshow("Test 2", outimgTemp1);
        Mat Rvec, R, T;
        // cout << matched_3d_points_t.size() << "\t->\t" << points_3d.size() << "\t|\t" << matched_keypoints_mat.size() << "\t->\t" << points_2d.size() << endl;
        // if(points_2d.rows < 4 && points_2d.cols <= 1) {
        //     waitKey(0);
        //     continue;
        // }
        solvePnP(points_3d, points_2d, camera_matrix,  distortion_mat, Rvec, T, false, SOLVEPNP_EPNP);
        Rodrigues(Rvec, R);
        Mat R_trans = R.t();
        Mat T_trans = -R_trans*T;

        Mat H = convertToHomogeneousMat(R_trans, T_trans);
        cout << T_trans.t() << endl;
        cv2eigen(H, prev_H);
        T_trans = T_trans.t();
        // cout << T.type() << endl;
        Point2f p = Point2f((int)(T_trans.at<double>(0,0)) + 500, (int)(T_trans.at<double>(0,2)) * (-1) + 400);
        circle(trajectory, p, 1, CV_RGB(255, 0, 0), FILLED);
        imshow("Trajectory", trajectory);
        waitKey(1);
    }
}

void getMatchesFlan(Mat prev_left_descriptors, Mat prev_right_descriptors, vector<DMatch> &stereo_matches) {

    vector<DMatch> matchesFlann;
    FlannBasedMatcher matcherFlann = FlannBasedMatcher(makePtr<flann::LshIndexParams>(12, 20, 2));
    matcherFlann.match(prev_left_descriptors, prev_right_descriptors, matchesFlann);
    auto min_maxFlann = minmax_element(matchesFlann.begin(), matchesFlann.end(), [](const DMatch &m1, const DMatch &m2) { return m1.distance < m2.distance; });
    double min_dist_flann = min_maxFlann.first->distance;
    double max_dist_flann = min_maxFlann.second->distance;
    for(int i = 0; i < matchesFlann.size(); i++) {
        if(matchesFlann[i].distance <= max(2 * min_dist_flann, 30.0)) {
            stereo_matches.push_back(matchesFlann[i]);
        }
    }
}

Mat convertToHomogeneousMat(Mat R, Mat T) {
    double zeros[3] = {0, 0, 0};
    Mat zeros_mat = Mat(1,3,CV_64F, zeros);
    Mat ones_mat = Mat(1,1,CV_64F, 1);
    Mat temp1, temp2, temp3;
    vconcat(R, zeros_mat, temp1);
    vconcat(T, ones_mat, temp2);
    hconcat(temp1, temp2, temp3);
    return temp3;
}

vector<string> split (string s, string delimiter) {
    size_t pos_start = 0, pos_end, delim_len = delimiter.length();
    string token;
    vector<string> res;

    while ((pos_end = s.find (delimiter, pos_start)) != string::npos) {
        token = s.substr (pos_start, pos_end - pos_start);
        pos_start = pos_end + delim_len;
        res.push_back (token);
    }

    res.push_back (s.substr (pos_start));
    return res;
}

void getProjectionMatrices(string path) {
    ifstream calibFile;
    calibFile.open(path+"/calib.txt");
    if(calibFile.is_open()) {
        string line;
        getline(calibFile, line);
        vector<string> l1 = split(line, " ");

        l1.erase(l1.begin());

        for(int i = 0; i < 4; i++) {
            float dummy = stof(l1[i]);
            lpm.at<float>(0,i) = dummy;
        }
        for(int i = 4; i < 8; i++) {
            float dummy = stof(l1[i]);
            lpm.at<float>(1,i-4) = dummy;
        }
        for(int i = 8; i < 12; i++) {
            float dummy = stof(l1[i]);
            lpm.at<float>(2,i-8) = dummy;
        }
        cout << lpm << endl;

        getline(calibFile, line);
        l1 = split(line, " ");
        l1.erase(l1.begin());
        for(int i = 0; i < 4; i++) {
            float dummy = stof(l1[i]);
            rpm.at<float>(0,i) = dummy;
        }
        for(int i = 4; i < 8; i++) {
            float dummy = stof(l1[i]);
            rpm.at<float>(1,i-4) = dummy;
        }
        for(int i = 8; i < 12; i++) {
            float dummy = stof(l1[i]);
            rpm.at<float>(2,i-8) = dummy;
        }
        cout << rpm << endl;
        calibFile.close();
    } else {
        cout << "Calibration File Could Not Open" << endl;
    }
}

void getMatchesBrute(Mat descriptors1, Mat descriptors2, vector<DMatch> &goodMatchBrute) {

    vector<DMatch> matchesBrute;
    vector<vector<DMatch>> matchesBrute1;
    Ptr<DescriptorMatcher> matcherBrute = DescriptorMatcher::create(DescriptorMatcher::BRUTEFORCE_L1);    
    matcherBrute->match(descriptors1, descriptors2, matchesBrute);
    matcherBrute->knnMatch(descriptors1, descriptors2, matchesBrute1, 2);
    auto min_maxBrute = minmax_element(matchesBrute.begin(), matchesBrute.end(), [](const DMatch &m1, const DMatch &m2) { return m1.distance < m2.distance; });
    double min_dist_brute = min_maxBrute.first->distance;
    double max_dist_brute = min_maxBrute.second->distance;
    // printf("−− Max dist Brute: %f \n", max_dist_brute);
    // printf("−− Min dist Brute: %f \n", min_dist_brute);
    for(int i = 0; i < matchesBrute1.size(); i++) {
        if(matchesBrute1[i][0].distance < 0.7f*matchesBrute1[i][1].distance) {
            goodMatchBrute.push_back(matchesBrute1[i][0]);
        }
    }
}