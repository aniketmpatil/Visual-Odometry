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

void plot();
void calculate_ATE(Point2f estimate_pt, Point2f gt_point, int img_seq, Mat prev_3d_points, VideoWriter video);
void getMatchesFlan(Mat prev_left_descriptors, Mat prev_right_descriptors, vector<DMatch> &stereo_matches);
void getMatchesBrute(Mat descriptors1, Mat descriptors2, vector<DMatch> &goodMatchBrute);
Mat convertToHomogeneousMat(Mat R, Mat T);
vector<string> split (string s, string delimiter);
void getProjectionMatrices(string path);
vector<Point3f> import_GT(string seq);

Mat lpm = cv::Mat(3, 4, CV_32F);
Mat rpm = cv::Mat(3, 4, CV_32F);
Mat camera_matrix, rot_matrix, trans_vect, distortion_mat;
vector<double> x_error, z_error, total_error, mean_error_arr, img_num_arr, dist_arr, mean2_arr;
Point2f prev_point;
double dist_trav = 0.0;

Mat trajectory = Mat(1000, 1000, CV_8UC3);

Mat prev_3d_points;
float hTemp[] = {1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1};
Mat prev_H = Mat(4, 4, CV_32F, hTemp);
Mat T_trans_prev;
vector<Point3f> ground_truth_poses;
int main(int argc, char** argv) {
    string seq = argv[1];
    string path = "dataset/sequences/"+seq;
    string left_images_path = path+"/image_0/*.png";
    string right_images_path = path+"/image_1/*.png";
    ground_truth_poses = import_GT(seq);
    getProjectionMatrices(path);
    // waitKey(0);
    vector<string> left_images_names;
    vector<string> right_images_names;
    glob(left_images_path, left_images_names);
    glob(right_images_path, right_images_names);
    cout << prev_H << endl;
    
    decomposeProjectionMatrix(lpm, camera_matrix, rot_matrix, trans_vect);
    cout << "Cam Mat: " << camera_matrix << endl;
    cout << "Rot Mat: " << rot_matrix << endl;
    cout << "Trans Mat: " << trans_vect << endl;
    distortion_mat = Mat::zeros(4,1,CV_64F);

    vector<KeyPoint> prev_left_keypoints, prev_right_keypoints, cur_left_keypoints;
    Mat prev_left_descriptors,prev_right_descriptors, cur_left_descriptors;
    Ptr<FeatureDetector> detector = ORB::create(1000);
    Ptr<DescriptorExtractor> descriptor = ORB::create(1000);

    Mat left_image = imread(left_images_names[0], IMREAD_GRAYSCALE);
    Mat right_image = imread(right_images_names[0], IMREAD_GRAYSCALE);

    VideoWriter trajectoryVideo("testing/trajectoryVideo.avi", cv::VideoWriter::fourcc('M','J','P','G'), 10, Size(1000,1000));
    VideoWriter leftVideo("testing/leftVideo.avi", cv::VideoWriter::fourcc('M','J','P','G'), 10, left_image.size());
    VideoWriter rightyVideo("testing/rightVideo.avi", cv::VideoWriter::fourcc('M','J','P','G'), 10, right_image.size());

    try {
        for(int image_seq = 1; image_seq < left_images_names.size(); image_seq++) {
            // cout << "Image : " << image_seq << endl;
            Mat prev_left_image = imread(left_images_names[image_seq-1], IMREAD_GRAYSCALE);
            Mat prev_right_image = imread(right_images_names[image_seq-1], IMREAD_GRAYSCALE);
            Mat cur_left_image = imread(left_images_names[image_seq], IMREAD_GRAYSCALE);
            
            

            detector->detect(prev_left_image, prev_left_keypoints);
            detector->detect(prev_right_image, prev_right_keypoints);
            detector->detect(cur_left_image, cur_left_keypoints);
            detector->detectAndCompute(prev_left_image, Mat(), prev_left_keypoints, prev_left_descriptors, true);
            detector->detectAndCompute(prev_right_image, Mat(), prev_right_keypoints, prev_right_descriptors, true);
            detector->detectAndCompute(cur_left_image, Mat(), cur_left_keypoints, cur_left_descriptors, true);
            // descriptor->compute(cur_left_image, cur_left_keypoints, cur_left_descriptors);
            

            vector<DMatch> stereo_matches;
            getMatchesFlan(prev_left_descriptors, prev_right_descriptors, stereo_matches);
            // getMatchesBrute(prev_left_descriptors, prev_right_descriptors, stereo_matches);
            // cout << "Matches : " << stereo_matches.size() << endl;
            // vector<DMatch> temporal_matches;
            // getMatchesFlan(prev_left_descriptors, cur_left_descriptors, temporal_matches);

            vector<Point2f> stereoPointL, stereoPointR, stereoPointL1;
            vector<KeyPoint> matched_l_keypoints, matched_r_keypoints;
            for(int i = 0; i < stereo_matches.size(); i++) {
                stereoPointL.push_back(prev_left_keypoints[stereo_matches[i].queryIdx].pt);
                stereoPointR.push_back(prev_right_keypoints[stereo_matches[i].trainIdx].pt);
                matched_l_keypoints.push_back(prev_left_keypoints[stereo_matches[i].queryIdx]);
                matched_r_keypoints.push_back(prev_right_keypoints[stereo_matches[i].trainIdx]);
            }
            // cout << stereoPointL.size() << ", " << stereoPointR.size() << endl;
            prev_3d_points.convertTo(prev_3d_points, CV_32F);
            triangulatePoints(lpm, rpm, stereoPointL, stereoPointR, prev_3d_points);
            
            // cout << prev_3d_points.size() << endl;
            // cout << prev_3d_points.type() << endl;
            // cout << "Old : " << prev_3d_points.col(0).t() << endl;
            for(int i = 0; i < prev_3d_points.cols; i++) {
                
                prev_3d_points.at<float>(0,i) = prev_3d_points.at<float>(0,i) / prev_3d_points.at<float>(3,i);
                prev_3d_points.at<float>(1,i) = prev_3d_points.at<float>(1,i) / prev_3d_points.at<float>(3,i);
                prev_3d_points.at<float>(2,i) = prev_3d_points.at<float>(2,i) / prev_3d_points.at<float>(3,i);
                // cout << "New : " << prev_3d_points.col(i).t() << endl;
                // cout << "A: " << prev_3d_points.col(i) << endl;
                // Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic> eigen_3d_point;
                // Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic> Heig;
                // cv2eigen(prev_3d_points.col(i), eigen_3d_point);
                // cv2eigen(prev_H, Heig);
                // Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic> transformed_pt;
                // transformed_pt = Heig*eigen_3d_point;
                // eigen2cv(transformed_pt, prev_3d_points.col(i));
                // prev_3d_points.col(i) = prev_H*prev_3d_points.col(i);
                // cout << "H: " << prev_H << endl;
                // cout << "B: " << prev_3d_points.col(i) << endl;   
            }

            Mat prev_3d_Pts_copy;
            prev_3d_points.copyTo(prev_3d_Pts_copy);

            for(int i=0;i<prev_3d_points.cols;i++)
            {
                prev_3d_points.at<float>(0,i)=prev_H.at<float>(0, 0)*prev_3d_Pts_copy.at<float>(0,i)+prev_H.at<float>(0, 1)*prev_3d_Pts_copy.at<float>(1,i)+prev_H.at<float>(0, 2)*prev_3d_Pts_copy.at<float>(2,i)+prev_H.at<float>(0, 3);
                prev_3d_points.at<float>(1,i)=prev_H.at<float>(1, 0)*prev_3d_Pts_copy.at<float>(0,i)+prev_H.at<float>(1, 1)*prev_3d_Pts_copy.at<float>(1,i)+prev_H.at<float>(1, 2)*prev_3d_Pts_copy.at<float>(2,i)+prev_H.at<float>(1, 3);
                prev_3d_points.at<float>(2,i)=prev_H.at<float>(2, 0)*prev_3d_Pts_copy.at<float>(0,i)+prev_H.at<float>(2, 1)*prev_3d_Pts_copy.at<float>(1,i)+prev_H.at<float>(2, 2)*prev_3d_Pts_copy.at<float>(2,i)+prev_H.at<float>(2, 3);

            }
            // cout << "H : " << prev_H << endl;
            // cout << "New 2 : " << prev_3d_points.col(0).t() << endl;


            // cout << lpm << endl << rpm << endl;
            // cout << prev_3d_points.col(0) << endl;
            
            // Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic> eigen_3d_points;
            // cv2eigen(prev_3d_points, eigen_3d_points);
            // Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic> transformed_pt;
            // transformed_pt = prev_H*eigen_3d_points;
            // eigen2cv(transformed_pt, prev_3d_points);
            // cout << prev_H << endl;
            // cout << "H: " << endl << prev_H << endl;
            // cout << "3D_pre: " << endl << eigen_3d_points.col(0) << endl;
            // cout << "3D_post: " << endl << transformed_pt.col(0) << endl;

            vector<Point2f> matched_keypoint;
            Mat matched_3d_point;
            vector<KeyPoint> tempVec, tempVec2, tempVec3;
            
            Mat status1, err1;
            // cout << "Stereo Point L: " << stereoPointL.size() << endl;
            bool flag = true;
            //********** Optical Flow Start ****************
            calcOpticalFlowPyrLK(prev_left_image, cur_left_image, stereoPointL, stereoPointL1, status1, err1);
            // cout << "Stereo Point L1: " << stereoPointL1.size() << endl;
            // cout << "StatSize: " << status1.size() << endl;
            status1.convertTo(status1,CV_32F);

            
            for(int i = 0; i < status1.rows; i++) {
                if(status1.at<float>(i,0) != 0) {
                    matched_keypoint.push_back(stereoPointL1[i]);
                    if(flag) {
                        flag = false;
                        // cout << prev_3d_points.col(i) << endl;
                        prev_3d_points(Range(0, prev_3d_points.rows - 1), Range(i, i+1)).copyTo(matched_3d_point);
                        // cout << matched_3d_point << endl;
                    } else {
                        Mat temp;
                        // cout << prev_3d_points.col(i) << endl;
                        prev_3d_points(Range(0, prev_3d_points.rows - 1), Range(i, i+1)).copyTo(temp);
                        // cout << temp << endl;
                        hconcat(matched_3d_point, temp, matched_3d_point);
                    }
                }
            }
            //*********** Optical Flow End *****************
            // cout << matched_3d_point << endl;
            // cout << "M3dP:" << matched_3d_point.type() << endl;
            // cout << "A : " << matched_keypoint.size() << endl;
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
            Mat outimgTemp;
            drawKeypoints(prev_left_image, matched_l_keypoints, outimgTemp, Scalar::all(-1), DrawMatchesFlags::DEFAULT);
            imshow("Left Keypoints", outimgTemp);
            // leftVideo.write(outimgTemp);
            Mat outimgTemp2;
            drawKeypoints(prev_right_image, matched_r_keypoints, outimgTemp2, Scalar::all(-1), DrawMatchesFlags::DEFAULT);
            imshow("Right Keypoints", outimgTemp2);
            // rightyVideo.write(outimgTemp2);
            Mat matched_keypoints_mat = Mat(matched_keypoint);
            matched_keypoints_mat = matched_keypoints_mat.reshape(1);
            Mat matched_3d_points_t = matched_3d_point.t();
            // cout << "matchedKeyPM:" << matched_keypoints_mat.type() << ", " << matched_keypoints_mat.size() << endl;
            // cout << "matched3DPM:" << matched_3d_points_t.type() << ", " << matched_3d_points_t.size() << endl;
            
            
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
                } else {
                    // cout << matched_3d_points_t.row(i) << endl;
                }
            }

            

            // cout << "point2d: " << points_2d.type() << ", " << points_2d.size() << endl; 
            // cout << "point3d: " << points_3d.type() << ", " << points_3d.size() << endl; 
            // cout << "B : " << points_2d.size() << endl;
            // Mat outimgTemp1;
            // drawKeypoints(cur_left_image, tempVec2, outimgTemp1, Scalar::all(-1), DrawMatchesFlags::DEFAULT);
            // imshow("Test 2", outimgTemp1);
            Mat Rvec, R, T;
            // cout << matched_3d_points_t.size() << "\t->\t" << points_3d.size() << "\t|\t" << matched_keypoints_mat.size() << "\t->\t" << points_2d.size() << endl;
            if(points_2d.rows < 4 && points_2d.cols <= 1) {
                waitKey(0);
                continue;
            }
            points_2d.convertTo(points_2d, CV_64F);
            points_3d.convertTo(points_3d, CV_64F);
            camera_matrix.convertTo(camera_matrix, CV_64F);
            distortion_mat.convertTo(distortion_mat, CV_64F);
            // cout << "Type 3pts : " << points_3d.type() << endl;
            // cout << "Type 2pts : " << points_2d.type() << endl;
            // cout << camera_matrix.type() << ", " << distortion_mat.type() << endl;


            // Mat selected_3D_points,selected_2D_points;
            // int number_of_points = points_3d.rows;
            // int points_to_select = 20;
            // int point_spacing = (int)(floor(number_of_points / points_to_select));
            // flag = true;
            // for(int i = 0; i < number_of_points; i += point_spacing) {
            //     if(flag) {
            //         points_3d(Range(i, i+1), Range(0, points_3d.cols)).copyTo(selected_3D_points);
            //         points_2d(Range(i, i+1), Range(0, points_2d.cols)).copyTo(selected_2D_points);
            //         flag = false;
            //     } else {
            //         Mat t1, t2;
            //         points_3d(Range(i, i+1), Range(0, points_3d.cols)).copyTo(t1);
            //         points_2d(Range(i, i+1), Range(0, points_2d.cols)).copyTo(t2);
            //         vconcat(selected_3D_points, t1, selected_3D_points);
            //         vconcat(selected_2D_points, t2, selected_2D_points);
            //     }
            // }
            // cout << matched_3d_points_t.rows << ", " << point_spacing << ", " << selected_3D_points.rows << endl;
            Mat selected_3D_points,selected_2D_points;
            int number_of_points = points_3d.rows;
            double points_not_to_select = 0.20;
            int offset = 0;
            if(number_of_points >= 20) {
                int diff = (int)(floor(number_of_points * points_not_to_select));
                offset = (int)(floor(diff / 2));
                if((points_3d.rows - (2*offset)) <= 6) {
                    offset = 0;
                }
            }
            cout << points_3d.size() << ", " << offset << endl;
            solvePnPRansac(points_3d.rowRange(offset, points_3d.rows-offset), points_2d.rowRange(offset, points_2d.rows-offset), camera_matrix,  distortion_mat, Rvec, T, false);
            // cout << "T : " << T.t() << "\t";
            
            // cout << "Type R : " << Rvec.type() << endl;
            // cout << "Type T : " << T.type() << endl;
            Rodrigues(Rvec, R);
            // cout << "R : " << R << endl;
            Mat R_trans = R.t();
            Mat T_trans = -R_trans*T;
            // cout << T_trans.t() << T_trans.at<double>(0,1) << endl;
            
            if((image_seq > 1) && (((T_trans.at<double>(0,0)- T_trans_prev.at<double>(0, 0)) > 10) ||
                ((T_trans.at<double>(1,0)- T_trans_prev.at<double>(1, 0)) > 10) ||
                ((T_trans.at<double>(2,0)- T_trans_prev.at<double>(2, 0)) > 10))
            ) {
                continue;
            }
            T_trans.copyTo(T_trans_prev);
            // cout << R_trans << endl << T_trans << endl;
            Mat H = convertToHomogeneousMat(R_trans, T_trans);
            // cout << H << endl << H.type() << endl;
            // cout << T_trans.t() << endl;
            // prev_H = H;
            // cout << "Calc : " << H << endl;
            H.convertTo(prev_H, CV_32F);
            // prev_H = H;
            T_trans = T_trans.t();
            // cout << T_trans.type() << endl;
            Point2f estimated_point = Point2f((int)(T_trans.at<double>(0,0)) + 500, (int)(T_trans.at<double>(0,2)) * (-1) + 700);
            Point2f ground_truth_point = Point2f((int)(ground_truth_poses[image_seq].x) + 500, (int)(ground_truth_poses[image_seq].z) * (-1) + 700);
            // circle(trajectory, estimated_point, 1, CV_RGB(255, 0, 0), FILLED);
            // circle(trajectory, ground_truth_point, 1, CV_RGB(0, 255, 0), FILLED);
            // imshow("Trajectory", trajectory);
            img_num_arr.push_back(image_seq);
            calculate_ATE(estimated_point, ground_truth_point, image_seq, prev_3d_points, trajectoryVideo);

            waitKey(1);
        }
    } catch(Exception &e) {
        imwrite("testing/trajectory.png", trajectory);    
    }
    imwrite("testing/trajectory.png", trajectory);
    trajectoryVideo.release();
    leftVideo.release();
    rightyVideo.release();
    plot();
    cout << "************** The End **************" << endl;
}

void calculate_ATE(Point2f estimate_pt, Point2f gt_point, int img_seq, Mat prev_3d_points, VideoWriter video){
    double x_err = abs(estimate_pt.x - gt_point.x);
    double z_err = abs(estimate_pt.y - gt_point.y);
    double tot_error = sqrt(pow(x_err, 2) + pow(z_err, 2));

    x_error.push_back(x_err);
    z_error.push_back(z_err);
    total_error.push_back(tot_error);

    float mean_error = 0;
    for(auto& itr : total_error)
        mean_error += itr;
    mean_error /= total_error.size();
    mean_error_arr.push_back(mean_error);
    cout << "Absolute Error: " << tot_error << endl;

    if(img_seq == 1){
        prev_point.x = gt_point.x;
        prev_point.y = gt_point.y;
    }
    double dist_bw_frames = norm(Mat(gt_point), Mat(prev_point));
    prev_point.x = gt_point.x;
    prev_point.y = gt_point.y;
    // dist_arr.push_back(dist_bw_frames);
    dist_trav += dist_bw_frames;
    double mean2 = tot_error / dist_trav;
    cout << "Mean Error: " << mean2 << endl;
    mean2_arr.push_back(mean2);
    string text = "Absolute Error: " + to_string(tot_error);
    // rectangle(trajectory, Point2f(580, 100) , Point2f(1000, 130), CV_RGB(0, 0, 0), cv::FILLED);
    rectangle(trajectory, Point2f(0, 869) , Point2f(420, 899), CV_RGB(0, 0, 0), cv::FILLED);
    circle(trajectory, estimate_pt, 1, CV_RGB(255, 0, 0), FILLED);
    circle(trajectory, gt_point, 1, CV_RGB(0, 255, 0), FILLED);
    putText(trajectory, text, Point2f(20,879), cv::FONT_HERSHEY_COMPLEX_SMALL, 0.8, Scalar(255,255,0), 1, 5);
    putText(trajectory, "Red: Estimated Trajectory", Point2f(20,829), cv::FONT_HERSHEY_COMPLEX_SMALL, 0.7, CV_RGB(255, 0, 0), 1, 5);
    putText(trajectory, "Green: Ground Truth", Point2f(20,849), cv::FONT_HERSHEY_COMPLEX_SMALL, 0.7, CV_RGB(0, 255, 0), 1, 5);
    Mat trajTemp = trajectory.clone();
        for(int i = 0; i < prev_3d_points.cols; i++) {
            // cout << (int)(prev_3d_points.at<float>(2,i)) << endl;
            if((abs((int)(prev_3d_points.at<float>(2,i))) * (-1) < 15) && ((int)(prev_3d_points.at<float>(2,i))) * (-1) < 0) {
                Point2f point_3d = Point2f((int)(prev_3d_points.at<float>(0,i)) + 500, (int)(prev_3d_points.at<float>(2,i)) * (-1) + 700);
                circle(trajTemp, point_3d, 1, CV_RGB(255, 255, 255), FILLED);
            }
        
        }

    imshow("Trajectory", trajTemp);
    // video.write(trajTemp);
    cout << "----------------------" << endl;
}

void plot(){

    // std::ifstream file;
    ofstream file("testing/output.dat");
    file << "# X" << " " << "Y" << " \n";
    for(int i = 0; i < total_error.size(); i++){
        file << "  " << img_num_arr.at(i) << " " << total_error.at(i) << " \n";
    }
    file.close();
    ofstream file2("testing/mean_err.dat");
    file2 << "# X" << " " << "Y" << " \n";
    for(int i = 0; i < total_error.size(); i++){
        file2 << "  " << img_num_arr.at(i) << " " << mean2_arr.at(i) << " \n";
    }
    file2.close();
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

vector<Point3f> import_GT(string seq) {
    vector<Point3f> gt_pose;
    Point3f pose;
    string op_str;
    double prev_x = 0, prev_y = 0, prev_z = 0, val;
    fstream myReadFile;
    myReadFile.open("dataset/poses/"+seq+".txt");
    try {
        while (!myReadFile.eof()) {
            getline(myReadFile, op_str);
            // myReadFile >> op_str;
            istringstream trans(op_str);
            for (int i=0; i<12; i++)
            {
                trans >> val;
                if(i==3) pose.x=val;
                if(i==7) pose.y=val;
                if(i==11) pose.z=val;
            }
            gt_pose.push_back(pose);
        }
    }
    catch(const char* msg) {
        cerr << "Error importing Ground Truths from txt file" << endl;
        cerr << msg << endl;
    }
    myReadFile.close();
    return gt_pose;
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