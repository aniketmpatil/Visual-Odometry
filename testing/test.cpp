#include "opencv2/opencv.hpp"

#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <chrono>

using namespace cv;
using namespace std;

void getKeyPoints(Mat inputImage, vector<KeyPoint>& keyPoints) {

    keyPoints.clear();

    for(int i = 3; i < inputImage.rows - 3; i++) {
        for(int j = 3; j < inputImage.cols - 3; j++) {
            int p = inputImage.at<uchar>(i,j); 
            int T = p * 0.7;
            int a[16];
            a[0] = inputImage.at<uchar>(i ,j - 3);
            a[1] = inputImage.at<uchar>(i - 1,j - 3);
            a[2] = inputImage.at<uchar>(i - 2,j - 2);
            a[3] = inputImage.at<uchar>(i - 3,j - 1);
            a[4] = inputImage.at<uchar>(i - 3,j);
            a[5] = inputImage.at<uchar>(i - 3,j + 1);
            a[6] = inputImage.at<uchar>(i - 2,j + 2);
            a[7] = inputImage.at<uchar>(i - 1,j + 3);
            a[8] = inputImage.at<uchar>(i ,j + 3);
            a[9] = inputImage.at<uchar>(i + 1,j + 3);
            a[10] = inputImage.at<uchar>(i + 2,j + 2);
            a[11] = inputImage.at<uchar>(i + 3,j + 1);
            a[12] = inputImage.at<uchar>(i + 3,j );
            a[13] = inputImage.at<uchar>(i + 3,j - 1);
            a[14] = inputImage.at<uchar>(i + 2,j - 2);
            a[15] = inputImage.at<uchar>(i + 1,j - 3);
            bool first = true;
            int firstN = 0;
            int mx = 0;
            int cur = 0, cur1 = 0;
            for( int k = 0; k < 16; k++) {
                if(a[k] < p - T || a[k] > p + T) {
                    cur++;
                } else {
                    if(first) {
                        first = false;
                        firstN = cur;
                    }
                    mx = cur>mx?cur:mx;
                    cur = 0;
                }

                // if () {
                //     cur1++;
                // } else {
                //     // if(first) {
                //     //     first = false;
                //     //     firstN = cur;
                //     // }
                //     mx = cur1>mx?cur1:mx;
                //     cur1 = 0;
                // }
            }
            cur = cur+firstN;
            mx = cur>mx?cur:mx;
            if(mx >= 12) {
                Point2f pt;
                pt.x = j;
                pt.y = i;
                KeyPoint keyPoint(pt, 3);
                bool flag = false;
                for(KeyPoint k : keyPoints) {
                    if(k.pt == keyPoint.pt) {
                        flag = true;
                        break;
                    }
                }
                if(!flag) {
                    keyPoints.push_back(keyPoint);
                }
            }
        }
    }
}

int main(int argc, char **argv) {

    Mat left = imread("testing/images/left.jpeg");
    Mat right = imread("testing/images/right.jpeg");
    vector<KeyPoint> keyPointVector1, keyPointVector2;
    Mat descriptors1,descriptors2;
    Ptr<FeatureDetector> detector = ORB::create();
    Ptr<DescriptorExtractor> descriptor = ORB::create();
    chrono::steady_clock::time_point t1 = chrono::steady_clock::now();
    detector->detect(left, keyPointVector1);
    // detector->detect(right, keyPointVector2);

    descriptor->compute(left, keyPointVector1, descriptors1);
    // descriptor->compute(right, keyPointVector2, descriptors2);

    chrono::steady_clock::time_point t2 = chrono::steady_clock::now();

    chrono::duration<double> time_used = chrono::duration_cast<chrono::duration<double> > (t2 - t1);
    cout << "extract ORB cost = " << time_used.count() << " seconds. " << endl;

    Mat outimg1;
    drawKeypoints(left, keyPointVector1, outimg1, Scalar::all(-1), DrawMatchesFlags::DEFAULT);
    imshow("ORB features", outimg1);
    imshow("Descriptors", descriptors1);
    t1 = chrono::steady_clock::now();
    FAST(left, keyPointVector1, 50);
    
    t2 = chrono::steady_clock::now();
    time_used = chrono::duration_cast<chrono::duration<double> > (t2 - t1);
    cout << "extract FAST cost = " << time_used.count() << " seconds. " << endl;
    Mat outimg2;
    drawKeypoints(left, keyPointVector1, outimg2, Scalar::all(-1), DrawMatchesFlags::DEFAULT);
    imshow("FAST features", outimg2);

    t1 = chrono::steady_clock::now();
    getKeyPoints(left, keyPointVector1);
    t2 = chrono::steady_clock::now();

    time_used = chrono::duration_cast<chrono::duration<double> > (t2 - t1);
    cout << "extract Custom cost = " << time_used.count() << " seconds. " << endl;
    Mat outimg3;
    drawKeypoints(left, keyPointVector1, outimg3, Scalar::all(-1), DrawMatchesFlags::DEFAULT);
    cout << keyPointVector1.size() << endl;
    // for(KeyPoint k : keyPointVector1) {
    //     cout << "(" << k.pt.x << ", " << k.pt.y << ")" << endl;
    // }
    imshow("Custom features", outimg3);

    waitKey(0);
    return 0;
}