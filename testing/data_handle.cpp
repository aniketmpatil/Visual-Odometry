
#include <fstream>
#include "opencv2/opencv.hpp"
#include <numeric>

// #include <iostream>
// #include <string>
// #include "opencv2/core/core.hpp"
// #include <opencv2/calib3d/calib3d.hpp>
// #include <opencv2/highgui/highgui.hpp>
// #include <opencv2/imgproc/imgproc.hpp>
// #include <opencv2/imgcodecs.hpp>

// // Try matplotlib-cpp to plot GT_poses
// #include "matplotlibcpp.h"
// #include <vector>
// namespace plt = matplotlibcpp;
// int main() {
//     std::vector<double> y = {1, 3, 2, 4};
//     plt::plot(y);
//     plt::savefig("minimal.pdf");
// }

using namespace std;
using namespace cv;

// vector<string> break_string(string inputString){
//     // string out_str;
//     vector<string> wordVector;
//     std::stringstream stringStream(inputString);
//     std::string line;
//     while(std::getline(stringStream, line)) 
//     {
//         std::size_t prev = 0, pos;
//         while ((pos = line.find_first_of(" ';", prev)) != std::string::npos)
//         {
//             if (pos > prev)
//                 wordVector.push_back(line.substr(prev, pos-prev));
//             prev = pos+1;
//         }
//         if (prev < line.length())
//             wordVector.push_back(line.substr(prev, std::string::npos));
//     }
//     return wordVector;
// }

// Function to import Ground Truth poses from the 
vector<Point3f> import_GT(){
    vector<Point3f> gt_pose;
    Point3f pose;
    string op_str;
    double prev_x = 0, prev_y = 0, prev_z = 0, val;
    fstream myReadFile;
    myReadFile.open("dataset/poses/sample.txt");
    try {
        while (!myReadFile.eof()) {
            getline(myReadFile, op_str);
            myReadFile >> op_str;
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

vector<double> calculate_error(vector<Point3f> estimate, vector<Point3f> gt){
    int i = 0;
    vector<double> traj_error;
    
    for (int i=0; i<estimate.size(); i++){
        double abs_error = sqrt(pow((estimate[i].x - gt[i].x), 2)
                            + pow((estimate[i].y - gt[i].y), 2)
                            + pow((estimate[i].z - gt[i].z), 2));
        traj_error.push_back(abs_error);
        double mean_error = std::accumulate(traj_error.begin(), traj_error.end(), 0.0) / traj_error.size();
    }
    return traj_error;
}


int main() {
    cout << "Hello" << endl;
    // float arr[3][3] = {{1, 0, 0}, {0, 1, 0}, {0, 0, 1}};
    // Mat rot = Mat(3, 3, CV_32F, arr);
    // Mat trans = {};
    vector<Point3f> op_pose = import_GT();
    cout << op_pose << endl;
    return 0;
}

