# Visual Odometry using Classical Computer Vision

### *RBE 549: Computer Vision - Worcester Polytechnic Institute, Fall 2021*
### Team Megatron (Team M)
### Members: Aniket Patil, Chinmay Todankar, Nihal Navale, Prathamesh Bhamare

--------------------------------------------------------------

## What is Visual Odometry?

Visual odometry (VO) is the process of determining the position and orientation (ego-motion) of a robot/agent by analyzing images taken from a monocular or stereo camera system attached to the robot/agent. Visual Odometry operates by estimating the pose of the robot/agent by analyzing the changes that motion induces on the images of its onboard cameras.

## Our Implementation:

![trajectoryVideoGIF](vo_final_results.gif)

## Requirements:

1. Ubuntu with [VSCode](https://code.visualstudio.com/) for best implementation (Open this cloned repository as a folder in VSCode)
2. [OpenCV](https://opencv.org/opencv-4-0/) (C++): [Steps for Installation](https://docs.opencv.org/4.x/d7/d9f/tutorial_linux_install.html)
3. [GNUplot](http://www.gnuplot.info/): sudo apt install gnuplot (Optional: Only to generate plots from the .dat files)

> IMPORTANT: You will also need the KITTI Dataset of grayscale sequences. [Download](https://s3.eu-central-1.amazonaws.com/avg-kitti/data_odometry_gray.zip)

Once downloaded, extract the Sequences folder into the Dataset folder such that the structure is like: 
 * [dataset](./dataset)
   * [poses](./dataset/poses)
   * [sequences](./dataset/sequences)

## How to run the code?

1. If running from VSCode, open the folder Visual-Odometry in the VSCode navigator. Then open the `main.cpp` file in the code editor and press `Ctrl + Shift + B` to build the code. Then run the following command in the terminal below:
  ```
  ./src/build/main 00
  ```
  Replace 00 with sequence number to run other sequences

2. If running from terminal, go to the parent folder of this repo, that is, Visual-Odometry and enter the command:
  ```
  g++ -g src/main.cpp -o src/build/main `pkg-config --cflags --libs opencv4`
  ```

## Our Results (Trajectory images and error plots):

![Results](https://user-images.githubusercontent.com/83787152/146295990-5f2f34e7-a012-4e1c-9113-e64d5370c9a5.png)

