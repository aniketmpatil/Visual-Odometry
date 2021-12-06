#include <stdio.h>
#include <iostream>
#include<cmath>
#include<vector>
#include <fstream>

#include "opencv2/core.hpp"
#include "opencv2/features2d.hpp"
// #include "opencv2/xfeatures2d.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/video/tracking.hpp"
#include "opencv2/core/core.hpp"
#include "opencv2/calib3d/calib3d.hpp"
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

//#include "opencv2/video/tracking.hpp"


using namespace std;
using namespace cv;
// using namespace cv::xfeatures2d;
using std::ofstream;

Mat find3Dpoints(Mat ,Mat );
tuple<Mat,Mat> find2Dpoints(Mat ,Mat ,Mat ,Mat );
Mat findhomogenousmat(Mat ,Mat );
vector<Point3f> groundtruth();
vector<Point3f> gt_poses;

int main()
{
	ofstream outdata;
  int flag=0; 
  int k=1;
  double prevp[3][1]={{0},{0},{0}};
    Mat startp=Mat(3,1,CV_64F,prevp);
    Mat prevpoint=Mat::zeros(1,3,CV_64F);
    double firstpoint[1][3]={0,0,0};
    Mat allpoints=Mat(1,3,CV_64F,firstpoint);
    Mat newpoint=Mat::zeros(1,3,CV_64F);
    double ho[4][4]={{1,0,0,0},{0,1,0,0},{0,0,1,0},{0,0,0,1}};
    Mat H =Mat(4,4,CV_64F,ho);
    //cout<<"Ho is"<<endl<<H<<endl;
    Mat traj = Mat::zeros(1000, 1000, CV_8UC3);
    Mat tinvprev=Mat::zeros(3,1,CV_64F);
    //cout<<"Tinvprev:"<<endl<<tinvprev<<endl;



    Mat frameL1, frameL2, frameR1,frameR2;
    string filename1,filename2,filename3,filename4;
    vector<Point2f>keypointscont;
    float x_err, z_err, tot_error;
    vector<float> x_error, z_error, t_error;
    x_error.clear();
    z_error.clear();
    t_error.clear();


    vector<Point3f> gt_poseplot=groundtruth();
	for (int m=1; m<1200; m=m+1)
	{	

		if (k<10)
		{
			filename1 = "dataset/sequences/00/image_0/00000" + to_string(k) + ".png";
      filename2 = "dataset/sequences/00/image_1/00000" + to_string(k) + ".png";		 	
		}

		if (k>=10&&k<100)
		{
			filename1 = "dataset/sequences/00/image_0/0000" + to_string(k) + ".png";
      filename2 = "dataset/sequences/00/image_1/0000" + to_string(k) + ".png";     
    
		}

		if(k>=100&&k<1000)
		{
			filename1 = "dataset/sequences/00/image_0/000" + to_string(k) + ".png";
      filename2 = "dataset/sequences/00/image_1/000" + to_string(k) + ".png";     
    
		}

    if(k>=1000&&k<2000)
    {
      filename1 = "dataset/sequences/00/image_0/00" + to_string(k) + ".png";
      filename2 = "dataset/sequences/00/image_1/00" + to_string(k) + ".png";     
    
    }

		if ((m+1)<10)
		{
			filename3 = "dataset/sequences/00/image_0/00000" + to_string(m+1) + ".png";
      filename4 = "dataset/sequences/00/image_1/00000" + to_string(m+1) + ".png";     
    
		}

		if ((m+1)>=10&&(m+1)<100)
		{
			filename3 = "dataset/sequences/00/image_0/0000" + to_string(m+1) + ".png";
      filename4 = "dataset/sequences/00/image_1/0000" + to_string(m+1) + ".png";     
    
		}

		if((m+1)>=100&&(m+1)<1000)
		{
			filename3 = "dataset/sequences/00/image_0/000" + to_string(m+1) + ".png";
      filename4 = "dataset/sequences/00/image_1/000" + to_string(m+1) + ".png";     
    
		}

    if((m+1)>=1000&&(m+1)<2000)
    {
      filename3 = "dataset/sequences/00/image_0/00" + to_string(m+1) + ".png";
      filename4 = "dataset/sequences/00/image_1/00" + to_string(m+1) + ".png";     
    
    }


		 frameL1=imread(filename1,IMREAD_GRAYSCALE);
		 frameR1=imread(filename2,IMREAD_GRAYSCALE);
		 frameL2=imread(filename3,IMREAD_GRAYSCALE);
		 frameR2=imread(filename4,IMREAD_GRAYSCALE);
     
    

		 if(frameL1.empty()==1||frameL2.empty()==1||frameR1.empty()==1||frameR2.empty()==1)
		 {
		 	//cout<<"Cannot read image "<<frameL1.empty()<<" " <<frameL2.empty()<<" "<<frameR1.empty()<<" "<<frameR2.empty()<<endl;
		 	continue;
		 }

    	
		Mat TDkeypoints,Twdpoints;
		
                                                                     
		tie(TDkeypoints,Twdpoints)=find2Dpoints(frameL1, frameR1, frameL2, H);
		////cout<<"3D keypoints all "<<endl<<TDkeypoints<<endl<<"2D key points all "<<Twdpoints<<endl;
		//cout<<"--------------------------------------------------------------------------------------------------------------"<<endl;

		Mat rvecn,tvecn;
		//float cammatlv[3][3]={{721.53,0,609.55},{0,721.53,172.85},{0,0,1}};
    float cammatlv[3][3]={{718.856,0,607.1928},{0,718.856,185.2157},{0,0,1}};
  		//float distcoefflv[4]={0,0,0,0};
  		
  		Mat cammatl=Mat(3,3,CV_32F, cammatlv);
      cammatl.convertTo(cammatl,CV_64F);
      //cout<<"cam mat"<<endl<<cammatl<<endl;
  		//Mat distcoeffl=Mat(1,4,CV_32F, distcoefflv);
      Mat dist_coeffs = Mat::zeros(4,1,CV_64F);
      //cout<<"Distortion coefficients"<<endl<<dist_coeffs<<endl;
  		Mat inli;

		int row=TDkeypoints.rows;
    //cout<<"3D auxx row: "<<row<<endl;
    /*if(row<41)
      continue;*/
		Mat auxx= TDkeypoints.rowRange(7,row-5);

		int rowtd=Twdpoints.rows;
    /*if(rowtd<41)
      continue;*/
    //cout<<"2D auxxtd row: "<<rowtd<<endl;

		Mat auxxtd=Twdpoints.rowRange(7,rowtd-5);

			
		//cout<<"3D Points used"<<auxx<<endl<<"2D points used: "<<auxxtd<<endl;


  		solvePnPRansac(auxx,auxxtd, cammatl, dist_coeffs, rvecn, tvecn,false,100,8.0	,0.99,inli);
      //cout<<"size of inliers: "<<inli.size()<<endl;
 		 Mat rotationmatrix;
 		 // Mat prevpoint(tvecn.size(),tvecn.type());
 		 Rodrigues(rvecn, rotationmatrix);

 		 bool rotnanflag=false;

 		 for(int g=0;g<rotationmatrix.rows;g++)
 		 {
 		 	for(int d=0;d<rotationmatrix.cols;d++)
 		 	{
 		 		if(isnan(abs(rotationmatrix.at<double>(g,d)))==true || abs(rotationmatrix.at<double>(g,d))>10)
 		 			{
 		 				rotnanflag=true;
 		 				//cout<<"Value of the element: "<<abs(rotationmatrix.at<double>(g,d))<<endl;
 		 				//cout<<"Rotnanflag true at ("<<g<<","<<d<<")"<<endl;
 		 			}

 		 	}
 		 }
 		 //cout<<"hello"<<endl;

 		for(int x=0;x<rvecn.rows;x++)
 		 {
 		 	for(int y=0;y<rvecn.cols;y++)
 		 	{
 		 		////cout<<"Value of the element in rvecn: "<<rvecn.at<double>(x,y)<<endl;

 		 		if(rvecn.at<double>(x,y)==0)
 		 			{
 		 				rotnanflag=true;
 		 				// //cout<<"Value of the element: "<<rvecn.at<double>(x,y)<<endl;
 		 				//cout<<"Rotnanflag in rvecn true at ("<<x<<","<<y<<")"<<endl;
 		 			}

 		 	}
 		 }


 		 float thres=3000.0;
 		 bool transnanflag=false;

 		 for(int a=0;a<tvecn.rows;a++)
 		 {
 		 	for(int b=0;b<tvecn.cols;b++)
 		 	{
 		 		////cout<<"Value of the element in tvecn: "<<tvecn.at<double>(a,b)<<endl;

 		 		if(isnan(abs(tvecn.at<double>(a,b)))==true || abs(tvecn.at<double>(a,b))>thres ||abs(tvecn.at<double>(a,b))<0.00000001)
 		 		{
 		 			transnanflag=true;
 		 			// //cout<<"Value of the element: "<<tvecn.at<double>(a,b)<<endl;
 		 			//cout<<"transnanflag in tvecn true at ("<<a<<","<<b<<")"<<endl;
 		 			

 		 		}
 		 	}
 		 }

     Mat rotmat=rotationmatrix.t();
     Mat tinv=-rotmat*tvecn;
     if (tinv.at<double>(2,0)<0)
      continue;

    if(abs(tinv.at<double>(0,0)-tinvprev.at<double>(0,0))>15 || abs(tinv.at<double>(1,0)-tinvprev.at<double>(1,0))>15 || abs(tinv.at<double>(2,0)-tinvprev.at<double>(2,0))>15)
      continue;

 		 //cout<<"rotnanflag: "<<rotnanflag<<endl<<"transflag: "<<transnanflag<<endl;

 		 if(rotnanflag==true||transnanflag==true)
     {
      m=m+1;
      continue;
     }
 		 	
     
     //cout<<"Rotation matrix inverse:"<<endl<<rotmat<<endl<<"Translation inverse:"<<endl<<tinv<<endl;
     H=findhomogenousmat(rotmat,tinv);
     //cout<<"Homogenous matrix"<<endl<<H<<endl;
     //cout<<"H matrix in main"<<endl<<H<<endl;

 		 

 		 Mat currpoint(tvecn.size(),tvecn.type());
     H(Range(0,3),Range(3,4)).copyTo(currpoint);
     //cout<<"Current point:"<<currpoint<<endl;
 		 Mat cpoint(1,3,tvecn.type());

 		  cpoint=currpoint.t();

 		 vconcat(allpoints,cpoint,allpoints);
 		 
  		 k=m+1;
      string text  = "Red color: estimated trajectory";
      string text2 = "Green color: groundtruth";
      string coord = "Coordinates x: " + to_string(cpoint.at<double>(0,0)) + " y: " + to_string(cpoint.at<double>(0,1)) + " z: " + to_string(cpoint.at<double>(0,2));

      
      tinv.copyTo(tinvprev);
      
  		x_err=abs(cpoint.at<double>(0,0)-gt_poseplot[m-1].x);
      z_err=abs((cpoint.at<double>(0,2))*(-1)-(gt_poseplot[m-1].z)*(-1));
      //cout << "Test" << endl;
      x_error.push_back(x_err);
      z_error.push_back(z_err);
      //cout<<"X_error: "<<x_err<<endl;
      //cout<<"Z_error: "<<z_err<<endl;
      tot_error=sqrt(x_err*x_err+z_err*z_err);
      t_error.push_back(tot_error);
      //cout << "Test" << endl;
      // cout << cpoint << endl;
      Point2f center = Point2f(int(cpoint.at<double>(0,0)) + 500, int(cpoint.at<double>(0,2))*(-1) + 400);
      //cout<<"centre is: "<<cpoint.at<double>(0,0)<<" "<<cpoint.at<double>(0,2)<<endl;
      //cout<<"centre is: "<<cpoint<<endl;
      Point2f t_center = Point2f(int(gt_poseplot[m-1].x) + 500, int(gt_poseplot[m-1].z)*(-1) + 400);
      circle(traj, center ,1, CV_RGB(255,0,0), 2);
      circle(traj, t_center,1, CV_RGB(0,255,0), 2);
      rectangle(traj, Point2f(380, 10), Point2f(1000, 170),  CV_RGB(0,0,0), cv::FILLED);
      putText(traj, text, Point2f(400,40), cv::FONT_HERSHEY_COMPLEX_SMALL, 0.7, Scalar(0, 0,255), 1, 5);
      namedWindow("Name",WINDOW_NORMAL);
      putText(traj, text2, Point2f(400,60), cv::FONT_HERSHEY_COMPLEX_SMALL, 0.7, Scalar(0,255,0), 1, 5);
      putText(traj, coord, Point2f(400,90), cv::FONT_HERSHEY_COMPLEX_SMALL, 0.7, Scalar(0,0,255), 1, 5);
      //cv::imshow( "Road facing camera", frameL1 );
      cv::imshow( "Trajectory", traj);
      waitKey(1);

       flag=flag+1;

    }
    
     float m_x=0;
     float m_z=0;
     float m_t=0;
     float mean_x=0;
     float mean_z=0;
     float mean_tot=0;

     for(int i=0;i<x_error.size();i++)
     {
      m_x=m_x+x_error[i];
      m_z=m_z+z_error[i];
      m_t=m_t+t_error[i];
     }

     mean_x=m_x/x_error.size();
      mean_z=m_z/z_error.size();
      mean_tot=m_t/t_error.size();

 		 outdata.open("allcoordinatepoints.dat"); // opens the file
   		if( !outdata ) 
   		{ // file couldn't be opened
      		//cout << "Error: file could not be opened" << endl;
      		exit(1);
   		}
   		for(int r=0;r<allpoints.rows;r++)
 		 {
 		 	for(int s=0;s<allpoints.cols;s++)
 		 	{
 		 		outdata<<allpoints.at<double>(r,s)<<" ";
 		 	}
 		 	outdata<<endl;
 		 }

 		outdata.close();
 		 

	    if(waitKey(0)==27)
            exit(0);

		return 0;
}




tuple<Mat,Mat> find2Dpoints(Mat img_1,Mat img_2,Mat img_3, Mat invH)
{

  int minHessian = 1000;

  Ptr<ORB> detector = ORB::create( minHessian );

  std::vector<KeyPoint> keypoints_1;
  std::vector<KeyPoint> keypoints_2;
  Mat descriptor,descriptor_2;
 
  detector->detect( img_1, keypoints_1 );
  detector->detect( img_2, keypoints_2 );

  detector->detectAndCompute(img_1, Mat(),keypoints_1,descriptor,true);
  detector->detectAndCompute(img_2, Mat(),keypoints_2,descriptor_2,true);
  ////cout<<"size of keypoints_1: "<<keypoints_1.size()<<endl;
  ////cout<<"size of keypoints_2: "<<keypoints_2.size()<<endl;
  
  //-- Draw keypoints
  Mat img_keypoints_1; 
  Mat img_keypoints_2;

  drawKeypoints( img_1, keypoints_1, img_keypoints_1, Scalar::all(-1), DrawMatchesFlags::DEFAULT );
  drawKeypoints( img_2, keypoints_2, img_keypoints_2, Scalar::all(-1), DrawMatchesFlags::DEFAULT );

  imshow("Keypoints 1", img_keypoints_1 );

  imshow("Keypoints 2", img_keypoints_2 );
  

  Ptr<DescriptorMatcher> matcher = DescriptorMatcher::create(DescriptorMatcher::BRUTEFORCE_L1 );

  std::vector< std::vector<DMatch> > knn_matches;
  matcher->knnMatch( descriptor, descriptor_2, knn_matches, 2 );
  //-- Filter matches using the Lowe's ratio test 
  const float ratio_thresh = 0.7f;
  std::vector<DMatch> good_matches;
  for (size_t i = 0; i < knn_matches.size(); i++)
  {
    if (knn_matches[i][0].distance < ratio_thresh * knn_matches[i][1].distance)
    {
      good_matches.push_back(knn_matches[i][0]);
    }
        
  }
  // cout << "Matches : " << good_matches.size() << endl;

  ////cout<<"Size of good matches: "<<good_matches.size()<<endl;
  
  vector<Point2f> keym1,keym2,keym3;


  for( int i = 0; i < (int)good_matches.size(); i++ )
   { //printf( "-- Good Match [%d] Keypoint 1: %d  -- Keypoint 2: %d  \n", i, good_matches[i].queryIdx, good_matches[i].trainIdx );

      
      keym1.push_back(keypoints_1.at(good_matches[i].queryIdx).pt);
      keym2.push_back(keypoints_2.at(good_matches[i].trainIdx).pt);

   }
  //  cout << keym1.size() << " " << keym2.size() << endl;
  Mat keymatches_img1=Mat(keym1);
  Mat keymatches_img2=Mat(keym2);
  Mat threedpoints;//4d homogenous coordinates
  //float projl[3][4]={{721.5377,0,609.5593,44.85728},{0,721.5377,172.854,0.216379},{0,0,1,0.00274}};
  float projl[3][4]={{718.856, 0, 607.1928, 0},{0, 718.856, 185.2157, 0},{0, 0, 1, 0}};

  Mat proj1=Mat(3,4,CV_32F, projl);
//  float projr[3][4]={{721.5377,0,609.5593,-339.5242},{0,721.5377,172.854,2.199936},{0,0,1,0.00272}};
    float projr[3][4]={{718.856, 0, 607.1928, -386.1448},{0, 718.856, 185.2157, 0},{0,0,1,0}};

  Mat proj2=Mat(3,4,CV_32F, projr);
  
  /*//cout<<"Proj1:"<<endl<<proj1<<endl;
  //cout<<"Proj2:"<<endl<<proj2<<endl;*/


  // Mat e;
  triangulatePoints(proj1, proj2,keym1,keym2,threedpoints);
  
  for(int j=0; j<threedpoints.cols;j++)
  {
    threedpoints.at<float>(0,j)=threedpoints.at<float>(0,j)/threedpoints.at<float>(3,j);
    threedpoints.at<float>(1,j)=threedpoints.at<float>(1,j)/threedpoints.at<float>(3,j);
    threedpoints.at<float>(2,j)=threedpoints.at<float>(2,j)/threedpoints.at<float>(3,j);

  }
  cout << threedpoints.col(0) << endl;
  // cout << proj1 << endl << proj2 << endl;
  Mat r;
  threedpoints(Range(0,4), Range(0,5)).copyTo(r);


  invH.convertTo(invH,CV_32F);
  //cout<<"H matrix after converting to 32F in function"<<endl<<invH<<endl;
  Mat threedpoints1;
  threedpoints.copyTo(threedpoints1);

   for(int i=0;i<threedpoints.cols;i++)
  {
    threedpoints.at<float>(0,i)=invH.at<float>(0, 0)*threedpoints1.at<float>(0,i)+invH.at<float>(0, 1)*threedpoints1.at<float>(1,i)+invH.at<float>(0, 2)*threedpoints1.at<float>(2,i)+invH.at<float>(0, 3);
    threedpoints.at<float>(1,i)=invH.at<float>(1, 0)*threedpoints1.at<float>(0,i)+invH.at<float>(1, 1)*threedpoints1.at<float>(1,i)+invH.at<float>(1, 2)*threedpoints1.at<float>(2,i)+invH.at<float>(1, 3);
    threedpoints.at<float>(2,i)=invH.at<float>(2, 0)*threedpoints1.at<float>(0,i)+invH.at<float>(2, 1)*threedpoints1.at<float>(1,i)+invH.at<float>(2, 2)*threedpoints1.at<float>(2,i)+invH.at<float>(2, 3);

  }
  Mat e;
  threedpoints(Range(0,4), Range(0,5)).copyTo(e);
  //cout<<"Threed points after multiplying with prev inv matrix: "<<endl<<e<<endl;
  

  //KLT for left image 1 left image 2
  Mat status1, err1;
  calcOpticalFlowPyrLK(img_1, img_3, keym1, keym3, status1, err1,Size(23,23), 3, TermCriteria(TermCriteria::COUNT+TermCriteria::EPS, 30, 0.01), 0, 1e-4 );

  status1.convertTo(status1,CV_32F);
  ////cout<<status1<<endl;
  vector<Point2f> twodkeypoints;
  Mat threedkeypoints(3,1,CV_32F);
  //cout<<"threedkeypoints before release: "<<threedkeypoints<<endl;
  threedkeypoints.release();
  //cout<<"threedkeypoints after release: "<<threedkeypoints<<endl;
  Mat d;
  //cout<<"Keym1 size: "<<keym1.size()<<endl<<"Keym3 size: "<<keym3.size()<<endl;//997
  ////cout<<"3Dkey points: "<<threedkeypoints<<endl;
  int ggflag=0;
  for(int i=1; i<status1.rows;i++)  // i starts from 2 because threedpoints has garbage value in the first entry
  {
    if(status1.at<float>(i,1)!=0)
    {
      twodkeypoints.push_back(keym3[i]);
      //keypointsfwd.push_back(keym3[i]);
      if(ggflag==0)
      {
        threedpoints(Range(0, threedpoints.rows-1), Range(i, i+1)).copyTo(threedkeypoints);
        ggflag=1;
      }
      else{
        threedpoints(Range(0, threedpoints.rows-1), Range(i, i+1)).copyTo(d);
      ////cout<<"size of d"<<d.size()<<endl<<"size of threedkeypoints"<<threedkeypoints.size()<<endl;
        ////cout<<"threedkeypoints size: "<<threedkeypoints.size()<<"D size: "<<d.size()<<endl;
      hconcat(threedkeypoints, d, threedkeypoints);

            }
    }
  
  }
    ////cout<<"3Dkey points after hconcat: "<<threedkeypoints<<endl;


  ////cout<<"Siz of threedkeypoints:"<<threedkeypoints.size()<<endl;// 3x994
  Mat D3=threedkeypoints.t();
  //cout<<"Siz of D3:"<<D3.size()<<endl;// 990x3
  ////cout<<"D3 is: "<<endl<<D3<<endl;

  Mat D2=Mat(twodkeypoints);

    Mat TDPoints, keymatches_img3,m,n;
    TDPoints.release();
    keymatches_img3.release();
    int fflag=0;
    //cout<<"D3 rows: "<<D3.rows<<endl;
  for(int i=0; i<D3.rows;i++)
  { 
    //cout<<"Z 3D POINTS: "<<D3.at<float>(i,2)<<endl;
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




   return make_tuple(TDPoints,keymatches_img3);

}

Mat findhomogenousmat(Mat rot_mat, Mat trans)
{
	
  Mat zer = Mat::zeros(1,3,CV_64F); 
  Mat on = Mat::ones(1,1,CV_64F);
  Mat homog,Ro,Tr;
  vconcat(rot_mat,zer,Ro);
  vconcat(trans,on,Tr);
  hconcat(Ro,Tr,homog);

  return homog;
}

vector<Point3f> groundtruth() {
      Point3f pose;    
      string line;

      int i = 0;
      ifstream myfile ("dataset/poses/00.txt");
      double x =0, y=0, z = 0, val = 0;
      double x_prev, y_prev, z_prev;
      if (myfile.is_open())
      {
        while (( getline (myfile,line) ))
        {
          z_prev = z;
          x_prev = x;
          y_prev = y;
          std::istringstream in(line);
          ////cout << line << '\n';
          for (int j=0; j<12; j++)  {
            in >> val ;
            ////cout << z << " \n";
            if (j==3) pose.x=val;
            if (j==7)  pose.y=val;
            if (j==11)  pose.z=val;

            ////cout << " Inside Vals " << x << " " << y << " " << z << " ";
          }
          /**
            //cout << x << " X value    \n ";
            //cout << y << " y valuee \n ";
            //cout << z << " z value    \n ";
          **/
          i++;
          gt_poses.push_back(pose);
        }
        myfile.close();
      }
        return gt_poses;
    }
