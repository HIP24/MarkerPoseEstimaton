#include <iostream>
#include <opencv2/calib3d.hpp>		// For chessboard corner detection
#include <opencv2/ml.hpp>			// For cv::Ptr<cv::ml::TrainData>
#include <opencv2/imgproc.hpp>		// For line drawing
#include <opencv2/highgui.hpp>		// For general cv
#include <fstream>					// For file stream csv
#include <opencv2/calib3d.hpp>		// For solvePnP

class poseEstimate{
	public:
		// Constructor
		poseEstimate(const std::vector<cv::Point3f>& activeSetXYZ, const cv::Mat& cameraMatrix, 
			const cv::Mat& distCoeffs):activeSetXYZ_(activeSetXYZ),cameraMatrix_(cameraMatrix), distCoeffs_(distCoeffs){}
		// Destructor
		~poseEstimate(){}
		
		// Getter
		cv::Mat getorginalImage() const{return objectImage;}
		cv::Mat getImageout() const{return imageoutput;}
		cv::Mat getcameraMatrix() const{return cameraMatrix_;}
		cv::Mat getdistCoeff() const{return distCoeffs_;}
		std::vector<cv::Point2f> getCornerXY() const{return corners;}
		std::vector<cv::Point3f> getActiveSetXYZ() const{return activeSetXYZ_;}

		void findChessboardCorners(const cv::Mat& input){
			cv::Size patternsize(9,6); //interior number of corners
			cv::Mat image_gray; //source image
			objectImage = input.clone();
			cv::cvtColor(objectImage, image_gray, CV_BGR2GRAY);	// Convert to Gray
			//CALIB_CB_FAST_CHECK saves a lot of time on images
			//that do not contain any chessboard corners
			bool patternfound = cv::findChessboardCorners(image_gray, patternsize, corners,
					cv::CALIB_CB_ADAPTIVE_THRESH + cv::CALIB_CB_NORMALIZE_IMAGE
					+ cv::CALIB_CB_FAST_CHECK);
			if(patternfound)
			imageoutput = objectImage.clone();
			cv::drawChessboardCorners(imageoutput, patternsize, cv::Mat(corners), patternfound);
		}

		void calculateTransRot(){
			// Call solvePnP()
			if(activeSetXYZ_.size() == corners.size()){
				cv::solvePnP(activeSetXYZ_, corners, cameraMatrix_, distCoeffs_, rvec, tvec);
			}else{
				std::cout << "Not the same size!" << std::endl;
			}
			
			/*std::cout << "Distance to Object: " << tvec.at<double>(2) << std::endl;
			std::cout << "x deviation: " << tvec.at<double>(0) << std::endl;
			std::cout << "y deviation: " << tvec.at<double>(1) << std::endl << std::endl;*/
		}
		void displayTransRot(const cv::Mat& out){
			cv::Mat frame = out.clone();
			std::string distanceText = "Distance to Object: " + std::to_string(tvec.at<double>(2));
			std::string xDeviationText = "x deviation: " + std::to_string(tvec.at<double>(0));
			std::string yDeviationText = "y deviation: " + std::to_string(tvec.at<double>(1));

			int fontFace = cv::FONT_HERSHEY_SIMPLEX;
			double fontScale = 0.5;
			int thickness = 1;
			int baseline = 0;
			cv::Size distanceTextSize = cv::getTextSize(distanceText, fontFace, fontScale, thickness, &baseline);
			cv::Size deviationTextSize = cv::getTextSize(xDeviationText, fontFace, fontScale, thickness, &baseline);

			int padding = 10;
			int y = padding + distanceTextSize.height;
			// Display deviation values on the left side of the frame
			cv::putText(frame, distanceText, cv::Point(padding, y), fontFace, fontScale, cv::Scalar(255, 255, 255), thickness);
			y += distanceTextSize.height + padding;
			cv::putText(frame, xDeviationText, cv::Point(padding, y), fontFace, fontScale, cv::Scalar(255, 255, 255), thickness);
			y += deviationTextSize.height + padding;
			cv::putText(frame, yDeviationText, cv::Point(padding, y), fontFace, fontScale, cv::Scalar(255, 255, 255), thickness);

			cv::imshow("CheesboardCorner", frame);

		}

	private:
		std::vector<cv::Point3f> activeSetXYZ_;
		std::vector<cv::Point2f> corners;
		cv::Mat cameraMatrix_, distCoeffs_, objectImage, imageoutput;
		// Define the rotation and translation vectors
		cv::Mat rvec, tvec;
};



int main(){
	cv::VideoCapture video;
	cv::Mat vidImg, camera_matrix, distortion_coeffs, vidImgUndistort;
	std::vector<cv::Point3f> activeSetXYZ;

	video.open("data/video.mp4");
	if (!video.isOpened()){
		std::cerr << "Could not open video!" << std::endl;
		exit(-1);
	}
	// Open calibration YAML file
	cv::FileStorage fs1("data/camera_params.yaml", cv::FileStorage::READ);
	if (!fs1.isOpened()){
		std::cerr << "Failed to open camera_params.yaml" << std::endl;
		exit(-1);
	}
	// Read the camera parameters
	fs1["camera_matrix"] >> camera_matrix;
	fs1["distortion_coefficients"] >> distortion_coeffs;
	// Release the file storage object and close the file
	fs1.release();

	// Load CSV file with real 3D measurments
	cv::Ptr<cv::ml::TrainData> tdata = cv::ml::TrainData::loadFromCSV("data/activeSet_XYZ.csv",0,0,1);
	cv::Mat samples = tdata->getSamples();
	for(int i = 0; i < samples.rows; i++){
		cv::Point3f temp(samples.at<float>(i,0), samples.at<float>(i,1), samples.at<float>(i,2));
		activeSetXYZ.push_back(temp);
	}

	// Init poste Estimate
	poseEstimate ePose(activeSetXYZ, camera_matrix, distortion_coeffs);
	key_t key;
	while(key != 27){                   // Do till <ESC> was pressed
        key = cv::waitKey(5);                  // Update window
		video >> vidImg;                     // Get image from
        if (vidImg.empty()){
			video.set(cv::CAP_PROP_POS_FRAMES, 0);
			video >> vidImg;
			// Exit the loop if the video is still empty
			if(vidImg.empty())	exit(-1);	
		}
		// Smartphone Camera was used 1280x720. Calibration tool: https://www.camcalib.io/
		cv::undistort(vidImg, vidImgUndistort, camera_matrix, distortion_coeffs);
		ePose.findChessboardCorners(vidImgUndistort);
		cv::Mat out = ePose.getImageout();
		// Resizing otherwise it would take too long to display every frame
		std::vector<cv::Point2f> corners = ePose.getCornerXY();
		cv::line(out,cv::Point_<int>(corners[53].x,corners[53].y),cv::Point_<int>(corners[51].x,corners[51].y),cv::Scalar(0,255,0),8);
		cv::line(out,cv::Point_<int>(corners[53].x,corners[53].y),cv::Point_<int>(corners[35].x,corners[35].y),cv::Scalar(0,0,255),8);
		cv::resize(out, out, cv::Size(out.cols/2, out.rows/2));
		if(!ePose.getCornerXY().empty()){
			ePose.calculateTransRot();
			ePose.displayTransRot(out);
		}
		//std::cout << "Corners: " << ePose.getCornerXY() << std::endl;
		
    }
	return 0;
}