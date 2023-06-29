#include <iostream>
#include <opencv2/calib3d.hpp>		// For chessboard corner detection
#include <opencv2/ml.hpp>			// For cv::Ptr<cv::ml::TrainData>
#include <opencv2/imgproc.hpp>		// For line drawing
#include <opencv2/highgui.hpp>		// For general cv
#include <fstream>					// For file stream csv
#include <opencv2/calib3d.hpp>		// For solvePnP
#include <numeric>					// For sdt::iota

class poseEstimateSolvePnP{
	public:
		// Constructor
		poseEstimateSolvePnP(const std::vector<cv::Point3f>& activeSetXYZ, const cv::Mat& cameraMatrix, 
			const cv::Mat& distCoeffs):activeSetXYZ_(activeSetXYZ),cameraMatrix_(cameraMatrix), distCoeffs_(distCoeffs){}
		// Destructor
		~poseEstimateSolvePnP(){}
		
		// Getter
		cv::Mat getorginalImage() const{return objectImage;}
		cv::Mat getImageout() const{return imageoutput;}
		cv::Mat getcameraMatrix() const{return cameraMatrix_;}
		cv::Mat getdistCoeff() const{return distCoeffs_;}
		std::vector<cv::Point2f> getCornerXY() const{return corners;}
		std::vector<cv::Point3f> getActiveSetXYZ() const{return activeSetXYZ_;}
		cv::Mat getTvec() const{return tvec;}
		cv::Mat getRvec() const{return rvec;}

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
			imageoutput = objectImage.clone();
			if(patternfound){
				cv::drawChessboardCorners(imageoutput, patternsize, cv::Mat(corners), patternfound);
			}
		}

		void calculateTransRot(){
			// Call solvePnP()
			if(activeSetXYZ_.size() == corners.size()){
				cv::solvePnP(activeSetXYZ_, corners, cameraMatrix_, distCoeffs_, rvec, tvec);
			}else{
				std::cout << "Not the same size!" << std::endl;
			}
		}
		void displayTransRot(const cv::Mat& out){
			cv::Mat frame = out.clone();
			std::string distanceText = "Distance to Object: " + std::to_string(tvec.at<double>(2));
			std::string xDeviationText = "x deviation: " + std::to_string(tvec.at<double>(0));
			std::string yDeviationText = "y deviation: " + std::to_string(tvec.at<double>(1));
			std::string rotx = "x rotation: " + std::to_string(rvec.at<double>(0));
			std::string roty = "y rotation: " + std::to_string(rvec.at<double>(1));
			std::string rotz = "z rotation: " + std::to_string(rvec.at<double>(2));

			int fontFace = cv::FONT_HERSHEY_SIMPLEX;
			double fontScale = 0.7;
			int thickness = 2;
			int baseline = 0;
			cv::Size distanceTextSize = cv::getTextSize(distanceText, fontFace, fontScale, thickness, &baseline);
			cv::Size deviationTextSize = cv::getTextSize(xDeviationText, fontFace, fontScale, thickness, &baseline);

			int padding = 10;
			int y = padding + distanceTextSize.height;
			// Display deviation values on the left side of the frame
			cv::putText(frame, distanceText, cv::Point(padding, y), fontFace, fontScale, cv::Scalar(0, 0, 0), thickness);
			y += distanceTextSize.height + padding;
			cv::putText(frame, xDeviationText, cv::Point(padding, y), fontFace, fontScale, cv::Scalar(0, 0, 0), thickness);
			y += deviationTextSize.height + padding;
			cv::putText(frame, yDeviationText, cv::Point(padding, y), fontFace, fontScale, cv::Scalar(0, 0, 0), thickness);
			y += deviationTextSize.height + padding;
			cv::putText(frame, rotx, cv::Point(padding, y), fontFace, fontScale, cv::Scalar(0, 0, 0), thickness);
			y += deviationTextSize.height + padding;
			cv::putText(frame, roty, cv::Point(padding, y), fontFace, fontScale, cv::Scalar(0, 0, 0), thickness);
			y += deviationTextSize.height + padding;
			cv::putText(frame, rotz, cv::Point(padding, y), fontFace, fontScale, cv::Scalar(0, 0, 0), thickness);
			// Generate Coordinatesystem
			
			std::vector<cv::Point3f> axisPoints = {cv::Point3f(0, 0, 0), cv::Point3f(0.1, 0, 0), cv::Point3f(0, 0.1, 0), cv::Point3f(0, 0, 0.1)};
			// Project the 3D axis points onto the image plane
			std::vector<cv::Point2f> projectedPoints;
			cv::projectPoints(axisPoints, rvec, tvec, cameraMatrix_, distCoeffs_, projectedPoints);
			// Draw the coordinate axes on the frame
			cv::line(frame, projectedPoints[0], projectedPoints[1], cv::Scalar(0, 0, 255), 3);
			cv::line(frame, projectedPoints[0], projectedPoints[2], cv::Scalar(0, 255, 0), 3);
			cv::line(frame, projectedPoints[0], projectedPoints[3], cv::Scalar(255, 0, 0), 3);
			cv::resize(frame, frame, cv::Size(frame.cols/1.2, frame.rows/1.2));
			cv::imshow("CheesboardCornersolvePnP", frame);
		}

	private:
		std::vector<cv::Point3f> activeSetXYZ_;
		std::vector<cv::Point2f> corners;
		cv::Mat cameraMatrix_, distCoeffs_, objectImage, imageoutput;
		// Define the rotation and translation vectors
		cv::Mat rvec, tvec;
};

class poseEstimateRansac{
	public:
		// Constructor
		poseEstimateRansac(const std::vector<cv::Point3f>& activeSetXYZ, const cv::Mat& cameraMatrix, 
				const cv::Mat& distCoeffs):activeSetXYZ_(activeSetXYZ),cameraMatrix_(cameraMatrix), distCoeffs_(distCoeffs){
					cameraMatrix_.convertTo(cameraMatrix_, CV_64F);
				}
		// Destructor
		~poseEstimateRansac(){}

		// Getter
		cv::Mat getImageout() const{return imageoutput;}
		std::vector<cv::Point2f> getCornerXY() const{return corners;}
		std::vector<cv::Point3f> getActiveSetXYZ() const{return activeSetXYZ_;}
		cv::Mat getTvec() const{return tvec_;}
		cv::Mat getRvec() const{return rvec_;}

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
			imageoutput = objectImage.clone();
			if(patternfound){
				cv::drawChessboardCorners(imageoutput, patternsize, cv::Mat(corners), patternfound);
			}
		}

		bool solvePnPOwn(const std::vector<cv::Point3f>& objectPoints,
                 const std::vector<cv::Point2f>& imagePoints,
                 cv::Mat& rvec,
                 cv::Mat& tvec){
			// Check if the number of points is at least 4
			if (objectPoints.size() < 4 || imagePoints.size() < 4){
				std::cout << "Insufficient number of points to compute homography!" << std::endl;
				return false;
			}

			// Construct the matrix A
			cv::Mat A(2 * objectPoints.size(), 9, CV_64F);
			for (size_t i = 0; i < objectPoints.size(); i++){
				const cv::Point3f &X = objectPoints[i];
				const cv::Point2f &x = imagePoints[i];

				A.at<double>(2 * i, 0) = -X.x;
				A.at<double>(2 * i, 1) = -X.y;
				A.at<double>(2 * i, 2) = -1;
				A.at<double>(2 * i, 3) = 0;
				A.at<double>(2 * i, 4) = 0;
				A.at<double>(2 * i, 5) = 0;
				A.at<double>(2 * i, 6) = x.x * X.x;
				A.at<double>(2 * i, 7) = x.x * X.y;
				A.at<double>(2 * i, 8) = x.x;

				A.at<double>(2 * i + 1, 0) = 0;
				A.at<double>(2 * i + 1, 1) = 0;
				A.at<double>(2 * i + 1, 2) = 0;
				A.at<double>(2 * i + 1, 3) = -X.x;
				A.at<double>(2 * i + 1, 4) = -X.y;
				A.at<double>(2 * i + 1, 5) = -1;
				A.at<double>(2 * i + 1, 6) = x.y * X.x;
				A.at<double>(2 * i + 1, 7) = x.y * X.y;
				A.at<double>(2 * i + 1, 8) = x.y;
			}

			// Perform singular value decomposition (SVD) on matrix A
			cv::Mat u, w, vt;
			cv::SVD::compute(A, w, u, vt);
			
			// Extract the column of V corresponding to the smallest singular value
			cv::Mat h = vt.row(vt.rows - 1).t();
			
			// Normalize the homography vector
			h /= cv::norm(h);
			
			// Reshape the normalized vector into a 3x3 matrix
			cv::Mat H(3, 3, CV_64F);
			for (int i = 0; i < 3; i++){
				for (int j = 0; j < 3; j++){
					H.at<double>(i, j) = h.at<double>(3 * i + j);
				}
			}
			// Compute the camera pose from the homography
			cv::Mat Kinv = cameraMatrix_.inv();
			cv::Mat h1 = H.col(0);
			cv::Mat h2 = H.col(1);
			cv::Mat h3 = H.col(2);
			cv::Mat r1 = Kinv * h1;
			cv::Mat r2 = Kinv * h2;
			cv::Mat r3 = r1.cross(r2);
			cv::Mat t = Kinv * h3;
			double norm1 = cv::norm(r1);
			double norm2 = cv::norm(r2);
			double tnorm = (norm1 + norm2) / 2.0;
			r1 /= norm1;
			r2 /= norm2;
			t /= tnorm;

			// Create the rotation matrix
			cv::Mat R(3, 3, CV_64F);
			for (int i = 0; i < 3; i++){
				R.at<double>(i, 0) = r1.at<double>(i, 0);
				R.at<double>(i, 1) = r2.at<double>(i, 0);
				R.at<double>(i, 2) = r3.at<double>(i, 0);
			}

			// Convert the rotation matrix to a rotation vector
			cv::Rodrigues(R, rvec);
			// Set the output translation vector
			tvec = t;
			return true;
		}
		bool solvePnPRansacOwn(int iterationsCount = 100, float reprojectionError = 8.0){
			int numPoints = activeSetXYZ_.size();
			int bestNumInliers = 0;
			cv::Mat bestRvec, bestTvec;

			for (int i = 0; i < iterationsCount; i++){
				// Randomly select a subset of points
				std::vector<int> indices(numPoints);
				std::iota(indices.begin(), indices.end(), 0);
				std::random_shuffle(indices.begin(), indices.end());
				std::vector<cv::Point3f> objectPointsSubset = {activeSetXYZ_[indices[0]], activeSetXYZ_[indices[1]], activeSetXYZ_[indices[2]], activeSetXYZ_[indices[3]]};
				std::vector<cv::Point2f> imagePointsSubset = {corners[indices[0]], corners[indices[1]], corners[indices[2]], corners[indices[3]]};

				// Estimate the pose using the selected subset of points
				cv::Mat rvecSubset, tvecSubset;
				bool success = solvePnPOwn(objectPointsSubset, imagePointsSubset, rvecSubset, tvecSubset);
				// Count the number of inliers
				if(success){
					int numInliers = 0;
					for (int j = 0; j < numPoints; j++){
						std::vector<cv::Point3f> objectPoint = {activeSetXYZ_[j]};
						std::vector<cv::Point2f> projectedPoint;
						cv::projectPoints(objectPoint, rvecSubset, tvecSubset, cameraMatrix_, distCoeffs_, projectedPoint);
						double distance = cv::norm(projectedPoint[0] - corners[j]);
						if (distance < reprojectionError){
							numInliers++;
						}
					}
					// Update the best pose
					if (numInliers > bestNumInliers){
						bestRvec = rvecSubset;
						bestTvec = tvecSubset;
						bestNumInliers = numInliers;
					}
				}
			}

			// Set the output pose to the best pose
			rvec_ = bestRvec;
			tvec_ = bestTvec;
			if(bestNumInliers > 0){
				return true;
			}
			return false;
		}

		void displayTransRot(const cv::Mat& out){
			cv::Mat frame = out.clone();
			std::string distanceText = "Distance to Object: " + std::to_string(tvec_.at<double>(2));
			std::string xDeviationText = "x deviation: " + std::to_string(tvec_.at<double>(0));
			std::string yDeviationText = "y deviation: " + std::to_string(tvec_.at<double>(1));
			std::string rotx = "x rotation: " + std::to_string(rvec_.at<double>(0));
			std::string roty = "y rotation: " + std::to_string(rvec_.at<double>(1));
			std::string rotz = "z rotation: " + std::to_string(rvec_.at<double>(2));

			int fontFace = cv::FONT_HERSHEY_SIMPLEX;
			double fontScale = 0.7;
			int thickness = 2;
			int baseline = 0;
			cv::Size distanceTextSize = cv::getTextSize(distanceText, fontFace, fontScale, thickness, &baseline);
			cv::Size deviationTextSize = cv::getTextSize(xDeviationText, fontFace, fontScale, thickness, &baseline);

			int padding = 10;
			int y = padding + distanceTextSize.height;
			// Display deviation values on the left side of the frame
			cv::putText(frame, distanceText, cv::Point(padding, y), fontFace, fontScale, cv::Scalar(0, 0, 0), thickness);
			y += distanceTextSize.height + padding;
			cv::putText(frame, xDeviationText, cv::Point(padding, y), fontFace, fontScale, cv::Scalar(0, 0, 0), thickness);
			y += deviationTextSize.height + padding;
			cv::putText(frame, yDeviationText, cv::Point(padding, y), fontFace, fontScale, cv::Scalar(0, 0, 0), thickness);
			y += deviationTextSize.height + padding;
			cv::putText(frame, rotx, cv::Point(padding, y), fontFace, fontScale, cv::Scalar(0, 0, 0), thickness);
			y += deviationTextSize.height + padding;
			cv::putText(frame, roty, cv::Point(padding, y), fontFace, fontScale, cv::Scalar(0, 0, 0), thickness);
			y += deviationTextSize.height + padding;
			cv::putText(frame, rotz, cv::Point(padding, y), fontFace, fontScale, cv::Scalar(0, 0, 0), thickness);
			// Generate Coordinatesystem
			
			std::vector<cv::Point3f> axisPoints = {cv::Point3f(0, 0, 0), cv::Point3f(0.1, 0, 0), cv::Point3f(0, 0.1, 0), cv::Point3f(0, 0, 0.1)};
			// Project the 3D axis points onto the image plane
			std::vector<cv::Point2f> projectedPoints;
			cv::projectPoints(axisPoints, rvec_, tvec_, cameraMatrix_, distCoeffs_, projectedPoints);
			// Draw the coordinate axes on the frame
			cv::line(frame, projectedPoints[0], projectedPoints[1], cv::Scalar(0, 0, 255), 3);
			cv::line(frame, projectedPoints[0], projectedPoints[2], cv::Scalar(0, 255, 0), 3);
			cv::line(frame, projectedPoints[0], projectedPoints[3], cv::Scalar(255, 0, 0), 3);
			cv::resize(frame, frame, cv::Size(frame.cols/1.2, frame.rows/1.2));
			cv::imshow("CheesboardCornerRansac", frame);
		}
		

	private:
		std::vector<cv::Point3f> activeSetXYZ_;
		std::vector<cv::Point2f> corners;
		cv::Mat cameraMatrix_, distCoeffs_, objectImage, imageoutput;
		// Define the rotation and translation vectors
		cv::Mat rvec_, tvec_;
};

void printError(cv::Mat solvepnpTvec, cv::Mat solvepnpRvec, cv::Mat ransacTvec, cv::Mat ransacRvec){
	std::cout << "Error between (solvepnpTvec - ransacTvec): " << std::endl << solvepnpTvec - ransacTvec << 
		std::endl << "Error between (solvepnpRvec - ransacRvec): " << std::endl << solvepnpRvec - ransacRvec << std::endl;
}


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
	poseEstimateSolvePnP ePose(activeSetXYZ, camera_matrix, distortion_coeffs);
    poseEstimateRansac ePose2(activeSetXYZ, camera_matrix, distortion_coeffs);
	key_t key;
	// Open the CSV file for writing
	std::ofstream file("data/Error.csv");
	std::string  string1 = "Error tvec";
	std::string  string2 = "Error rvec";
    file << string1 << "," << string2 << std::endl;
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
		// FOR solvePnP
		cv::undistort(vidImg, vidImgUndistort, camera_matrix, distortion_coeffs);
		ePose.findChessboardCorners(vidImgUndistort);
		cv::Mat out = ePose.getImageout();
		if(!ePose.getCornerXY().empty() && ePose.getActiveSetXYZ().size() == ePose.getCornerXY().size()){
			ePose.calculateTransRot();
			ePose.displayTransRot(out);
		}else{
			cv::resize(out, out, cv::Size(out.cols/1.2, out.rows/1.2));
			cv::imshow("CheesboardCornersolvePnP", out);
		}
		// FOR Ransac
		ePose2.findChessboardCorners(vidImg);
		cv::Mat out2 = ePose.getImageout();
		if(!ePose2.getCornerXY().empty() && ePose2.getActiveSetXYZ().size() == ePose2.getCornerXY().size()){
			bool success = ePose2.solvePnPRansacOwn(500,4.0);
			// Draw the coordinate axes on the frame
            if (success){
				ePose2.displayTransRot(out2);  
			}
		}else{
			cv::resize(out2, out2, cv::Size(out2.cols/1.2, out2.rows/1.2));
			cv::imshow("CheesboardCornerRansac", out2);
		}
		// Print Error between solvePnP and own Ransac
		printError(ePose.getTvec(), ePose.getRvec(), ePose2.getTvec(), ePose2.getRvec());
		cv::Mat t = ePose.getTvec() - ePose2.getTvec();
		cv::Mat r = ePose.getRvec() - ePose2.getRvec();
		for(signed int i = 0; i < t.rows; i++){
			file << t.at<double>(i) << "," << r.at<double>(i) << std::endl;
		}
		file << std::endl;
		//file << ePose.getTvec()-ePose2.getTvec() << ";" << ePose.getRvec()-ePose2.getRvec() << std::endl;
	}
	// Release the video capture
	video.release();
	file.close();
	return 0;
}