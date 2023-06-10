#include <iostream>
#include <opencv2/imgproc.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/xfeatures2d/nonfree.hpp>
#include <opencv2/opencv.hpp>
#include <vector>
#include <numeric>

// Function to compute the homography using RANSAC
cv::Mat findHomographyOwnRANSAC(const std::vector<cv::Point2f> &srcPoints,
                                const std::vector<cv::Point2f> &dstPoints,
                                int maxIterations = 1000,
                                double distanceThreshold = 3.0,
                                double confidence = 0.99)
{
    // Number of points
    int numPoints = srcPoints.size();

    // Best homography
    cv::Mat bestH;

    // Best number of inliers
    int bestNumInliers = 0;

    // Random number generator
    cv::RNG rng;

    // RANSAC iterations
    for (int i = 0; i < maxIterations; i++)
    {
        // Randomly select 4 points
        std::vector<int> indices(numPoints);
        std::iota(indices.begin(), indices.end(), 0);
        std::random_shuffle(indices.begin(), indices.end(), rng);
        std::vector<cv::Point2f> srcPointsSubset = {srcPoints[indices[0]], srcPoints[indices[1]], srcPoints[indices[2]], srcPoints[indices[3]]};
        std::vector<cv::Point2f> dstPointsSubset = {dstPoints[indices[0]], dstPoints[indices[1]], dstPoints[indices[2]], dstPoints[indices[3]]};

        // Compute the homography using the selected points
        cv::Mat A(8, 9, CV_64F);
        for (int j = 0; j < 4; j++)
        {
            float X = srcPointsSubset[j].x;
            float Y = srcPointsSubset[j].y;
            float u = dstPointsSubset[j].x;
            float v = dstPointsSubset[j].y;

            A.at<double>(2 * j, 0) = -X;
            A.at<double>(2 * j, 1) = -Y;
            A.at<double>(2 * j, 2) = -1.0;
            A.at<double>(2 * j, 6) = u * X;
            A.at<double>(2 * j, 7) = u * Y;
            A.at<double>(2 * j, 8) = u;

            A.at<double>(2 * j + 1, 3) = -X;
            A.at<double>(2 * j + 1, 4) = -Y;
            A.at<double>(2 * j + 1, 5) = -1.0;
            A.at<double>(2 * j + 1, 6) = v * X;
            A.at<double>(2 * j + 1, 7) = v * Y;
            A.at<double>(2 * j + 1, 8) = v;
        }

        cv::Mat U, D, Vt;
        cv::SVDecomp(A, U, D, Vt, cv::SVD::MODIFY_A | cv::SVD::FULL_UV);
        cv::Mat H = Vt.row(Vt.rows - 1).reshape(0, 3);

        // Count the number of inliers
        int numInliers = 0;
        for (int j = 0; j < numPoints; j++)
        {
            cv::Mat srcPoint = (cv::Mat_<double>(3, 1) << srcPoints[j].x, srcPoints[j].y, 1.0);
            cv::Mat dstPoint = (cv::Mat_<double>(3, 1) << dstPoints[j].x, dstPoints[j].y, 1.0);
            cv::Mat projectedPoint = H * srcPoint;
            projectedPoint /= projectedPoint.at<double>(2);
            double distance = cv::norm(projectedPoint - dstPoint);
            if (distance < distanceThreshold)
            {
                numInliers++;
            }
        }

        // Update the best homography
        if (numInliers > bestNumInliers)
        {
            bestH = H;
            bestNumInliers = numInliers;
        }

        // Update the number of iterations based on the inlier ratio
        double inlierRatio = static_cast<double>(numInliers) / numPoints;
        double logConfidence = log(1.0 - confidence);
        double logOneMinusInlierRatio = log(1.0 - pow(inlierRatio, 4));
        maxIterations = static_cast<int>(logConfidence / logOneMinusInlierRatio);
    }

    // Refine the homography using all inliers
    std::vector<cv::Point2f> srcPointsInliers, dstPointsInliers;
    for (int j = 0; j < numPoints; j++)
    {
        cv::Mat srcPoint = (cv::Mat_<double>(3, 1) << srcPoints[j].x, srcPoints[j].y, 1.0);
        cv::Mat dstPoint = (cv::Mat_<double>(3, 1) << dstPoints[j].x, dstPoints[j].y, 1.0);
        cv::Mat projectedPoint = bestH * srcPoint;
        projectedPoint /= projectedPoint.at<double>(2);
        double distance = cv::norm(projectedPoint - dstPoint);
        if (distance < distanceThreshold)
        {
            srcPointsInliers.push_back(srcPoints[j]);
            dstPointsInliers.push_back(dstPoints[j]);
        }
    }
    bestH = cv::findHomography(srcPointsInliers, dstPointsInliers);

    return bestH;
}

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


int main()
{
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
		//cv::undistort(vidImg, vidImgUndistort, camera_matrix, distortion_coeffs);
		ePose.findChessboardCorners(vidImg);
		cv::Mat out = ePose.getImageout();
		// Resizing otherwise it would take too long to display every frame
		std::vector<cv::Point2f> corners = ePose.getCornerXY();
		//cv::line(out,cv::Point_<int>(corners[53].x,corners[53].y),cv::Point_<int>(corners[51].x,corners[51].y),cv::Scalar(0,255,0),8);
		//cv::line(out,cv::Point_<int>(corners[53].x,corners[53].y),cv::Point_<int>(corners[35].x,corners[35].y),cv::Scalar(0,0,255),8);
		cv::resize(out, out, cv::Size(out.cols/2, out.rows/2));
		
		if(!ePose.getCornerXY().empty() && ePose.getActiveSetXYZ().size() == ePose.getCornerXY().size()){
			ePose.calculateTransRot();
			ePose.displayTransRot(out);
		}else{
			cv::imshow("CheesboardCorner", out);
		}
		//std::cout << "Corners: " << ePose.getCornerXY() << std::endl;
		
    }

    // Calculate the homography using RANSAC
    cv::Mat homography = findHomographyOwnRANSAC(matched_pts1, matched_pts2);



    // Warp the first image to align with the second image
    cv::Mat warped_img1;
    cv::warpPerspective(image1, warped_img1, homography, image2.size());

    // Create a canvas to combine the two images
    cv::Mat canvas(image2.rows, image2.cols + image1.cols, CV_8UC3);
    image2.copyTo(canvas(cv::Rect(0, 0, image2.cols, image2.rows)));
    warped_img1.copyTo(canvas(cv::Rect(image2.cols, 0, image1.cols, image1.rows)));

    // Draw correspondence lines between the matched points
    for (size_t i = 0; i < matched_pts1.size(); i++)
    {
        cv::Point2f pt1 = matched_pts1[i];
        cv::Point2f pt2 = matched_pts2[i] + cv::Point2f(image2.cols, 0);
        cv::line(canvas, pt1, pt2, cv::Scalar(0, 255, 0), 2);
    }

    // Save the result
    cv::imwrite("data/RANSAC_own.jpg", canvas);

    // Display the result
    cv::imshow("RANSAC", canvas);
    cv::waitKey(0);

    return 0;
}
