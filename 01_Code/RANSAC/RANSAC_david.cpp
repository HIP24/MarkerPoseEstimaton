#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/xfeatures2d/nonfree.hpp>
#include <opencv2/opencv.hpp>
#include <vector>
#include <numeric>

// Estimate pose using your own implementation
void estimatePose(const std::vector<cv::Point3f>& objectPoints,
                  const std::vector<cv::Point2f>& imagePoints,
                  const cv::Mat& cameraMatrix,
                  const cv::Mat& distCoeffs,
                  cv::Mat& rvec,
                  cv::Mat& tvec) {
    // Convert object points to homogeneous coordinates
    std::vector<cv::Point3f> objectPointsHomogeneous;
    for (const auto& point : objectPoints) {
        objectPointsHomogeneous.push_back(cv::Point3f(point.x, point.y, point.z));
    }

    // Normalize image points
    std::vector<cv::Point2f> normalizedImagePoints;
    cv::undistortPoints(imagePoints, normalizedImagePoints, cameraMatrix, distCoeffs);

    // Number of points
    const int numPoints = objectPoints.size();

    // Build the equation system: AX = 0
    cv::Mat A(2 * numPoints, 12, CV_64F);
    for (int i = 0; i < numPoints; ++i) {
        const cv::Point3f& objectPoint = objectPointsHomogeneous[i];
        const cv::Point2f& imagePoint = normalizedImagePoints[i];

        const double X = objectPoint.x;
        const double Y = objectPoint.y;
        const double Z = objectPoint.z;
        const double u = imagePoint.x;
        const double v = imagePoint.y;

        A.at<double>(2 * i, 0) = 0.0;
        A.at<double>(2 * i, 1) = 0.0;
        A.at<double>(2 * i, 2) = 0.0;
        A.at<double>(2 * i, 3) = 0.0;
        A.at<double>(2 * i, 4) = -X;
        A.at<double>(2 * i, 5) = -Y;
        A.at<double>(2 * i, 6) = -Z;
        A.at<double>(2 * i, 7) = -1.0;
        A.at<double>(2 * i, 8) = v * X;
        A.at<double>(2 * i, 9) = v * Y;
        A.at<double>(2 * i, 10) = v * Z;
        A.at<double>(2 * i, 11) = v;

        A.at<double>(2 * i + 1, 0) = X;
        A.at<double>(2 * i + 1, 1) = Y;
        A.at<double>(2 * i + 1, 2) = Z;
        A.at<double>(2 * i + 1, 3) = 1.0;
        A.at<double>(2 * i + 1, 4) = 0.0;
        A.at<double>(2 * i + 1, 5) = 0.0;
        A.at<double>(2 * i + 1, 6) = 0.0;
        A.at<double>(2 * i + 1, 7) = 0.0;
        A.at<double>(2 * i + 1, 8) = -u * X;
        A.at<double>(2 * i + 1, 9) = -u * Y;
        A.at<double>(2 * i + 1, 10) = -u * Z;
        A.at<double>(2 * i + 1, 11) = -u;
    }

    // Perform SVD on A
    cv::Mat U, D, Vt;
    cv::SVD::compute(A, D, U, Vt, cv::SVD::FULL_UV);

    // Extract the last column of V
    cv::Mat X = Vt.row(Vt.rows - 1).t();

    // Extract the estimated rotation and translation
    cv::Mat R(3, 3, CV_64F);
    cv::Mat t(3, 1, CV_64F);
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            R.at<double>(i, j) = X.at<double>(3 * i + j, 0);
        }
        t.at<double>(i, 0) = X.at<double>(9 + i, 0);
    }

    // Normalize rotation matrix
    cv::Mat w, u, vt;
    cv::SVD::compute(R, w, u, vt);
    R = u * vt;

    // Output rotation and translation vectors
    rvec.create(3, 1, CV_64F);
    tvec.create(3, 1, CV_64F);

    cv::Rodrigues(R, rvec);
    tvec = t;
}


void solveRANSAC(const std::vector<cv::Point3f>& objectPoints,
                    const std::vector<cv::Point2f>& imagePoints,
                    const cv::Mat& cameraMatrix,
                    const cv::Mat& distCoeffs,
                    cv::Mat& bestRvec,
                    cv::Mat& bestTvec,
                    int maxIterations = 1000,
                    double distanceThreshold = 3.0){
    // Number of points
    int numPoints = objectPoints.size();

    // Best number of inliers
    int bestNumInliers = 0;

    // Random number generator
    cv::RNG rng;

    // RANSAC iterations
    for (int i = 0; i < maxIterations; i++){
        // Randomly select 4 points
        std::vector<int> indices(numPoints);
        std::iota(indices.begin(), indices.end(), 0);
        std::random_shuffle(indices.begin(), indices.end(), rng);
        std::vector<cv::Point3f> objectPointsSubset = {objectPoints[indices[0]], objectPoints[indices[1]], objectPoints[indices[2]], objectPoints[indices[3]]};
        std::vector<cv::Point2f> imagePointsSubset = {imagePoints[indices[0]], imagePoints[indices[1]], imagePoints[indices[2]], imagePoints[indices[3]]};
        cv::Mat rvec, tvec;
        estimatePose(objectPointsSubset, imagePointsSubset, cameraMatrix, distCoeffs, rvec, tvec);
        
        // Calculate reprojection error and count inliers
        int inlierCount = 0;
        for (int j = 0; j < numPoints; j++) {
            // Project 3D object point to 2D image point
            std::vector<cv::Point3f> singleObjectPoint = {objectPoints[j]};
            std::vector<cv::Point2f> singleImagePoint = {imagePoints[j]};
            std::vector<cv::Point2f> projectedPoints;
            cv::projectPoints(singleObjectPoint, rvec, tvec, cameraMatrix, distCoeffs, projectedPoints);
            // Calculate Euclidean distance between projected point and image point
            double dist = cv::norm(cv::Mat(projectedPoints[0]), cv::Mat(singleImagePoint[0]), cv::NORM_L2);
            std::cout << dist << std::endl;

            // Check if point is an inlier
            if (dist <= distanceThreshold){
                std::cout << "test!!!" << std::endl;
                inlierCount++;
            }
        }
        // Compute reprojection error and count inliers
        /*int inlierCount = 0;
        std::vector<cv::Point2f> projectedPoints;
        cv::projectPoints(objectPoints, rvec, tvec, cameraMatrix, distCoeffs, projectedPoints);

        for (unsigned int j = 0; j < objectPoints.size(); ++j) {
            double error = cv::norm(projectedPoints[j] - imagePoints[j]);
            if (error < distanceThreshold)
                inlierCount++;
        }*/
        // Update best pose estimate if the current pose has more inliers
        if (inlierCount > bestNumInliers){
            bestNumInliers = inlierCount;
            bestRvec = rvec.clone();
            bestTvec = tvec.clone();
        }

    }
}


std::vector<cv::Point2f> findChessboardCorners(const cv::Mat& input)
{
    cv::Size patternsize(9,6); //interior number of corners
    cv::Mat image_gray; //source image
    cv::Mat objectImage = input.clone();
    std::vector<cv::Point2f> corners;
    cv::cvtColor(objectImage, image_gray, CV_BGR2GRAY); // Convert to Gray
    //CALIB_CB_FAST_CHECK saves a lot of time on images
    //that do not contain any chessboard corners
    bool patternfound = cv::findChessboardCorners(image_gray, patternsize, corners,
            cv::CALIB_CB_ADAPTIVE_THRESH + cv::CALIB_CB_NORMALIZE_IMAGE
            + cv::CALIB_CB_FAST_CHECK);
    if (patternfound)
    {
        return corners;
    }
    else
    {
        return std::vector<cv::Point2f>();
    }
}

int main(){
    // Open the video file
    cv::VideoCapture cap("../solvePNP/data/video.mp4");
    if (!cap.isOpened()){
        std::cout << "Failed to open video file!" << std::endl;
        return -1;
    }

    // Camera intrinsic parameters
    cv::Mat camera_matrix;
    cv::Mat distortion_coeffs;
    cv::Size chessboardSize(9, 6);
    double squareSize = 0.026; // Size of a chessboard square (in meters)
    
	// Open calibration YAML file
	cv::FileStorage fs1("../solvePNP/data/camera_params.yaml", cv::FileStorage::READ);
	if (!fs1.isOpened()){
		std::cerr << "Failed to open camera_params.yaml" << std::endl;
		exit(-1);
	}
	// Read the camera parameters
	fs1["camera_matrix"] >> camera_matrix;
	fs1["distortion_coefficients"] >> distortion_coeffs;
	// Release the file storage object and close the file
	fs1.release();

    // Real-world coordinates of the chessboard corners (in meters)
    std::vector<cv::Point3f> objectPoints;
    for (int i = 0; i < chessboardSize.height; i++){
        for (int j = 0; j < chessboardSize.width; j++){
            objectPoints.push_back(cv::Point3f(j * squareSize, i * squareSize, 0));
        }
    }
    cv::Mat rvec, tvec;
    // Process each frame of the video
    while (true){
        // Read the next frame
        cv::Mat frame;
        cap >> frame;
        if (frame.empty()){
            break;
        }

        // Find the chessboard corners in the frame
        std::vector<cv::Point2f> imagePoints = findChessboardCorners(frame);

        // Draw the chessboard corners on the frame
        if (!imagePoints.empty())
        {
            cv::drawChessboardCorners(frame, cv::Size(9, 6), cv::Mat(imagePoints), true);

            // Estimate the pose of the chessboard
            int maxIterations = 500;
            double distanceThreshold = 3.0;
            solveRANSAC(objectPoints, imagePoints, camera_matrix, distortion_coeffs, rvec, tvec, maxIterations, distanceThreshold);
            //v::solvePnPRansac(objectPoints, imagePoints, camera_matrix, distortion_coeffs, rvec, tvec);
            

            // Draw the coordinate axes on the frame
/*
                std::vector<cv::Point3f> axisPoints = {cv::Point3f(0, 0, 0), cv::Point3f(0.1, 0, 0), cv::Point3f(0, 0.1, 0), cv::Point3f(0, 0, 0.1)};

                // Project the 3D axis points onto the image plane
                std::vector<cv::Point2f> projectedPoints;
                cv::projectPoints(axisPoints, rvec, tvec, camera_matrix, distortion_coeffs, projectedPoints);

                // Draw the coordinate axes on the frame
                cv::line(frame, projectedPoints[0], projectedPoints[1], cv::Scalar(0, 0, 255), 3);
                cv::line(frame, projectedPoints[0], projectedPoints[2], cv::Scalar(0, 255, 0), 3);
                cv::line(frame, projectedPoints[0], projectedPoints[3], cv::Scalar(255, 0, 0), 3);*/

        }

        // Display the result
        cv::imshow("Frame", frame);
        if (cv::waitKey(1) == 27)
        {
            break;
        }
    }

    // Release the video capture
    cap.release();

    return 0;
}