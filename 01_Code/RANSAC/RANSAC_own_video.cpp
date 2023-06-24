#include <iostream>
#include <opencv2/imgproc.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/xfeatures2d/nonfree.hpp>
#include <opencv2/opencv.hpp>
#include <vector>
#include <numeric>

bool solvePnPOwn(const std::vector<cv::Point3f>& objectPoints,
                 const std::vector<cv::Point2f>& imagePoints,
                 const cv::Mat& cameraMatrix,
                 cv::Mat& rvec,
                 cv::Mat& tvec)
{
    // Convert the input data to the right format
    cv::Mat objectPointsMat(objectPoints.size(), 3, CV_32F);
    for (size_t i = 0; i < objectPoints.size(); i++)
    {
        objectPointsMat.at<float>(i, 0) = objectPoints[i].x;
        objectPointsMat.at<float>(i, 1) = objectPoints[i].y;
        objectPointsMat.at<float>(i, 2) = objectPoints[i].z;
    }
    cv::Mat imagePointsMat(imagePoints.size(), 2, CV_32F);
    for (size_t i = 0; i < imagePoints.size(); i++)
    {
        imagePointsMat.at<float>(i, 0) = imagePoints[i].x;
        imagePointsMat.at<float>(i, 1) = imagePoints[i].y;
    }

    // Compute the homography between the object points and the image points
    cv::Mat H = cv::findHomography(objectPointsMat, imagePointsMat);

    // Compute the camera pose from the homography
    cv::Mat Kinv = cameraMatrix.inv();
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
    for (int i = 0; i < 3; i++)
    {
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


bool solvePnPRansacOwn(const std::vector<cv::Point3f>& objectPoints,
                       const std::vector<cv::Point2f>& imagePoints,
                       const cv::Mat& cameraMatrix,
                       const cv::Mat& distCoeffs,
                       cv::Mat& rvec,
                       cv::Mat& tvec,
                       int iterationsCount = 100,
                       float reprojectionError = 8.0)
{
    int numPoints = objectPoints.size();
    int bestNumInliers = 0;
    cv::Mat bestRvec, bestTvec;

    for (int i = 0; i < iterationsCount; i++)
    {
        // Randomly select a subset of points
        std::vector<int> indices(numPoints);
        std::iota(indices.begin(), indices.end(), 0);
        std::random_shuffle(indices.begin(), indices.end());
        std::vector<cv::Point3f> objectPointsSubset = {objectPoints[indices[0]], objectPoints[indices[1]], objectPoints[indices[2]], objectPoints[indices[3]]};
        std::vector<cv::Point2f> imagePointsSubset = {imagePoints[indices[0]], imagePoints[indices[1]], imagePoints[indices[2]], imagePoints[indices[3]]};

        // Estimate the pose using the selected subset of points
        cv::Mat rvecSubset, tvecSubset;
        //bool success = cv::solvePnP(objectPointsSubset, imagePointsSubset, cameraMatrix, distCoeffs, rvecSubset, tvecSubset);
        bool success = solvePnPOwn(objectPoints, imagePoints, cameraMatrix, rvecSubset, tvecSubset);

        // Count the number of inliers
        if (success)
        {
            int numInliers = 0;
            for (int j = 0; j < numPoints; j++)
            {
                std::vector<cv::Point3f> objectPoint = {objectPoints[j]};
                std::vector<cv::Point2f> projectedPoint;
                cv::projectPoints(objectPoint, rvecSubset, tvecSubset, cameraMatrix, distCoeffs, projectedPoint);
                double distance = cv::norm(projectedPoint[0] - imagePoints[j]);
                if (distance < reprojectionError)
                {
                    numInliers++;
                }
            }

            // Update the best pose
            if (numInliers > bestNumInliers)
            {
                bestRvec = rvecSubset;
                bestTvec = tvecSubset;
                bestNumInliers = numInliers;
            }
        }
    }

    // Set the output pose to the best pose
    rvec = bestRvec;
    tvec = bestTvec;

    return (bestNumInliers > 0);
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

int main()
{
    // Open the video file
    cv::VideoCapture cap("../solvePNP/data/video.mp4");
    if (!cap.isOpened())
    {
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
    camera_matrix.convertTo(camera_matrix, CV_64F);


    // Real-world coordinates of the chessboard corners (in meters)
    std::vector<cv::Point3f> objectPoints;
    for (int i = 0; i < chessboardSize.height; i++)
    {
        for (int j = 0; j < chessboardSize.width; j++)
        {
            objectPoints.push_back(cv::Point3f(j * squareSize, i * squareSize, 0));
        }
    }

    // Process each frame of the video
    while (true)
    {
        // Read the next frame
        cv::Mat frame;
        cap >> frame;
        if (frame.empty())
        {
            break;
        }

        // Find the chessboard corners in the frame
        std::vector<cv::Point2f> imagePoints = findChessboardCorners(frame);

        // Draw the chessboard corners on the frame
        if (!imagePoints.empty())
        {
            cv::drawChessboardCorners(frame, cv::Size(9, 6), cv::Mat(imagePoints), true);

            // Estimate the pose of the chessboard
            cv::Mat rvec, tvec;
            //bool success = cv::solvePnPRansac(objectPoints, imagePoints, camera_matrix, distortion_coeffs, rvec, tvec);
            bool success = solvePnPRansacOwn(objectPoints, imagePoints, camera_matrix, distortion_coeffs, rvec, tvec);
            std::cout << "tvec: " << tvec << std::endl;
            std::cout << "rvec: " << rvec << std::endl;

            // Draw the coordinate axes on the frame
            if (success)
            {
                std::vector<cv::Point3f> axisPoints = {cv::Point3f(0, 0, 0), cv::Point3f(0.1, 0, 0), cv::Point3f(0, 0.1, 0), cv::Point3f(0, 0, 0.1)};

                // Project the 3D axis points onto the image plane
                std::vector<cv::Point2f> projectedPoints;
                cv::projectPoints(axisPoints, rvec, tvec, camera_matrix, distortion_coeffs, projectedPoints);

                // Draw the coordinate axes on the frame
                cv::line(frame, projectedPoints[0], projectedPoints[1], cv::Scalar(0, 0, 255), 3);
                cv::line(frame, projectedPoints[0], projectedPoints[2], cv::Scalar(0, 255, 0), 3);
                cv::line(frame, projectedPoints[0], projectedPoints[3], cv::Scalar(255, 0, 0), 3);
            }
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
