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

    std::cout << "srcPointsInliers size: " << srcPointsInliers.size() << std::endl;
    std::cout << "dstPointsInliers size: " << dstPointsInliers.size() << std::endl;



    bestH = cv::findHomography(srcPointsInliers, dstPointsInliers);

    return bestH;
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
             bool success = cv::solvePnPRansac(objectPoints, imagePoints, camera_matrix, distortion_coeffs, rvec, tvec);


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
