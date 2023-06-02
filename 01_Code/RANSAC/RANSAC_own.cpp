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

int main()
{
    // Read the two images
    cv::Mat image1 = cv::imread("data/img1.jpg");
    cv::Mat image2 = cv::imread("data/img2.jpg");

    if (image1.empty())
    {
        std::cout << "Image was not found, try again!" << std::endl;
        return -1;
    }

    if (image2.empty())
    {
        std::cout << "Image2 was not found, try again!" << std::endl;
        return -1;
    }

    // Create a SIFT object
    cv::Ptr<cv::Feature2D> f2d = cv::xfeatures2d::SIFT::create(0, 3, 0.3, 10, 1.6);

    // Detect SIFT features in the images
    std::vector<cv::KeyPoint> keypoints1, keypoints2;
    cv::Mat descriptors1, descriptors2;
    f2d->detectAndCompute(image1, cv::noArray(), keypoints1, descriptors1);
    f2d->detectAndCompute(image2, cv::noArray(), keypoints2, descriptors2);

    // Initialize brute force matcher
    cv::BFMatcher matcher(cv::NORM_L2);

    // Perform brute force matching between the descriptors
    std::vector<cv::DMatch> matches;
    matcher.match(descriptors1, descriptors2, matches);

    // Draw the matched keypoints on the images
    cv::Mat img_matches;
    cv::drawMatches(image1, keypoints1, image2, keypoints2, matches, img_matches);
    cv::imshow("SIFT", img_matches);

    // Extract the matched keypoints for calculating the homography
    std::vector<cv::Point2f> matched_pts1, matched_pts2;
    for (const cv::DMatch &match : matches)
    {
        matched_pts1.push_back(keypoints1[match.queryIdx].pt);
        matched_pts2.push_back(keypoints2[match.trainIdx].pt);
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
