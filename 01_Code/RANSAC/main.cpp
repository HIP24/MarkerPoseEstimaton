#include <iostream>
#include <opencv2/imgproc.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/xfeatures2d/nonfree.hpp>
#include <opencv2/opencv.hpp>
#include <vector>

int main()
{
    // Read the two images
    cv::Mat image1 = cv::imread("img1.jpg");
    cv::Mat image2 = cv::imread("img2.jpg");

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
    for (const cv::DMatch& match : matches)
    {
        matched_pts1.push_back(keypoints1[match.queryIdx].pt);
        matched_pts2.push_back(keypoints2[match.trainIdx].pt);
    }

    // Calculate the homography using RANSAC
    cv::Mat homography = cv::findHomography(matched_pts1, matched_pts2, cv::RANSAC);

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

    // Display the result
    cv::imshow("RANSAC", canvas);
    cv::waitKey(0);

    return 0;
}
