#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/calib3d/calib3d.hpp>

// DLT without linear Regression
// https://www.youtube.com/watch?v=3NcQbZu6xt8
// What is SVD?
// https://www.youtube.com/watch?v=nbBvuuNVfco&t=482s

// DLT-Methode zur Pose Estimation
cv::Mat estimatePoseDLT(const std::vector<cv::Point2f>& imagePoints, const std::vector<cv::Point3f>& worldPoints, const cv::Mat& cameraMatrix)
{
    // Überprüfen, ob genügend Punkte vorhanden sind
    if (imagePoints.size() < 6 || worldPoints.size() < 6 || imagePoints.size() != worldPoints.size())
    {
        std::cerr << "Nicht genügend 2D-3D-Korrespondenzen!" << std::endl;
        return cv::Mat();
    }

    // Gleichungssystem aufstellen
    cv::Mat A(2 * imagePoints.size(), 12, CV_64F);
    for (int i = 0; i < imagePoints.size(); i++)
    {
        const cv::Point2f& imgPoint = imagePoints[i];
        const cv::Point3f& worldPoint = worldPoints[i];

        A.at<double>(2 * i, 0) = -worldPoint.x;
        A.at<double>(2 * i, 1) = -worldPoint.y;
        A.at<double>(2 * i, 2) = -worldPoint.z;
        A.at<double>(2 * i, 3) = -1.0;
        A.at<double>(2 * i, 8) = imgPoint.x * worldPoint.x;
        A.at<double>(2 * i, 9) = imgPoint.x * worldPoint.y;
        A.at<double>(2 * i, 10) = imgPoint.x * worldPoint.z;
        A.at<double>(2 * i, 11) = imgPoint.x;

        A.at<double>(2 * i + 1, 4) = -worldPoint.x;
        A.at<double>(2 * i + 1, 5) = -worldPoint.y;
        A.at<double>(2 * i + 1, 6) = -worldPoint.z;
        A.at<double>(2 * i + 1, 7) = -1.0;
        A.at<double>(2 * i + 1, 8) = imgPoint.y * worldPoint.x;
        A.at<double>(2 * i + 1, 9) = imgPoint.y * worldPoint.y;
        A.at<double>(2 * i + 1, 10) = imgPoint.y * worldPoint.z;
        A.at<double>(2 * i + 1, 11) = imgPoint.y;
    }

    // Singular Value Decomposition (SVD) durchführen
    cv::SVD svd(A, cv::SVD::FULL_UV);

    // Die Lösung ist die letzte Spalte der V-Matrix der SVD
    cv::Mat poseVector = svd.vt.row(svd.vt.rows - 1);

    // Pose-Matrix aufstellen
    cv::Mat poseMatrix(3, 4, CV_64F);
    for (int i = 0; i < 3; i++)
    {
        for (int j = 0; j < 4; j++)
        {
            poseMatrix.at<double>(i, j) = poseVector.at<double>(4 * i + j);
        }
    }

    // Kameramatrix anwenden, um die Pose in Weltkoordinaten zu transformieren
    cv::Mat cameraPose = cameraMatrix.inv() * poseMatrix;

    return cameraPose;
}

int main()
{
    // Beispielaufruf der DLT-Methode
    std::vector<cv::Point2f> imagePoints;   // 2D-Bildpunkte
    std::vector<cv::Point3f> worldPoints;   // 3D-Weltpunkte
    cv::Mat cameraMatrix;                   // Kameramatrix

    // Füllen Sie imagePoints, worldPoints und cameraMatrix mit den entsprechenden Werten

    cv::Mat estimatedPose = estimatePoseDLT(imagePoints, worldPoints, cameraMatrix);

    // Verwenden Sie die geschätzte Pose (Kameramatrix) für weitere Berechnungen oder Visualisierungen

    return 0;
}
