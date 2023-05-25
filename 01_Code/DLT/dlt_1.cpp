#include <iostream>
#include <opencv2/calib3d.hpp>		// For chessboard corner detection
#include <opencv2/ml.hpp>			// For cv::Ptr<cv::ml::TrainData>
#include <opencv2/imgproc.hpp>		// For line drawing and cvtColor
#include <opencv2/highgui.hpp>		// For general cv
#include <opencv2/core.hpp>         // For cv::sqrt



class directLinearTransformation{
    public:
        // Constructor
        directLinearTransformation(std::vector<cv::Point3f> activeSetXYZ):activeSetXYZ_(activeSetXYZ){}
        // Destructor
        ~directLinearTransformation(){}

        // Getter
		cv::Mat getorginalImage() const{return objectImage;}
		cv::Mat getImageout() const{return imageoutput;}
		std::vector<cv::Point2f> getCornerXY() const{return corners;}


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
            // Check if there are enough maches available 'cause you need at least 6 (2x6) for 11 unkown variables
            if (corners.size() < 6 || activeSetXYZ_.size() < 6 || corners.size() != activeSetXYZ_.size()){
                std::cerr << "Not enough 2D-3D-correspndence!" << std::endl;
                exit(-1);
            }
            // See Moodle slide S.34 "PoseEstimation"
            cv::Mat Y(corners.size(), 4, CV_64F);
            cv::Mat X(activeSetXYZ_.size(), 4, CV_64F);
            cv::Mat out;

            for(unsigned int i = 0; i < corners.size(); i++){
                Y.at<double>(i,0) = corners[i].x;
                Y.at<double>(i,1) = corners[i].y;
                Y.at<double>(i,2) = 1.0;
                Y.at<double>(i,3) = 0.0;
            }
            for(unsigned int i = 0; i < activeSetXYZ_.size(); i++){
                X.at<double>(i,0) = activeSetXYZ_[i].x;
                X.at<double>(i,1) = activeSetXYZ_[i].y;
                X.at<double>(i,2) = activeSetXYZ_[i].z;
                X.at<double>(i,3) = double(i)+1.0;
            }
            cv::Mat XT;
            cv::transpose(X,XT);
            out = (XT*Y)/(XT*X);
            cv::transpose(out,out);
            // Perform Singular Value Decomposition (SVD)
            cv::Mat U, S, Vt;
            cv::SVD::compute(out, S, U, Vt);

            // Separate K and T
            cv::sqrt(S,S);
            cv::Mat K = U * cv::Mat::diag(S);
            cv::Mat T = cv::Mat::diag(S) * Vt.t();

            std::cout << T << std::endl;
		}


    private:
        std::vector<cv::Point3f> activeSetXYZ_;
		std::vector<cv::Point2f> corners;
		cv::Mat objectImage, imageoutput;

};

int main(){
    cv::VideoCapture video;
	cv::Mat vidImg;
	std::vector<cv::Point3f> activeSetXYZ;

    // Open video input
	video.open("data/video.mp4");
	if (!video.isOpened()){
		std::cerr << "Could not open video!" << std::endl;
		exit(-1);
	}
	
	// Load CSV file with real 3D measurments
	cv::Ptr<cv::ml::TrainData> tdata = cv::ml::TrainData::loadFromCSV("data/activeSet_XYZ.csv",0,0,1);
	cv::Mat samples = tdata->getSamples();
	for(int i = 0; i < samples.rows; i++){
		cv::Point3f temp(samples.at<float>(i,0), samples.at<float>(i,1), samples.at<float>(i,2));
		activeSetXYZ.push_back(temp);
	}
    // Init poste Estimate
	directLinearTransformation ePose(activeSetXYZ);

    key_t key;
    while(key != 27){                   // Do till <ESC> was pressed
        key = cv::waitKey();                  // Update window
		video >> vidImg;                     // Get image from
        if (vidImg.empty()){
			video.set(cv::CAP_PROP_POS_FRAMES, 0);
			video >> vidImg;
			// Exit the loop if the video is still empty
			if(vidImg.empty())	exit(-1);	
		}
		ePose.findChessboardCorners(vidImg);
        cv::Mat out = ePose.getImageout();

		// Resizing otherwise it would take too long to display every frame
		cv::resize(out, out, cv::Size(out.cols/2, out.rows/2));
		if(!ePose.getCornerXY().empty()){
			ePose.calculateTransRot();
			//ePose.displayTransRot(out);
		}
		cv::imshow("ChessboardCorner", out);
    }


    return 0;
}