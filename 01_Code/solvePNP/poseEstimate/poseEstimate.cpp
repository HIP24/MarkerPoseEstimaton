//#include <opencv2/opencv.hpp>	easy road - it will include everything but cost time
#include <iostream>
#include <opencv2/imgproc.hpp>		// For line drawing
#include <opencv2/features2d.hpp>	// For SIFT sruff
#include <opencv2/highgui.hpp>		// For general cv
#include <fstream>					// For file stream csv
#include <opencv2/ml.hpp>			// For cv::Ptr<cv::ml::TrainData>
#include <opencv2/calib3d.hpp>		// For solvePnP

class poseEstimate{
	public:
		// Constructor
		poseEstimate(const std::vector<cv::Point3f>& activeSetXYZ, const cv::Mat& cameraMatrix, const cv::Mat& distCoeffs):activeSetXYZ_(activeSetXYZ), 
			cameraMatrix_(cameraMatrix), distCoeffs_(distCoeffs){}
		// Destructor
		~poseEstimate(){}
		
		void calculateTransRot(const std::vector<cv::Point2f>& imagePoints){
			// Define the rotation and translation vectors
			cv::Mat rvec, tvec;
			// Call solvePnP()
			cv::solvePnP(activeSetXYZ_, imagePoints, cameraMatrix_, distCoeffs_, rvec, tvec);
			std::cout << "Distance to Object: " << tvec.at<double>(2) << std::endl;
			std::cout << "x deviation: " << tvec.at<double>(0) << std::endl;
			std::cout << "y deviation: " << tvec.at<double>(1) << std::endl << std::endl;
		}
	private:
		std::vector<cv::Point3f> activeSetXYZ_;
		cv::Mat cameraMatrix_, distCoeffs_;
};

class loadObjectData{
	public:
		// Constructor
		loadObjectData(const std::string& imagePath, const std::string& videoPath, const std::string& calibrationPath,
		 const std::string& ofdPath, const std::string& activeSetPath, const std::string& activeSetXYZPath){
			// Load the object image
			picture_ = cv::imread(imagePath);
			if (picture_.empty()){
				std::cerr << "Could not read image!" << std::endl;
				exit(-1);
			}
			// Load video
			video.open(videoPath);
			if (!video.isOpened()){
				std::cerr << "Could not open video!" << std::endl;
				exit(-1);
			}
			
			// Open calibration YAML file
			cv::FileStorage fs1(calibrationPath, cv::FileStorage::READ);
			if (!fs1.isOpened())
			{
				std::cerr << "Failed to open camera_params.yaml" << std::endl;
				exit(-1);
			}
			// Read the camera parameters
			fs1["camera_matrix"] >> camera_matrix;
			fs1["distortion_coefficients"] >> distortion_coeffs;
			// Release the file storage object and close the file
			fs1.release();
			
			// Read featuredescriptor *.csv
			cv::FileStorage fs2(ofdPath, cv::FileStorage::READ);
			if (!fs2.isOpened())
			{
				std::cerr << "Failed to open featuredescriptor.yaml" << std::endl;
				exit(-1);
			}
			cv::FileNode keypointsNode = fs2["keypoints"];
			cv::FileNodeIterator it = keypointsNode.begin(), it_end = keypointsNode.end();
			for (; it != it_end; ++it){
				cv::KeyPoint kp;
				(*it)["x"] >> kp.pt.x;
				(*it)["y"] >> kp.pt.y;
				(*it)["size"] >> kp.size;
				(*it)["angle"] >> kp.angle;
				(*it)["response"] >> kp.response;
				(*it)["octave"] >> kp.octave;
				(*it)["class_id"] >> kp.class_id;
				trainkey.push_back(kp);
			}
			fs2["descriptors"] >> train_descriptors;
			fs2.release();
			
			// Open the activeSet.csv
			std::ifstream activeSetfile(activeSetPath);
			if (!activeSetfile.is_open()){
				std::cerr << "Failed to open activeSet.csv" << std::endl;
				exit(-1);
			}
			// Read the data
			int value;
			while (activeSetfile >> value) {
				activeSetIndex.push_back(value);
			}
			// Close the file
			activeSetfile.close();
				
			// Load CSV file - siehe Übung 2
			cv::Ptr<cv::ml::TrainData> tdata = cv::ml::TrainData::loadFromCSV(activeSetXYZPath,0,0,1);
			cv::Mat samples = tdata->getSamples();                                                   
			cv::Mat target = tdata->getResponses();	
			for(int i = 0;i < samples.rows; i++){
				//std::cout << target.at<float>(i,0) << "In constructor ActiveSetXYZ:" << samples.at<float>(i,0) << ' ' 
				//	<< samples.at<float>(i,1) << ' ' << samples.at<float>(i,2) << std::endl;
				cv::Point3f temp(samples.at<float>(i,0), samples.at<float>(i,1), samples.at<float>(i,2));
				activeSetXYZ.push_back(temp);
			}
		}
		// Destructor
		~loadObjectData(){}
		
		// Getter
		cv::Mat getImage() const {return picture_;}
		cv::VideoCapture getVideo() const {return video;}
		cv::Mat getCameramatrix() const {return camera_matrix;}
		cv::Mat getDistortcoeff() const {return distortion_coeffs;}
		cv::Mat getTraindiscriptor() const {return train_descriptors;}
		std::vector<cv::KeyPoint> getTrainkey() const {return trainkey;}
		std::vector<int> getActiveSetIndex() const {return activeSetIndex;}
		std::vector<cv::Point3f> getActiveSetXYZ() const {return activeSetXYZ;}
		
		// For Debugging - simply show video and image object in one window
		void display(){
			while(true){
				// Read a frame from the video
				cv::Mat frame;
				video >> frame;
				if (frame.empty()){
					video.set(cv::CAP_PROP_POS_FRAMES, 0);
					video >> frame;
					// Exit the loop if the video is still empty
					if(frame.empty())	exit(-1);	
				}

				// Display object image on the left half of the window
				cv::Mat displayImage = picture_.clone();
				cv::resize(displayImage, displayImage, cv::Size(800, 600));

				// Display video frame on the right half of the window
				cv::Mat displayFrame = frame.clone();
				cv::resize(displayFrame, displayFrame, cv::Size(800, 600));
				
				// Concatenate the object image and video frame horizontally
				cv::Mat display;
				cv::hconcat(displayImage, displayFrame, display);
				cv::imshow("Object Image and Video", display);
				// Wait for key press to exit
				int key = cv::waitKey(1);
				if (key == 27) {
					break;
				}
			}
		}
	private:
		cv::Mat picture_, camera_matrix, distortion_coeffs, train_descriptors;
		cv::VideoCapture video;
		std::vector<cv::KeyPoint> trainkey;
		std::vector<int> activeSetIndex;
		std::vector<cv::Point3f> activeSetXYZ;
};

class SIFTmatcher{
	public:
		// Constructor
		SIFTmatcher(const cv::Mat img, cv::Mat obj_descriptors, std::vector<int> activeSetindex, std::vector<cv::KeyPoint> obj_keypoints, int threshold = 250):threshold_(threshold){
			// Create SIFT detector
			detector = cv::SIFT::create(nfeatures, nOctaveLayers, contrastThreshold, edgeThreshold, sigma); // create SIFT Detector
			image_obj = img.clone( );
			// Create a new matrix to hold the selected descriptors
			obj_descriptors_ = cv::Mat(activeSetindex.size(), obj_descriptors.cols, obj_descriptors.type());
			// Extract the descriptors corresponding to activeSetindex
			for(size_t i = 0; i < activeSetindex.size(); i++){
				int index = activeSetindex[i];
				cv::Mat descriptorRow = obj_descriptors.row(index-1);
				descriptorRow.copyTo(obj_descriptors_.row(i));
			}
			// Extract the keypoints corresponding to activeSetindex
			for(size_t i = 0; i < activeSetindex.size(); i++){
				int idx = activeSetindex[i];
				obj_keypoints_.push_back(obj_keypoints[idx-1]);
			}
		} 
		// Destructor
		~SIFTmatcher(){}
		
		// Getter
		cv::Mat getObjDescriptor() const{return obj_descriptors_;}		
		
		// match function taken from moodle file and adapted for this program
		void match(cv::Mat& img, cv::Mat& result, std::vector<cv::Point2f>& imagePoints, int min_MATCH = 10){
			std::vector<cv::DMatch> matches;                    // Memory for matches
			std::vector<cv::KeyPoint> keypoints;                // Memory for Keypoints
			cv::Mat descriptors;                                  // Memory for Keypoints
			cv::Mat img_gray;                                     // Gray representation of RGB image
			cv::cvtColor(img, img_gray, CV_BGR2GRAY);           // Convert to mono
			detector->detect(img_gray, keypoints);               // Get Features
			detector->compute(img_gray, keypoints, descriptors); // Get Descriptors
			//--- Match with learned object ---//
			cv::BFMatcher matcher(cv::NORM_L2); // Create BF matcher. This datastructure contains info for math descriptors and metrices
			matcher.match(obj_descriptors_, descriptors, matches); // Do matching

			//std::cout << "Got " << matches.size() << " matches\nFilter matches...\n";
			std::vector<cv::DMatch> goodMatches; // Memory for matches
			for(size_t i = 0; i < matches.size(); i++ ) {
				if(matches[i].distance < threshold_){
					goodMatches.push_back(matches[i]);
				}
			}

			std::vector<cv::Point2f> temp; // 2D coordinates of keypoints in the image
			for (size_t i = 0; i < goodMatches.size(); i++) {
				cv::Point2f point = keypoints[goodMatches[i].trainIdx].pt;
				temp.push_back(point);
			}
			imagePoints = temp;

			//std::cout << "Result: " << goodMatches.size() << " matches\n";
			// std::cout << "Result: " << goodKeypoints.size( ) << " matches\n";
			//std::cout << "Stored: " << obj_keypoints_.size() << " matches\n";
			// std::sort(matches.begin(), matches.end());
			// matches.erase(matches.begin() + 10, matches.end());
			cv::Mat image_obj_resized;
			cv::resize(image_obj, image_obj_resized, cv::Size(640, 480));
			if(matches.size() >= unsigned(min_MATCH)){
				try {
					// Resize keypoints
					std::vector<cv::KeyPoint> obj_keypoints_resized = obj_keypoints_;
					for (auto& kp : obj_keypoints_resized) {
						kp.pt.x *= 640.0f / image_obj.cols;
						kp.pt.y *= 480.0f / image_obj.rows;
						kp.size *= 640.0f / image_obj.cols;
					}
					cv::drawMatches(image_obj_resized, obj_keypoints_resized, img, keypoints, goodMatches, result); // Draw final results
				} catch(...){
					result = img.clone();
				}
			} else {
				result = img.clone();
			}
		}
		

		
		
	private:
		cv::Mat obj_descriptors_, image_obj;
		std::vector<cv::KeyPoint> obj_keypoints_;
		cv::Ptr<cv::SIFT> detector;
		int threshold_;
		
		// Defined hyperparameters from last program
		int nfeatures = 0;
		int nOctaveLayers = 7;
		double edgeThreshold = 2.25;
		double contrastThreshold = 0.1;
		double sigma = 1.6;
};

int main(int argc, char* argv[]){
	 if (argc != 2) {
        std::cout << "Usage: " << argv[0] << " <threshold>" << std::endl;
        exit(-1);
    }
	int threshold = atoi(argv[1]);
	std::string imagePath = "../data/train_image.jpg";
	std::string videoPath = "../data/video.mp4";
	std::string calibrationPath = "camera_params.yaml";
	std::string trainfeatdescrPath = "../data/featuredescriptor.yaml";
	std::string activeSetPath = "../data/activeSet.csv";
	std::string activeSetXYZPath = "../data/activeSet_XYZ.csv";
	loadObjectData input(imagePath, videoPath, calibrationPath, trainfeatdescrPath, activeSetPath, activeSetXYZPath);
	// Init poste Estimate
	poseEstimate ePose(input.getActiveSetXYZ(), input.getCameramatrix(), input.getDistortcoeff());
	// Init SIFTMatcher
	SIFTmatcher match(input.getImage(), input.getTraindiscriptor(), input.getActiveSetIndex(), input.getTrainkey(), threshold);
	// Saving in video1 so it can be used in video1 >> vidImg;
	cv::VideoCapture video1 = input.getVideo();
	key_t key;
	cv::Mat vidImg, vidImgUndistort, result;
	std::vector<cv::Point2f> imagePoints;
    while(key != 27){                   // Do till <ESC> was pressed
        video1 >> vidImg;                     // Get image from video1
		if(vidImg.empty()){					// Start video from the beginning
			video1.set(cv::CAP_PROP_POS_FRAMES, 0);
			video1 >> vidImg;
			// Exit the loop if the video is still empty
			if(vidImg.empty())	exit(-1);	
		}
		// Smartphone Camera was used 1280x720. Calibration tool: https://www.camcalib.io/
		cv::undistort(vidImg, vidImgUndistort, input.getCameramatrix(), input.getDistortcoeff());
        
		match.match(vidImgUndistort, result, imagePoints, 10); // Do SIFT matching
		cv::imshow("SIFT Matching Results", result);             // Show results
		// Do pose estimation only if there are 10 good matches
		if(imagePoints.size() == 10){
			ePose.calculateTransRot(imagePoints);
		}
        key = cv::waitKey(5);                  // Update window
    }	
	return 0;
}