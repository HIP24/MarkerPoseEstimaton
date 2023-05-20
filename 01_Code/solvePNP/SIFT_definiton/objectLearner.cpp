#include <iostream>
#include <opencv2/calib3d.hpp>		// For chessboard corner detection

#include <opencv2/imgproc.hpp>		// For line drawing
#include <opencv2/features2d.hpp>	// For SIFT sruff
#include <opencv2/highgui.hpp>		// For general cv
#include <fstream>					// For file stream csv


class loadObjectData{
	public:
		// Constructor
		loadObjectData(const std::string& imagePath, const std::string& videoPath){
			// Load the object image
			picture_ = cv::imread(imagePath);
			if (picture_.empty()){
				std::cerr << "Could not read image!" << std::endl;
				exit(-1);
			}
			// Initialize the video capture
			video.open(videoPath);
			if (!video.isOpened()){
				std::cerr << "Could not open video!" << std::endl;
				exit(-1);
			}
		}
		// Destructor
		~loadObjectData(){}
		
		// For Debugging
		// Getter
		cv::Mat getImage() const {return picture_;}
		cv::VideoCapture getVideo() const {return video;}
		
		
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
		cv::Mat picture_;
		cv::VideoCapture video;
};

class ObjectLearner{
	public:
		// Constructor
		ObjectLearner(){
			//cv::namedWindow("KeyPointimage", cv::WINDOW_NORMAL);
		}
		// Destructor
		~ObjectLearner(){}

		void findChessboardCorners(const cv::Mat& input){
			cv::Size patternsize(4,4); //interior number of corners
			cv::Mat image_gray; //source image
			objectImage = input.clone();
			cv::cvtColor(objectImage, image_gray, CV_BGR2GRAY);	// Convert to Gray
			//CALIB_CB_FAST_CHECK saves a lot of time on images
			//that do not contain any chessboard corners
			bool patternfound = cv::findChessboardCorners(image_gray, patternsize, corners,
					cv::CALIB_CB_ADAPTIVE_THRESH + cv::CALIB_CB_NORMALIZE_IMAGE
					+ cv::CALIB_CB_FAST_CHECK);
			if(patternfound)
			//cv::cornerSubPix(image_gray, corners, cv::Size(11, 11), cv::Size(-1, -1), cv::TermCriteria(CV_TERMCRIT_EPS + CV_TERMCRIT_ITER, 30, 0.1));
			imageoutput = objectImage.clone();
			cv::drawChessboardCorners(imageoutput, patternsize, cv::Mat(corners), patternfound);
		}
		
		// Getter
		cv::Mat getImageout() const{return imageoutput;}
		/*void getoutput() const{
			cv::Mat resizeimageout = imageout.clone();
			cv::imshow("KeyPointimage", resizeimageout);
		}
		std::vector<cv::Point2f> getPoints() const{return points;}
		
		void createTrackbars(){
			cv::createTrackbar("Features(Default:0)(0-1000)", "KeyPointimage", &nfeatures, 1000, on_trackbar0, this);
			cv::createTrackbar("OctaveLayers(1-20)", "KeyPointimage", &nOctaveLayers, 20, on_trackbar4, this);
			cv::createTrackbar("Edge Threshold(Default:10)(0-20x100)", "KeyPointimage", &edgeThresholdint, 2000, on_trackbar1, this);
			cv::createTrackbar("Contrast Threshold(Default:0.04)(0-0.2x100)", "KeyPointimage", &contrastThresholdint, 20, on_trackbar2, this);
			cv::createTrackbar("Sigma(Default:1.6)(1-5x100)", "KeyPointimage", &sigmaint, 500, on_trackbar3, this);
		}
		// on_trackbar* function in general are updating the data and therfore the SIFT image
		// Some error cases with 0 are dealt with simple if conditions
		// Since trackbars are only using int values, a conversion was necessery for float values in edge, contrast and sigma
		static void on_trackbar0(int, void* userdata){
			ObjectLearner* obj_learner = static_cast<ObjectLearner*>(userdata);
			obj_learner->SIFTdetectcompute(obj_learner->objectImage);
		}
		static void on_trackbar1(int, void* userdata){
			ObjectLearner* obj_learner = static_cast<ObjectLearner*>(userdata);
			obj_learner->edgeThreshold = static_cast<double>(obj_learner->edgeThresholdint)/100;
			obj_learner->SIFTdetectcompute(obj_learner->objectImage);
		}
		static void on_trackbar2(int, void* userdata){
			ObjectLearner* obj_learner = static_cast<ObjectLearner*>(userdata);
			obj_learner->contrastThreshold = static_cast<double>(obj_learner->contrastThresholdint)/100;
			obj_learner->SIFTdetectcompute(obj_learner->objectImage);
		}
		static void on_trackbar3(int, void* userdata){
			ObjectLearner* obj_learner = static_cast<ObjectLearner*>(userdata);
			if(obj_learner->sigmaint < 100){
				obj_learner->sigma = static_cast<double>(obj_learner->sigmaint+100.0)/100;
			}else{
				obj_learner->sigma = static_cast<double>(obj_learner->sigmaint)/100;
			}
			obj_learner->SIFTdetectcompute(obj_learner->objectImage);
		}
		static void on_trackbar4(int, void* userdata){
			ObjectLearner* obj_learner = static_cast<ObjectLearner*>(userdata);
			if(obj_learner->nOctaveLayers < 1)	obj_learner->nOctaveLayers = 1;
			obj_learner->SIFTdetectcompute(obj_learner->objectImage);
		}
		void createMouseCallback(){
			cv::setMouseCallback("KeyPointimage", onMouse, this);
		}
		static void onMouse(int event, int x, int y, int, void* userdata){
			ObjectLearner* obj_learner = static_cast<ObjectLearner*>(userdata);
			if(event == cv::EVENT_LBUTTONDOWN){
				// save points
				obj_learner->points.push_back(cv::Point2f(x, y));
				// draw x on selected point
				cv::Point center(x, y);
				int radius = 10;
				int thickness = 2;
				cv::Scalar color(0, 0, 0); // black color
				// If clicked draw black cross on image
				cv::line(obj_learner->imageout, cv::Point(center.x - radius, center.y - radius), cv::Point(center.x + radius, center.y + radius), color, thickness);
				cv::line(obj_learner->imageout, cv::Point(center.x - radius, center.y + radius), cv::Point(center.x + radius, center.y - radius), color, thickness);
			}
			
		}
		void deletePoints(){
			points.clear();
			std::cout << "Points deleted!" << std::endl;
			imageout = imageoutoriginal.clone();
		}
		
		void getID(){
			std::vector<int> poi_indices; // poi = point of interest
			// Loop over all the keypoints detected by SIFT
			for(unsigned int i = 0; i < keypoints.size(); i++){
				// Check if the current keypoint has coordinates that match any of the points of interest
				for (unsigned int j = 0; j < points.size(); j++){
					cv::Point2f poi = points[j];
					cv::KeyPoint kp = keypoints[i];
					if (std::abs(kp.pt.x - poi.x) < 5 && std::abs(kp.pt.y - poi.y) < 5){
						// If there is a match, store the index of the keypoint in the poi_indices vector and break out of the inner loop
						poi_indices.push_back(i+1);
						break;
					}
				}
			}
			// Open the CSV file for writing
			std::ofstream file("../data/activeSet.csv");
			// Loop over the poi_indices and write them to the file
			for (unsigned int i = 0; i < poi_indices.size(); i++){
				file << poi_indices[i] << std::endl;
			}
			// Close the file
			file.close();
			numerateFeature(poi_indices);
			cv::imwrite("../data/chosenSIFT.jpg", imageoutSIFTpicked);
		}
		
		void SIFTdetectcompute(const cv::Mat& img){
			if(detector) detector.release();	// Release SIFT detector because hyperp needs to be updated every time trackbar is moved
			detector = cv::SIFT::create(nfeatures, nOctaveLayers, contrastThreshold, edgeThreshold, sigma); // create SIFT Detector
			objectImage = img.clone();						// Store object
			cv::Mat image_gray;	
			cv::cvtColor(img, image_gray, CV_BGR2GRAY);  	// Convert to Gray
			// Detect keypoints and extract descriptors
			detector->detectAndCompute(image_gray, cv::noArray(), keypoints, descriptors); //get Features and Descriptors
			//or cv::drawKeypoints(objectImage, keypoints, imageout, cv::Scalar(0,0,255), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
			objectImage.copyTo(imageout);
			// Not using drawKeypoints()function otherwise circle would appear a little small because of image size
			for (size_t i = 0; i < keypoints.size(); i++){
				cv::Point center(cvRound(keypoints[i].pt.x), cvRound(keypoints[i].pt.y));
				int radius = 7; // Set the radius of the circle to 20 pixels
				cv::circle(imageout, center, radius, cv::Scalar(0, 0, 255), 2);
			}
			// Create imageout copy so it can be used if the user decides to delete chossen features (delete X)
			imageoutoriginal = imageout.clone();
		}
		
		void numerateFeature(const std::vector<int>& index){
			bool check = false;
			imageoutSIFTpicked = imageoutoriginal.clone();
			// Numerate features: If the feature was picked the color is GREEN otherwise RED
			for(size_t i = 0; i < keypoints.size(); i++){
				for(size_t j = 0; j < index.size(); j++){
					if(i+1 == unsigned(index[j])){
						cv::putText(imageout, std::to_string(i+1), cv::Point(keypoints[i].pt.x+8, keypoints[i].pt.y-8), 
							cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(0,255,0), 1.5, cv::LINE_AA);
						cv::putText(imageoutSIFTpicked, std::to_string(i+1), cv::Point(keypoints[i].pt.x+8, keypoints[i].pt.y-8), 
							cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(0,255,0), 1.5, cv::LINE_AA);
						check = true;
					}						
				}
				if(check == false){
					cv::putText(imageout, std::to_string(i+1), cv::Point(keypoints[i].pt.x+8, keypoints[i].pt.y-8), 
						cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(255,0,0), 1.5, cv::LINE_AA);
					cv::putText(imageoutSIFTpicked, std::to_string(i+1), cv::Point(keypoints[i].pt.x+8, keypoints[i].pt.y-8), 
						cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(255,0,0), 1.5, cv::LINE_AA);
				}else{
					check = false;
				}
			}
		}
		
		void saveObject(std::string filename){
			// Save the object image
			cv::imwrite("../data/SIFTimage.jpg", imageoutoriginal);
			// Save the keypoints and descriptors
			cv::FileStorage myfile("../data/" + filename + ".yaml", cv::FileStorage::WRITE);
			myfile << "keypoints" << "[";
			for (const auto& lala : keypoints){
				myfile << "{";
				myfile << "x" << lala.pt.x;
				myfile << "y" << lala.pt.y;
				myfile << "size" << lala.size;
				myfile << "angle" << lala.angle;
				myfile << "response" << lala.response;
				myfile << "octave" << lala.octave;
				myfile << "class_id" << lala.class_id;
				myfile << "}";
			}
			myfile << "]";
			myfile << "descriptors" << descriptors;
			myfile.release();		
		}*/
		
	private:
		cv::Mat objectImage, imageoutput;
		std::vector<cv::Point2f> corners;
		/*cv::Mat objectImage, imageout, imageoutoriginal, imageoutSIFTpicked, descriptors;
		std::vector<cv::KeyPoint> keypoints;
		cv::Ptr<cv::SIFT> detector;
		// For getting index
		std::vector<cv::Point2f> points;
		// Trackbar variables
		int nfeatures = 0;
		int nOctaveLayers = 3;
		double edgeThreshold = 10;
		double contrastThreshold = 0.04;
		double sigma = 1.6;
		// Trackbar only works with integers - need workaround
		int edgeThresholdint;
		int contrastThresholdint;
		int sigmaint;*/
		
};



int main(){
	std::string imagePath = "../data/train_image.jpg";
	std::string videoPath = "../data/video.mp4";
	loadObjectData input(imagePath, videoPath);
	cv::Mat img1 = input.getImage();
	cv::VideoCapture video1 = input.getVideo();
	cv::Mat vidImg;
	/*cv::imshow("Image", input.getImage());
	cv::waitKey();
	input.display();*/
	ObjectLearner object;

	object.findChessboardCorners(img1);
	cv::Mat out = object.getImageout();
	cv::resize(out, out, cv::Size(out.cols/2, out.rows/2));
	cv::imshow("CheesboardCorner", out);
	cv::waitKey();

	// Create Trackbars for hyperparameters
	//object.createTrackbars();
	// Create Mouseclick function so i don't need to search for index manually
	//object.createMouseCallback();
	

	// SIFT for video so i can better define what "good" features are
	/*key_t key;
	while(key != 27){                   // Do till <ESC> was pressed
        video1 >> vidImg;                     // Get image from
        if (vidImg.empty()){
			video1.set(cv::CAP_PROP_POS_FRAMES, 0);
			video1 >> vidImg;
			// Exit the loop if the video is still empty
			if(vidImg.empty())	exit(-1);	
		}
		object.SIFTdetectcompute(vidImg);
		object.getoutput();
        key = cv::waitKey();                  // Update window
    }*/
	
	// Picking SIFT features for farther use
	/*object.SIFTdetectcompute(img1);
	std::cout << 
	"################################################\n"
	"# Pick Features by left-clicking on the image! #\n"
	"################################################\n" << std::endl;
	std::cout << "Please Press \nTo save [s]\nTo delete chossen points [d]\nTo save Point of Interest [f]\nTo leave [ESC]\n" << std::endl;
	std::cout << "################################################" << std::endl;
	for(;;){
		int key = cv::waitKey(5);
		if(key == 's') {  			// Save object with SIFT freatures if 's' key is pressed
			object.saveObject("featuredescriptor");
		}
		else if(key == 27){			// Exit 'ESC' program
			break;
		}else if(key == 'd'){ 		// Clear the points vector (picked features) when 'd' key is pressed
			object.deletePoints();	
		}else if(key == 'f'){		// Save picked indexes as *.csv if 'f' is pressed
			object.getID();
		}
		object.getoutput();			// Print out SIFT image
	}*/
	return 0;
}