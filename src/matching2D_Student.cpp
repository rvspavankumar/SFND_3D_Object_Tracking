#include <numeric>
#include "matching2D.hpp"

using namespace std;

// Find best matches for keypoints in two camera images based on several matching methods
void matchDescriptors(std::vector<cv::KeyPoint> &kPtsSource, std::vector<cv::KeyPoint> &kPtsRef, cv::Mat &descSource, cv::Mat &descRef,
                      std::vector<cv::DMatch> &matches, std::string descriptorType, std::string matcherType, std::string selectorType)
{
    // configure matcher
    bool crossCheck = false;
    cv::Ptr<cv::DescriptorMatcher> matcher;

    if (matcherType.compare("MAT_BF") == 0)
    {
        int normType = cv::NORM_L2;//cv::NORM_L2;//cv::NORM_HAMMING;
         if (descriptorType.compare("DES_HOG") == 0 )
         {
             normType = cv::NORM_L2;
         }
        matcher = cv::BFMatcher::create(normType, crossCheck);
        //cout << "MAT_BF Matching with cross check" << crossCheck;
    }
    else if (matcherType.compare("MAT_FLANN") == 0)
    {
        if (descSource.type() != CV_32F)
        {
            descSource.convertTo(descSource, CV_32F);
            descRef.convertTo(descRef, CV_32F);
        }
        matcher = cv::DescriptorMatcher::create(cv::DescriptorMatcher::FLANNBASED);
        //cout << "MAT_FLANN matching";
    }
    else
    {
        cerr << "wrong matcher type" << endl;
        exit(-1);
    }
    

    // perform matching task
    if (selectorType.compare("SEL_NN") == 0)
    { // nearest neighbor (best match)
        double t = (double)cv::getTickCount();

        matcher->match(descSource, descRef, matches); // Finds the best match for each descriptor in desc1
        t = ((double)cv::getTickCount() - t) / cv::getTickFrequency();
        //cout << " NN n= " << matches.size() << "matches in " << 1000 * t / 1.0 << "ms" << endl;
    }
    else if (selectorType.compare("SEL_KNN") == 0)
    { // k nearest neighbors (k=2)
    

        vector<vector<cv::DMatch>> KnnMatches;
        double t = (double)cv::getTickCount();
        matcher->knnMatch(descSource, descRef, KnnMatches, 2);
        t = ((double)cv::getTickCount() - t) / cv::getTickFrequency();
        //cout << " KNN n=" << KnnMatches.size() << "matches in" << 1000 * t / 1.0 << "ms" << endl;

        //filtering
        const double ratioThr = 0.8;
        for (int i = 0; i < KnnMatches.size(); i++)
        {
            if (KnnMatches[i][0].distance < ratioThr * KnnMatches[i][1].distance)
            {
                matches.push_back(KnnMatches[i][0]);
            }
        }
        //cout << "Filtering removed" << KnnMatches.size() - matches.size() << "keypoints" << endl;
        //cout << " KNN n = " << matches.size() << "matches in " << 1000 * t / 1.0 << "ms" << endl;
    }
    else
    {
        cerr << " no selector type" << endl;
        exit(-1);
    }

    
}

// Use one of several types of state-of-art descriptors to uniquely identify keypoints
void descKeypoints(vector<cv::KeyPoint> &keypoints, cv::Mat &img, cv::Mat &descriptors, string descriptorType, bool bVis)
{
    // select appropriate descriptor
    cv::Ptr<cv::DescriptorExtractor> extractor;
    if (descriptorType.compare("BRISK") == 0)
    {

        int threshold = 30;        // FAST/AGAST detection threshold score.
        int octaves = 3;           // detection octaves (use 0 to do single scale)
        float patternScale = 1.0f; // apply this scale to the pattern used for sampling the neighbourhood of a keypoint.

        extractor = cv::BRISK::create(threshold, octaves, patternScale);
    }
    else if (descriptorType.compare("BRIEF") == 0)
    {
        int bytes = 32;
        bool bOrientation = false;

        extractor = cv::xfeatures2d::BriefDescriptorExtractor::create(bytes, bOrientation);
    }
    else if (descriptorType.compare("ORB") == 0)
    {
        int features = 500;
        float scale = 1.2f;
        int nlevels = 8;
        int edgeThr = 31;
        int firstLevel = 0;
        int WATK = 2;
        cv::ORB::ScoreType scoreType = cv::ORB::HARRIS_SCORE;
        int patchSize = 31;
        int fastThr = 20;

        extractor = cv::ORB::create(features, scale, nlevels, edgeThr, firstLevel, WATK, scoreType, patchSize, fastThr);
    }
    else if (descriptorType.compare("FREAK") == 0)
    {
        bool orientationNormalized = true;
        bool scaleNormalized = true;
        float patternScale = 22.0f;
        int nOctaves = 4;
        const std::vector<int> selectedPairs = std::vector<int>();

        extractor = cv::xfeatures2d::FREAK::create(orientationNormalized, scaleNormalized, patternScale, nOctaves, selectedPairs);
    }
    else if (descriptorType.compare("AKAZE") == 0)
    {
        cv::AKAZE::DescriptorType descriptorTyp = cv::AKAZE::DESCRIPTOR_MLDB;
        int descriptorsize = 0;
        int descriptorChannels = 3;
        float threshold = 0.001f;
        int nOctaves = 4;
        int nOctaveLayers = 4;
        cv::KAZE::DiffusivityType diffus = cv::KAZE::DIFF_PM_G2;

        extractor = cv::AKAZE::create(descriptorTyp, descriptorsize, descriptorChannels, threshold, nOctaves, nOctaveLayers, diffus);
    }
    else if (descriptorType.compare("SIFT") == 0)
    {
        int nfeatures = 0;
        int nOctaveLayers = 3; 
        double contrastThreshold = 0.04;
        double edgeThreshold = 10.0;
        double sigma = 1.6;

        extractor = cv::xfeatures2d::SIFT::create(nfeatures, nOctaveLayers, contrastThreshold, edgeThreshold, sigma);
    }
    else
    {
        cerr << "wrong descriptor type"<<endl;
        exit (-1);
    }
    

    // perform feature description
    double t = (double)cv::getTickCount();
    extractor->compute(img, keypoints, descriptors);
    t = ((double)cv::getTickCount() - t) / cv::getTickFrequency();
    cout << descriptorType << " descriptor extraction in " << 1000 * t / 1.0 << " ms" << endl;
}

// Detect keypoints in image using the traditional Shi-Thomasi detector
void detKeypointsShiTomasi(vector<cv::KeyPoint> &keypoints, cv::Mat &img, bool bVis)
{
    // compute detector parameters based on image size
    int blockSize = 4;       //  size of an average block for computing a derivative covariation matrix over each pixel neighborhood
    double maxOverlap = 0.0; // max. permissible overlap between two features in %
    double minDistance = (1.0 - maxOverlap) * blockSize;
    int maxCorners = img.rows * img.cols / max(1.0, minDistance); // max. num. of keypoints

    double qualityLevel = 0.01; // minimal accepted quality of image corners
    double k = 0.04;

    // Apply corner detection
    double t = (double)cv::getTickCount();
    vector<cv::Point2f> corners;
    cv::goodFeaturesToTrack(img, corners, maxCorners, qualityLevel, minDistance, cv::Mat(), blockSize, false, k);

    // add corners to result vector
    for (auto it = corners.begin(); it != corners.end(); ++it)
    {

        cv::KeyPoint newKeyPoint;
        newKeyPoint.pt = cv::Point2f((*it).x, (*it).y);
        newKeyPoint.size = blockSize;
        keypoints.push_back(newKeyPoint);
    }
    t = ((double)cv::getTickCount() - t) / cv::getTickFrequency();
    //cout << "Shi-Tomasi detection with n=" << keypoints.size() << " keypoints in " << 1000 * t / 1.0 << " ms" << endl;

    // visualize results
    if (bVis)
    {
        cv::Mat visImage = img.clone();
        cv::drawKeypoints(img, keypoints, visImage, cv::Scalar::all(-1), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
        string windowName = "Shi-Tomasi Corner Detector Results";
        cv::namedWindow(windowName, 6);
        imshow(windowName, visImage);
        cv::waitKey(0);
    }
}

// Detect keypoints in image using the traditional Shi-Thomasi detector
void detKeypointsHarris(vector<cv::KeyPoint> &key_points, cv::Mat &img, bool bVis)
{
	// Detector parameters
    int blockSize = 2;     // for every pixel, a blockSize × blockSize neighborhood is considered
    int apertureSize = 3;  // aperture parameter for Sobel operator (must be odd)
    int minResponse = 100; // minimum value for a corner in the 8bit scaled response matrix
    double k = 0.04;       // Harris parameter (see equation for details)

    double t = (double)cv::getTickCount();

    // Detect Harris corners and normalize output
    cv::Mat dst, dst_norm, dst_norm_scaled;
    dst = cv::Mat::zeros(img.size(), CV_32FC1);
    cv::cornerHarris(img, dst, blockSize, apertureSize, k, cv::BORDER_DEFAULT);
    cv::normalize(dst, dst_norm, 0, 255, cv::NORM_MINMAX, CV_32FC1, cv::Mat());
    cv::convertScaleAbs(dst_norm, dst_norm_scaled);

    bool suppression = false;

   // TODO: Your task is to locate local maxima in the Harris response matrix 
    // and perform a non-maximum suppression (NMS) in a local neighborhood around 
    // each maximum. The resulting coordinates shall be stored in a list of keypoints 
    double overlap = 0;
    for (int r = 0; r < dst_norm.rows; r++)
    {
        for (int c = 0; c < dst_norm.cols; c++)
        {
            int corner_value = (int)dst_norm_scaled.at<float>(r, c);
            if (corner_value > minResponse)
            {
                cv::KeyPoint new_key_point;
                new_key_point.pt = cv::Point2f(r,c);
                new_key_point.size = 2*apertureSize;
                new_key_point.response = corner_value;

                //perform a non-maximum suppression (NMS) in a local neighborhood around
                if(suppression ==  true)
                {
                    bool overlapped = false;
                    for (auto i = key_points.begin(); i != key_points.end(); ++i)
                    {
                        double newoverlap = cv::KeyPoint::overlap(new_key_point, *i);
                        if (newoverlap > overlap)
                        {
                            overlapped = true;
                            if(new_key_point.response > (*i).response)
                            {
                                *i = new_key_point;
                                break;
                            }
                        }
                    }
                    if(!overlapped)
                    {
                        key_points.push_back(new_key_point);
                    }
                }
                else
                {
                    key_points.push_back(new_key_point);
                }

            }
        }
    }
	
    t = ((double)cv::getTickCount() - t)/cv::getTickFrequency();
    cout << "HARRIS detection with n=" << key_points.size() << " keypoints in " << 1000 * t / 1.0 << " ms" << endl;

    if(bVis)
    {
    	cv::Mat cornersImg = img.clone();
    	cv::drawKeypoints(img, key_points, cornersImg, cv::Scalar::all(-1), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
    	string windowName = "Harris Corner Detector Results";  
    	cv::imshow(windowName, cornersImg);
    	cv::waitKey(0);
    } 
}

void detKeypointsModern(vector<cv::KeyPoint> &keypoints, cv::Mat &img, string detectorType, bool bVis)
{
        // select appropriate descriptor
    cv::Ptr<cv::FeatureDetector> detector;
    if (detectorType.compare("BRISK") == 0)
    {

        int threshold = 30;        // FAST/AGAST detection threshold score.
        int octaves = 3;           // detection octaves (use 0 to do single scale)
        float patternScale = 1.0f; // apply this scale to the pattern used for sampling the neighbourhood of a keypoint.

        detector = cv::BRISK::create(threshold, octaves, patternScale);
    }
    else if (detectorType.compare("FAST") == 0)
    {
        int threshold = 30;
        bool bNms = true;
        cv::FastFeatureDetector::DetectorType type = cv::FastFeatureDetector::TYPE_9_16;

        detector = cv::FastFeatureDetector::create(threshold, bNms, type);
    }
    else if (detectorType.compare("ORB") == 0)
    {
        int features = 500;
        float scale = 1.2f;
        int nlevels = 8;
        int edgeThr = 31;
        int firstLevel = 0;
        int WATK = 2;
        cv::ORB::ScoreType scoreType = cv::ORB::HARRIS_SCORE;
        int patchSize = 31;
        int fastThr = 20;

        detector = cv::ORB::create(features, scale, nlevels, edgeThr, firstLevel, WATK, scoreType, patchSize, fastThr);
    }
    else if (detectorType.compare("AKAZE") == 0)
    {
        cv::AKAZE::DescriptorType descriptorType = cv::AKAZE::DESCRIPTOR_MLDB;
        int descriptorsize = 0;
        int descriptorChannels = 3;
        float threshold = 0.001f;
        int nOctaves = 4;
        int nOctaveLayers = 4;
        cv::KAZE::DiffusivityType diffus = cv::KAZE::DIFF_PM_G2;

        detector = cv::AKAZE::create(descriptorType, descriptorsize, descriptorChannels, threshold, nOctaves, nOctaveLayers, diffus);
    }
    else if (detectorType.compare("SIFT") == 0)
    {
        int nfeatures = 0;
        int nOctaveLayers = 3; 
        double contrastThreshold = 0.04;
        double edgeThreshold = 10.0;
        double sigma = 1.6;

        detector = cv::xfeatures2d::SIFT::create(nfeatures, nOctaveLayers, contrastThreshold, edgeThreshold, sigma);
    }

    // perform feature description
    double t = (double)cv::getTickCount();
    detector->detect(img, keypoints);
    t = ((double)cv::getTickCount() - t) / cv::getTickFrequency();
    cout << detectorType << "detection with n = " << keypoints.size() << "keypoints in " << 1000 * t / 1.0 << " ms" << endl;

    if(bVis)
    {
    	cv::Mat cornersImg = img.clone();
    	cv::drawKeypoints(img, keypoints, cornersImg, cv::Scalar::all(-1), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
    	string windowName = detectorType + "Detector Results";  
    	cv::imshow(windowName, cornersImg);
    	cv::waitKey(0);
    }
}