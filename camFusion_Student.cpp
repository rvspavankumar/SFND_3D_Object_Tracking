
#include <iostream>
#include <algorithm>
#include <numeric>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include "camFusion.hpp"
#include "dataStructures.h"

using namespace std;


// Create groups of Lidar points whose projection into the camera falls into the same bounding box
void clusterLidarWithROI(std::vector<BoundingBox> &boundingBoxes, std::vector<LidarPoint> &lidarPoints, float shrinkFactor, cv::Mat &P_rect_xx, cv::Mat &R_rect_xx, cv::Mat &RT)
{
    // loop over all Lidar points and associate them to a 2D bounding box
    cv::Mat X(4, 1, cv::DataType<double>::type);
    cv::Mat Y(3, 1, cv::DataType<double>::type);

    for (auto it1 = lidarPoints.begin(); it1 != lidarPoints.end(); ++it1)
    {
        // assemble vector for matrix-vector-multiplication
        X.at<double>(0, 0) = it1->x;
        X.at<double>(1, 0) = it1->y;
        X.at<double>(2, 0) = it1->z;
        X.at<double>(3, 0) = 1;

        // project Lidar point into camera
        Y = P_rect_xx * R_rect_xx * RT * X;
        cv::Point pt;
        // pixel coordinates
        pt.x = Y.at<double>(0, 0) / Y.at<double>(2, 0); 
        pt.y = Y.at<double>(1, 0) / Y.at<double>(2, 0); 

        vector<vector<BoundingBox>::iterator> enclosingBoxes; // pointers to all bounding boxes which enclose the current Lidar point
        for (vector<BoundingBox>::iterator it2 = boundingBoxes.begin(); it2 != boundingBoxes.end(); ++it2)
        {
            // shrink current bounding box slightly to avoid having too many outlier points around the edges
            cv::Rect smallerBox;
            smallerBox.x = (*it2).roi.x + shrinkFactor * (*it2).roi.width / 2.0;
            smallerBox.y = (*it2).roi.y + shrinkFactor * (*it2).roi.height / 2.0;
            smallerBox.width = (*it2).roi.width * (1 - shrinkFactor);
            smallerBox.height = (*it2).roi.height * (1 - shrinkFactor);

            // check wether point is within current bounding box
            if (smallerBox.contains(pt))
            {
                enclosingBoxes.push_back(it2);
            }

        } // eof loop over all bounding boxes

        // check wether point has been enclosed by one or by multiple boxes
        if (enclosingBoxes.size() == 1)
        { 
            // add Lidar point to bounding box
            enclosingBoxes[0]->lidarPoints.push_back(*it1);
        }

    } // eof loop over all Lidar points
}

/* 
* The show3DObjects() function below can handle different output image sizes, but the text output has been manually tuned to fit the 2000x2000 size. 
* However, you can make this function work for other sizes too.
* For instance, to use a 1000x1000 size, adjusting the text positions by dividing them by 2.
*/
void show3DObjects(std::vector<BoundingBox> &boundingBoxes, cv::Size worldSize, cv::Size imageSize, bool bWait)
{
    // create topview image
    cv::Mat topviewImg(imageSize, CV_8UC3, cv::Scalar(255, 255, 255));

    for(auto it1=boundingBoxes.begin(); it1!=boundingBoxes.end(); ++it1)
    {
        // create randomized color for current 3D object
        cv::RNG rng(it1->boxID);
        cv::Scalar currColor = cv::Scalar(rng.uniform(0,150), rng.uniform(0, 150), rng.uniform(0, 150));

        // plot Lidar points into top view image
        int top=1e8, left=1e8, bottom=0.0, right=0.0; 
        float xwmin=1e8, ywmin=1e8, ywmax=-1e8;
        for (auto it2 = it1->lidarPoints.begin(); it2 != it1->lidarPoints.end(); ++it2)
        {
            // world coordinates
            float xw = (*it2).x; // world position in m with x facing forward from sensor
            float yw = (*it2).y; // world position in m with y facing left from sensor
            xwmin = xwmin<xw ? xwmin : xw;
            ywmin = ywmin<yw ? ywmin : yw;
            ywmax = ywmax>yw ? ywmax : yw;

            // top-view coordinates
            int y = (-xw * imageSize.height / worldSize.height) + imageSize.height;
            int x = (-yw * imageSize.width / worldSize.width) + imageSize.width / 2;

            // find enclosing rectangle
            top = top<y ? top : y;
            left = left<x ? left : x;
            bottom = bottom>y ? bottom : y;
            right = right>x ? right : x;

            // draw individual point
            cv::circle(topviewImg, cv::Point(x, y), 4, currColor, -1);
        }

        // draw enclosing rectangle
        cv::rectangle(topviewImg, cv::Point(left, top), cv::Point(right, bottom),cv::Scalar(0,0,0), 2);

        // augment object with some key data
        char str1[200], str2[200];
        sprintf(str1, "id=%d, #pts=%d", it1->boxID, (int)it1->lidarPoints.size());
        putText(topviewImg, str1, cv::Point2f(left-250, bottom+50), cv::FONT_ITALIC, 2, currColor);
        sprintf(str2, "xmin=%2.2f m, yw=%2.2f m", xwmin, ywmax-ywmin);
        putText(topviewImg, str2, cv::Point2f(left-250, bottom+125), cv::FONT_ITALIC, 2, currColor);  
    }

    // plot distance markers
    float lineSpacing = 2.0; // gap between distance markers
    int nMarkers = floor(worldSize.height / lineSpacing);
    for (size_t i = 0; i < nMarkers; ++i)
    {
        int y = (-(i * lineSpacing) * imageSize.height / worldSize.height) + imageSize.height;
        cv::line(topviewImg, cv::Point(0, y), cv::Point(imageSize.width, y), cv::Scalar(255, 0, 0));
    }

    // display image
    string windowName = "3D Objects";
    cv::namedWindow(windowName, 1);
    cv::imshow(windowName, topviewImg);

    if(bWait)
    {
        cv::waitKey(0); // wait for key to be pressed
    }
}


// associate a given bounding box with the keypoints it contains
void clusterKptMatchesWithROI(BoundingBox &boundingBox, std::vector<cv::KeyPoint> &kptsPrev, std::vector<cv::KeyPoint> &kptsCurr, std::vector<cv::DMatch> &kptMatches)
{
    std::vector<cv::DMatch> matchesInsideBox;
    std::vector<double> matchesDistance;

    for(auto it = kptMatches.begin(); it != kptMatches.end(); ++it)
    {
        cv::KeyPoint kyptPre = kptsPrev[it->queryIdx];
        cv::KeyPoint kyptCurr = kptsCurr[it->trainIdx];

        if(boundingBox.roi.contains(kyptCurr.pt))
        {
            matchesInsideBox.push_back(*it);
            matchesDistance.push_back(cv::norm(kyptCurr.pt - kyptPre.pt));
        }
    }

    double meanDistance = std::accumulate(matchesDistance.begin(), matchesDistance.end(), 0.0) / matchesDistance.size();
    for(int id = 0; id <matchesDistance.size(); ++id)
    {
        if (matchesDistance[id] < meanDistance)
        {
            boundingBox.keypoints.push_back(kptsCurr[matchesInsideBox[id].trainIdx]);
            boundingBox.kptMatches.push_back(matchesInsideBox[id]);
        }
    }

    bool Debug = false;

}


// Compute time-to-collision (TTC) based on keypoint correspondences in successive images
void computeTTCCamera(std::vector<cv::KeyPoint> &kptsPrev, std::vector<cv::KeyPoint> &kptsCurr, 
                      std::vector<cv::DMatch> kptMatches, double frameRate, double &TTC, cv::Mat *visImg)
{
    vector<double> disRatios;
    for (auto it1 = kptMatches.begin(); it1 != kptMatches.end() -1; ++it1)
    {
        cv::KeyPoint OuterkyptCurr = kptsCurr.at(it1->trainIdx);
        cv::KeyPoint OuterkyptPrev = kptsPrev.at(it1->queryIdx);

        for (auto it2 = kptMatches.begin() + 1; it2 != kptMatches.end(); ++it2)
        {
            double minDistance = 100;

            cv::KeyPoint InnerkyptCurr = kptsCurr.at(it2->trainIdx);
            cv::KeyPoint InnerkyptPrev = kptsPrev.at(it2->queryIdx);

            double distanceCurr = cv::norm(OuterkyptCurr.pt - InnerkyptCurr.pt);
            double distancePrev = cv::norm(OuterkyptPrev.pt - InnerkyptPrev.pt);

            if (distancePrev > std::numeric_limits<double>::epsilon() && distanceCurr >= minDistance) 
            {
                double distanceRatio = distanceCurr/distancePrev;
                disRatios.push_back(distanceRatio);
            }

        } 
    }

    if(disRatios.size() == 0)
    {
        TTC = NAN;
        return;
    }


    std::sort(disRatios.begin(), disRatios.end());
    long medianIndex = floor(disRatios.size() / 2.0);
    double medianDistRatio = (disRatios.size() % 2 == 0) ? (disRatios[medianIndex - 1] + disRatios[medianIndex]) /2.0 : disRatios[medianIndex];

    double dT = 1/frameRate;
    TTC = -dT / (1 - medianDistRatio);
    std::cout << "TTC Calculated by camer images is  " << TTC << std::endl;
}


void computeTTCLidar(std::vector<LidarPoint> &lidarPointsPrev,
                     std::vector<LidarPoint> &lidarPointsCurr, double frameRate, double &TTC)
{
    double dT = 1.0/frameRate; //time between two frames
    double laneWidth = 4.0;

    double percentage = 0.2;  //20% of libar points will only considerd
    int lidarPtPrev = lidarPointsPrev.size() * percentage;
    int lidarPtCurr = lidarPointsCurr.size() * percentage;

    if (lidarPointsPrev.size() < 10)
    {
        lidarPtPrev = lidarPointsPrev.size();
    }
    if (lidarPointsCurr.size() < 10)
    {
        lidarPtCurr = lidarPointsCurr.size();
    }

    //distance to the preceding vehicle based of lidar points
    double AvgxDisPrev = 1e9, AvgxDisCurr = 1e9;
    vector<double> xDisPrev, xDisCurr;

    for (auto i = lidarPointsPrev.begin(); i != lidarPointsPrev.end(); ++i)
    {
        if(abs(i->y) <= (laneWidth/2.0))
        {
            if (xDisPrev.size() < lidarPtPrev)
            {
                xDisPrev.push_back(i->x);
            }
            else
            {
                auto max = max_element(std::begin(xDisPrev), std::end(xDisPrev));
                if (i->x < *max)
                {
                    xDisPrev.erase(max);
                    xDisPrev.push_back(i->x);
                }
            }
            
        }   
    }
    AvgxDisPrev = std::accumulate(xDisPrev.begin(), xDisPrev.end(), 0.0) / lidarPtPrev;


    for (auto i = lidarPointsCurr.begin(); i != lidarPointsCurr.end(); ++i)
    {
        if(abs(i->y) <= (laneWidth/2.0))
        {
            if (xDisCurr.size() < lidarPtCurr)
            {
                xDisCurr.push_back(i->x);
            }
            else
            {
                auto max = max_element(std::begin(xDisCurr), std::end(xDisCurr));
                if (i->x < *max)
                {
                    xDisCurr.erase(max);
                    xDisCurr.push_back(i->x);
                }
            }
            
        }   
    }
    AvgxDisCurr = std::accumulate(xDisCurr.begin(), xDisCurr.end(), 0.0) / lidarPtCurr;

    TTC = (AvgxDisCurr * dT) / (AvgxDisPrev - AvgxDisCurr);
    std::cout << "TTC Calculated by lidar is  " << TTC << std::endl;

    bool Debug = false;
    if (Debug)
    {
        std::cout << "Average X distance previous cycle is  " << AvgxDisPrev << std::endl;
        std::cout << "Average X distance current cycle is  " << AvgxDisCurr << std::endl;
        //std::cout << "TTC Calculated by lidar is  " << TTC << std::endl;
    }

}


void matchBoundingBoxes(std::vector<cv::DMatch> &matches, std::map<int, int> &bbBestMatches, DataFrame &prevFrame, DataFrame &currFrame)
{
    int  count[prevFrame.boundingBoxes.size()][currFrame.boundingBoxes.size()] = {0};

    for (auto it1 = matches.begin(); it1 != matches.end(); ++it1)
    {
        int PrevKyptId = it1->queryIdx;
        int CurrKyptId = it1->trainIdx;

        cv::KeyPoint prevKypt = prevFrame.keypoints[PrevKyptId];
        cv::KeyPoint currKypt = currFrame.keypoints[CurrKyptId];

        std::vector<int> prevBoundingBoxids, currBoundingBoxids;

        for (auto it2 = prevFrame.boundingBoxes.begin(); it2 != prevFrame.boundingBoxes.end(); ++it2)
        {
            if (it2->roi.contains(prevKypt.pt))
            {
                prevBoundingBoxids.push_back(it2->boxID);
            }
        }


        for (auto it3 = currFrame.boundingBoxes.begin(); it3 != currFrame.boundingBoxes.end(); ++it3)
        {
            if (it3->roi.contains(currKypt.pt))
            {
                currBoundingBoxids.push_back(it3->boxID);
            }
        }

        for (auto prevId : prevBoundingBoxids)
        {
            for (auto currId : currBoundingBoxids)
            {
                count[prevId][currId]++;
            }
        }
    }

    for (int prvid = 0; prvid < prevFrame.boundingBoxes.size(); ++prvid)
    {
        int maxCount = 0;
        int maxId = 0;
        for (int currid = 0; currid < currFrame.boundingBoxes.size(); ++currid)
        {
            if(count[prvid][currid] > maxCount)
            {
                maxCount = count[prvid][currid];
                maxId = currid;
            }
        }
        bbBestMatches.insert({prvid, maxId});
    }
}
