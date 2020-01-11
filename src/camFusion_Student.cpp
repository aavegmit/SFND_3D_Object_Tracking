
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
        pt.x = Y.at<double>(0, 0) / Y.at<double>(0, 2); // pixel coordinates
        pt.y = Y.at<double>(1, 0) / Y.at<double>(0, 2);

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

double distance(cv::Point2f p, cv::Point2f q) {
    return sqrt ((p.x - q.x) * (p.x - q.x) + (p.y - q.y) * (p.y - q.y));
}

// associate a given bounding box with the keypoints it contains
void clusterKptMatchesWithROI(BoundingBox &boundingBox, std::vector<cv::KeyPoint> &kptsPrev, std::vector<cv::KeyPoint> &kptsCurr, std::vector<cv::DMatch> &kptMatches)
{
    // Loop over the matches and fetch the corresponding prev and curr keypoint
    std::vector<double> distances;
    std::map<cv::DMatch, double> matchToDistance;

    for (auto match : kptMatches) {
        auto prevKeypoint = kptsPrev[match.queryIdx];
        auto currKeypoint = kptsCurr[match.trainIdx];
        // If the curr keypoint is in the ROI
        if (boundingBox.roi.contains(currKeypoint.pt)) {
            // Find the distance between the two points
            double dist = distance(prevKeypoint.pt, currKeypoint.pt);
            // std::cout << "Distance between matches: " << dist 
            // << " prev pt:(" << prevKeypoint.pt.x << "," << prevKeypoint.pt.y << ") - "
            // << "curr pt:(" << currKeypoint.pt.x << "," << currKeypoint.pt.y << ")"
            // << std::endl;
            distances.push_back(dist);
            matchToDistance[match] = dist;
        }
    }

    // Find the median distance
    sort(distances.begin(), distances.end());
    double medianDist = distances[distances.size() * 0.5]; 

    // Loop over the mathches_distance map 
    for (auto it = matchToDistance.begin(); it != matchToDistance.end(); ++it) {
        if (it->second < (medianDist * 0.50) || it->second > (medianDist * 1.50)) {
            // This is an outlier, do not consider for the match
        }
        else {
            // add the match to the result vector if the distance is +- 50% of the median
            boundingBox.kptMatches.push_back(it->first);
        }
    }

}


// Compute time-to-collision (TTC) based on keypoint correspondences in successive images
void computeTTCCamera(std::vector<cv::KeyPoint> &kptsPrev, std::vector<cv::KeyPoint> &kptsCurr, 
                      std::vector<cv::DMatch> kptMatches, double frameRate, double &TTC, cv::Mat *visImg)
{
    vector<double> distRatios;
    for (auto it1 = kptMatches.begin(); it1 != kptMatches.end(); it1++) {
        cv::KeyPoint kpOuterCurr = kptsCurr.at(it1->trainIdx); 
        cv::KeyPoint kpOuterPrev = kptsPrev.at(it1->queryIdx);
        for (auto it2 = kptMatches.begin() + 1; it2 != kptMatches.end(); ++it2) {
            cv::KeyPoint kpInnerCurr = kptsCurr.at(it2->trainIdx);  // kptsCurr is indexed by trainIdx, see NOTE in matchBoundinBoxes
            cv::KeyPoint kpInnerPrev = kptsPrev.at(it2->queryIdx);  // kptsPrev is indexed by queryIdx, see NOTE in matchBoundinBoxes

            // Use cv::norm to calculate the current and previous Euclidean distances between each keypoint in the pair
            double distCurr = cv::norm(kpOuterCurr.pt - kpInnerCurr.pt);
            double distPrev = cv::norm(kpOuterPrev.pt - kpInnerPrev.pt);

            double minDist = 100.0;  // Threshold the calculated distRatios by requiring a minimum current distance between keypoints 

            // Avoid division by zero and apply the threshold
            if (distPrev > std::numeric_limits<double>::epsilon() && distCurr >= minDist) {
                double distRatio = distCurr / distPrev;
                distRatios.push_back(distRatio);
            }
        }
    }

     // Only continue if the vector of distRatios is not empty
    if (distRatios.size() == 0)
    {
        TTC = std::numeric_limits<double>::quiet_NaN();
        return;
    }

    // As with computeTTCLidar, use the median as a reasonable method of excluding outliers
    std::sort(distRatios.begin(), distRatios.end());
    double medianDistRatio = distRatios[distRatios.size() / 2];

    // Finally, calculate a TTC estimate based on these 2D camera features
    TTC = (-1.0 / frameRate) / (1 - medianDistRatio);
}

// Find the closest point in X direction after removing the outliers
// We remove the outliers by considering the 50th percentile value
// instead of the min (100th percentile)
double closestPointWithoutOutliers(std::vector<LidarPoint> &lidarPoints) {
    double perc = .50;
    vector<double> lidarPointsX;
    for (auto point : lidarPoints) {
        lidarPointsX.push_back(point.x);
    }
    sort(lidarPointsX.begin(), lidarPointsX.end());
    int ind = lidarPointsX.size() * (1 - perc);
    // std::cout << "Min point in Lidar is : " << lidarPointsX[0] 
    // << " 99.9 percentile: " << lidarPointsX[ind] 
    // << " Max: " << lidarPointsX[lidarPointsX.size() - 1] << std::endl;
    return lidarPointsX[ind];
}

void computeTTCLidar(std::vector<LidarPoint> &lidarPointsPrev,
                     std::vector<LidarPoint> &lidarPointsCurr, double frameRate, double &TTC)
{
    double dt = 1 / frameRate;

    // Find the closest point in previous frame
    double minXPrev = closestPointWithoutOutliers(lidarPointsPrev);

    // Find the closest point in current frame
    double minXCurr = closestPointWithoutOutliers(lidarPointsCurr);

    // calculate the TTC
    TTC = minXCurr * dt / (minXPrev - minXCurr); 
    std::cout << "Frame rate: " << frameRate << " TTC: " << TTC << std::endl;
}

vector<int> findBoundingBoxes(DataFrame &frame, cv::KeyPoint keypoint) {
    vector<int> boundingBoxes;
    for (auto it = frame.boundingBoxes.begin(); it != frame.boundingBoxes.end(); ++it) {
        if ((*it).roi.contains(keypoint.pt)) {
            boundingBoxes.push_back((*it).boxID);
        }
    }
    return boundingBoxes;
}

void initialise_map_or_nothing(std::map<int, std::map<int, int>> &mmap, int idx1, int idx2) {

    if (mmap.find(idx1) == mmap.end() || mmap[idx1].find(idx2) == mmap[idx1].end()) {
        mmap[idx1][idx2] = 0;
    }
    return;
}

void matchBoundingBoxes(std::vector<cv::DMatch> &matches, std::map<int, int> &bbBestMatches, DataFrame &prevFrame, DataFrame &currFrame)
{

    // Multimap of (current bounding box id, prev bounding box id)
    //std::multimap<int, int> boundingBoxesMatches;
    std::map<int, std::map<int, int>> boundingBoxesMatches;

    for (auto it = matches.begin(); it != matches.end(); ++it) {
        auto prevKeypoint = prevFrame.keypoints[(*it).queryIdx];
        auto currKeypoint = currFrame.keypoints[(*it).trainIdx];
        
        auto prevBoundingBoxes = findBoundingBoxes(prevFrame, prevKeypoint);
        auto currBoundingBoxes = findBoundingBoxes(currFrame, currKeypoint);

        if (prevBoundingBoxes.size() > 0 && currBoundingBoxes.size() > 0) {
            //std::cout << "Box id: " << prevBoundingBoxes[0] << " matches with box id: " << currBoundingBoxes[0] << std::endl;
            for (int pi = 0; pi < prevBoundingBoxes.size(); ++pi) {
                for (int ci = 0; ci < currBoundingBoxes.size(); ++ci) {
                    //boundingBoxesMatches.insert(std::pair <int, int> (currBoundingBoxes[ci], prevBoundingBoxes[pi])) ;
                    initialise_map_or_nothing(boundingBoxesMatches,currBoundingBoxes[ci],prevBoundingBoxes[pi]);
                    ++boundingBoxesMatches[currBoundingBoxes[ci]][prevBoundingBoxes[pi]];
                }
            }
        }
    }

    for (auto itr = boundingBoxesMatches.begin(); itr != boundingBoxesMatches.end(); ++itr) {
        int prevBB = -1;
        int maxCount = -1;
        for (auto itr2 = itr->second.begin(); itr2 != itr->second.end(); ++itr2) {
            // std::cout << "Box id: " 
            // << itr->first 
            // << " matches with box id: " 
            // << itr2->first 
            // << " "
            // << itr2->second
            // << " times" 
            // << std::endl;

            if (itr2->second > maxCount) {
                maxCount = itr2->second;
                prevBB = itr2->first;
            }
        } // Inner loop ends

        if (prevBB > -1) {
            bbBestMatches[prevBB] = itr->first;
            //std::cout << "** Current box id: " << itr->first << " matched with " << prevBB << std::endl;
        }
    }

}
