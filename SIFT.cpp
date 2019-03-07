#include <iostream>
#include <vector>
#include <cmath>
#include <string>
#include <sstream>
#include <opencv2/opencv.hpp>
#include "opencv2/features2d/features2d.hpp"

using namespace std;
using namespace cv;

#define PI 3.14159265

static const double sigma = 1.6;
static const unsigned octave = 4;
static const unsigned octaveLayer = 5;
static const unsigned border = 5;
static const unsigned n_keypoint_bin = 36;
static const unsigned n_cell_bin = 8;
static const unsigned thres = 9;
static const unsigned cell_width = 4;
static const unsigned block_width = 4;
static const double match_ratio = 0.8;
static const double feature_thres = 0.2;

struct extrema
{
    double octave;
    int layer;
    Point2f pt;
    double angle;
    double magnitude;
    vector<double> keypoint_bin;
    vector<double> cell_bin;
    vector<vector<double> > NeighborCells;
};

pair<double, int> Max(const vector<double>& a) {
    double maximum = 1e-8;
    int mxIdx = 0;
    for(int i=0; i<a.size(); i++){
        if(a[i] > maximum){
            maximum = a[i];
            mxIdx = i;
        }
    }
    return pair<double, int>(maximum, mxIdx);
}

pair<double, int> Min(const vector<double>& a) {
    double smallest = 1e+8;
    int smIdx = 0;
    for(int i=0; i<a.size(); i++){
        if(a[i] < smallest){
            smallest = a[i];
            smIdx = i;
        }
    }
    return pair<double, int>(smallest, smIdx);
}

pair<double, int> secondMin(const vector<double>& a, const int& index){
    double SMin = 1e+8;
    int smIdx = 0;
    for(int i = 0; i<a.size(); i++){
        if (a[i] < SMin && i != index){
            SMin = a[i];
            smIdx = i;
        }
    }
    return pair<double, int>(SMin, smIdx);
}

void rescale(Mat& colorfulImage, Mat& image){
    cvtColor(colorfulImage, image, COLOR_BGR2GRAY);
    // Shrink the image size
    while(image.rows > 600){
        resize(image, image, Size(image.cols/2,image.rows/2)); 
        resize(colorfulImage, colorfulImage, Size(colorfulImage.cols/2,colorfulImage.rows/2));
    }
}

void buildGaussianPyramid(const Mat& image, vector<vector<Mat> >& scaleSpace){
    for(int i=0; i<octave; i++){
        vector<Mat> tmpScale;
        for(int j=0; j<octaveLayer; j++){
            Mat copyImage = image.clone();
            if(i > 0)
                resize(copyImage, copyImage, Size(copyImage.cols/pow(2,(i+1)), copyImage.rows/pow(2,(i+1))));
            GaussianBlur(copyImage, copyImage, Size(), pow(2,(j-1)/2.0 + i));
            // cout << pow(2,(j-1)/2.0 + i) << "   ";
            tmpScale.push_back(copyImage);
        }
        // cout << endl;
        scaleSpace.push_back(tmpScale);
    }
}

void buildDoGPyramid(const vector<vector<Mat> >& scaleSpace, vector<vector<Mat> >& dogScaleSpace){
    for(int i=0; i<octave; i++){
        vector<Mat> tmpDogScale;
        for(int j=0; j<octaveLayer-1; j++){
            Mat tmp = scaleSpace[i][j+1] - scaleSpace[i][j];
            // sharpen2(tmp,tmp);
            tmpDogScale.push_back(tmp);
        }
        dogScaleSpace.push_back(tmpDogScale);
    }
}

bool findExtrema(const vector<vector<Mat> > & dogScaleSpace, const double& val, const unsigned& i, const unsigned& j, const unsigned& r, const unsigned& c, const double& threshold){
    if( val > threshold &&
    ((val >= (double)dogScaleSpace[i][j].at<uchar>(r-1,c-1)    && val >= (double)dogScaleSpace[i][j].at<uchar>(r,c-1)       && val >= (unsigned)dogScaleSpace[i][j].at<uchar>(r+1,c-1)    &&
    val >= (double)dogScaleSpace[i][j].at<uchar>(r-1,c)      && val >= (double)dogScaleSpace[i][j].at<uchar>(r+1,c)       &&
    val >= (double)dogScaleSpace[i][j].at<uchar>(r-1,c+1)    && val >= (double)dogScaleSpace[i][j].at<uchar>(r,c+1)       && val >= (double)dogScaleSpace[i][j].at<uchar>(r+1,c+1)    &&
    val >= (double)dogScaleSpace[i][j-1].at<uchar>(r-1,c-1)  && val >= (double)dogScaleSpace[i][j-1].at<uchar>(r,c-1)     && val >= (double)dogScaleSpace[i][j-1].at<uchar>(r+1,c-1)  &&
    val >= (double)dogScaleSpace[i][j-1].at<uchar>(r-1,c)    && val >= (double)dogScaleSpace[i][j-1].at<uchar>(r,c)       && val >= (double)dogScaleSpace[i][j-1].at<uchar>(r+1,c)    &&
    val >= (double)dogScaleSpace[i][j-1].at<uchar>(r-1,c+1)  && val >= (double)dogScaleSpace[i][j-1].at<uchar>(r,c+1)     && val >= (double)dogScaleSpace[i][j-1].at<uchar>(r+1,c+1)  &&
    val >= (double)dogScaleSpace[i][j+1].at<uchar>(r-1,c-1)  && val >= (double)dogScaleSpace[i][j+1].at<uchar>(r,c-1)     && val >= (double)dogScaleSpace[i][j+1].at<uchar>(r+1,c-1)  &&
    val >= (double)dogScaleSpace[i][j+1].at<uchar>(r-1,c)    && val >= (double)dogScaleSpace[i][j+1].at<uchar>(r,c)       && val >= (double)dogScaleSpace[i][j+1].at<uchar>(r+1,c)    &&
    val >= (double)dogScaleSpace[i][j+1].at<uchar>(r-1,c+1)  && val >= (double)dogScaleSpace[i][j+1].at<uchar>(r,c+1)     && val >= (double)dogScaleSpace[i][j+1].at<uchar>(r+1,c+1)) ||
    // Smaller than other 26 pixels
    (val <= (double)dogScaleSpace[i][j].at<uchar>(r-1,c-1)    && val <= (double)dogScaleSpace[i][j].at<uchar>(r,c-1)       && val <= (double)dogScaleSpace[i][j].at<uchar>(r+1,c-1)     &&
    val <= (double)dogScaleSpace[i][j].at<uchar>(r-1,c)      && val <= (double)dogScaleSpace[i][j].at<uchar>(r+1,c)       &&
    val <= (double)dogScaleSpace[i][j].at<uchar>(r-1,c+1)    && val <= (double)dogScaleSpace[i][j].at<uchar>(r,c+1)       && val <= (double)dogScaleSpace[i][j].at<uchar>(r+1,c+1)     &&
    val <= (double)dogScaleSpace[i][j-1].at<uchar>(r-1,c-1)  && val <= (double)dogScaleSpace[i][j-1].at<uchar>(r,c-1)     && val <= (double)dogScaleSpace[i][j-1].at<uchar>(r+1,c-1)   &&
    val <= (double)dogScaleSpace[i][j-1].at<uchar>(r-1,c)    && val <= (double)dogScaleSpace[i][j-1].at<uchar>(r,c)       && val <= (double)dogScaleSpace[i][j-1].at<uchar>(r+1,c)     &&
    val <= (double)dogScaleSpace[i][j-1].at<uchar>(r-1,c+1)  && val <= (double)dogScaleSpace[i][j-1].at<uchar>(r,c+1)     && val <= (double)dogScaleSpace[i][j-1].at<uchar>(r+1,c+1)   &&
    val <= (double)dogScaleSpace[i][j+1].at<uchar>(r-1,c-1)  && val <= (double)dogScaleSpace[i][j+1].at<uchar>(r,c-1)     && val <= (double)dogScaleSpace[i][j+1].at<uchar>(r+1,c-1)   &&
    val <= (double)dogScaleSpace[i][j+1].at<uchar>(r-1,c)    && val <= (double)dogScaleSpace[i][j+1].at<uchar>(r,c)       && val <= (double)dogScaleSpace[i][j+1].at<uchar>(r+1,c)     &&
    val <= (double)dogScaleSpace[i][j+1].at<uchar>(r-1,c+1)  && val <= (double)dogScaleSpace[i][j+1].at<uchar>(r,c+1)     && val <= (double)dogScaleSpace[i][j+1].at<uchar>(r+1,c+1))))
        return true;
    else
        return false;
}

double calAngle(double tangentX, double tangentY)
{
    double param = tangentY / double(tangentX);
    double tol = pow(10,-8);
    double angle;
    if(abs(tangentX - 0) < tol && abs(tangentY - 0) < tol)
        angle = tol;
    else if(abs(tangentX - 0) < tol && tangentY > 99999)
        angle = 90;
    else if(abs(tangentX - 0) < tol && tangentY < -99999)
        angle = 270;
    else{
        angle = atan(param) * 180 / PI;
        if(tangentX < 0 && tangentY < 0)
            angle += 180;
        else if(tangentX < 0 && tangentY > 0)
            angle += 180;
    }
    if(angle < 0)
        angle += 360;
    return angle;
}

void Normalize(vector<double>& tmp){
    double maximum = Max(tmp).first;
    for(auto &e:tmp)
      e /= maximum;
}

void ReNormalize(vector<double>& tmp){
    for(auto& e:tmp){
        if(e > feature_thres){
            e = 0;
        }
    }
    double maximum = Max(tmp).first;
    for(auto &e:tmp)
      e /= maximum;
}

double weight(const int& i, const int& j, const double& var) {
    return 1 / (2 * M_PI) * std::exp(-0.5 * (i * i + j * j) / var / var);
}

vector<double> calcOrientationHist(const Mat& image, const Point2f& tmp, const Point2f& center, const double& angle){
    vector<double> tmpBin(n_cell_bin);
    Mat rot_mat = getRotationMatrix2D(tmp, angle, 1.0);
    Mat tmp_image;
    warpAffine(image, tmp_image, rot_mat, image.size());
    // imshow("test", tmp_image);
    // waitKey(0);

    for(unsigned r = tmp.y; r < tmp.y+cell_width; r++){
        for(unsigned c = tmp.x; c < tmp.x+cell_width; c++){
            double neighborC, neighborR, neighborAngle, neighborParam, neighborMag;
            neighborC = (double)image.at<uchar>(r, c+1) - (double)image.at<uchar>(r, c-1);
            neighborR = (double)image.at<uchar>(r+1, c) - (double)image.at<uchar>(r-1, c);
            neighborAngle = calAngle(neighborC, neighborR);
            neighborMag = pow(neighborC * neighborC + neighborR * neighborR, 0.5);
            neighborMag *= weight(c-center.x, r-center.y, 0.5 * block_width);

            if( 0 <= int(neighborAngle/45) && int(neighborAngle/45) <= 9){
                int bin = neighborAngle/45;
                double proportion = (neighborAngle-bin*45)/45.0 - 0.5;
                // cout << bin << "    " << proportion << "    " << 1-proportion << endl;
                
                if(proportion > 0){
                    tmpBin[bin] += (1-proportion) * neighborMag;
                    if(bin < tmpBin.size()-1){
                        tmpBin[bin+1] += proportion * neighborMag;
                    }
                    else
                        tmpBin[0] += proportion * neighborMag;
                }
                else{
                    tmpBin[bin] += 1-abs(proportion) * neighborMag;
                    if(bin > 0)
                        tmpBin[bin-1] += abs(proportion) * neighborMag;
                    else
                        tmpBin[tmpBin.size()-1] += abs(proportion) * neighborMag;
                }
            }
        }
    }
    Normalize(tmpBin);
    // ReNormalize(tmpBin);
    return tmpBin;
}

vector<double> keypoint_Orientation(const Mat& image, const Point2f& center, const double& sigma){
    vector<double> KeyOrient(36);
    for(int r = center.y - 7; r < center.y + 7; r++){
        for(int c = center.x - 7; c < center.x + 7; c++){
            if(r*r+c*c <= 7){
                double tangentR, tangentC, magnitude, angle;
                tangentC = (double)image.at<uchar>(r, c+1) - (double)image.at<uchar>(r, c-1);
                tangentR = (double)image.at<uchar>(r+1, c) - (double)image.at<uchar>(r-1, c);
                angle = calAngle(tangentC, tangentR);
                magnitude = pow(tangentC * tangentC + tangentR * tangentR, 0.5);
                magnitude *= weight(c-center.x, r-center.y, sigma);

                if( 0 <= int(angle/10) && int(angle/10) <= KeyOrient.size()-1){
                    int bin = angle/10;
                    double proportion = (angle-bin*10)/10.0 - 0.5;
                    KeyOrient[bin] += 1-abs(proportion) * magnitude;
                    if(proportion > 0){
                        if(bin < KeyOrient.size()-1){
                            KeyOrient[bin+1] += proportion * magnitude;
                        }
                        else
                            KeyOrient[0] += proportion * magnitude;
                    }
                    else{
                        if(bin > 0)
                            KeyOrient[bin-1] += abs(proportion) * magnitude;
                        else
                            KeyOrient[KeyOrient.size()-1] += abs(proportion) * magnitude;
                    }
                }
            }
        }
    }
    return KeyOrient;
}

void findScaleSpaceExtrema(vector<extrema>& KP, const vector<vector<Mat> >& dogScaleSpace, const Mat& image, unsigned& cnt){
    for(unsigned i = 0; i < octave; i++){
        for(unsigned j = 1; j < dogScaleSpace[i].size()-1; j++){
            for(unsigned r = border; r < dogScaleSpace[i][j].rows-border; r++){
                for(unsigned c = border; c < dogScaleSpace[i][j].cols-border; c++){
                    cnt ++;
                    double val = (double)dogScaleSpace[i][j].at<uchar>(r,c);
                    if( findExtrema(dogScaleSpace, val, i, j, r, c, thres)){
                        extrema tmpKp;
                        double tangentR, tangentC, angle;
                        tmpKp.octave = i;
                        tmpKp.layer = j;
                        const Mat& img = dogScaleSpace[i][j];
                        // tmpKp.pt.y = r / (double)img.cols * (double)image.cols;
                        // tmpKp.pt.x = c / (double)img.rows * (double)image.rows;
                        tmpKp.pt = Point2f (c / (double)img.rows * (double)image.rows, r / (double)img.cols * (double)image.cols);

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
                        const float img_scale = 1.f/(255);
                        const float deriv_scale = img_scale*0.5f;
                        const float second_deriv_scale = img_scale;
                        const float cross_deriv_scale = img_scale*0.25f;

                        const Mat& prev = dogScaleSpace[i][j-1];
                        const Mat& next = dogScaleSpace[i][j+1];
                        float xi, xr, xc, contr;

                        Vec3f dD((img.at<uchar>(r, c+1) - img.at<uchar>(r, c-1))*deriv_scale,
                                (img.at<uchar>(r+1, c) - img.at<uchar>(r-1, c))*deriv_scale,
                                (next.at<uchar>(r, c) - prev.at<uchar>(r, c))*deriv_scale);
                        
                        float v2 = (float)img.at<uchar>(r, c)*2;
                        float dxx = (img.at<uchar>(r, c+1) + img.at<uchar>(r, c-1) - v2)*second_deriv_scale;
                        float dyy = (img.at<uchar>(r+1, c) + img.at<uchar>(r-1, c) - v2)*second_deriv_scale;
                        float dss = (next.at<uchar>(r, c) + prev.at<uchar>(r, c) - v2)*second_deriv_scale;
                        float dxy = (img.at<uchar>(r+1, c+1) - img.at<uchar>(r+1, c-1) -
                                    img.at<uchar>(r-1, c+1) + img.at<uchar>(r-1, c-1))*cross_deriv_scale;
                        float dxs = (next.at<uchar>(r, c+1) - next.at<uchar>(r, c-1) -
                                    prev.at<uchar>(r, c+1) + prev.at<uchar>(r, c-1))*cross_deriv_scale;
                        float dys = (next.at<uchar>(r+1, c) - next.at<uchar>(r-1, c) -
                                    prev.at<uchar>(r+1, c) + prev.at<uchar>(r-1, c))*cross_deriv_scale;
                        
                        Matx33f H(dxx, dxy, dxs,
                                  dxy, dyy, dys,
                                  dxs, dys, dss);

                        Vec3f X = H.solve(dD, DECOMP_LU);

                        xi = -X[2];
                        xr = -X[1];
                        xc = -X[0];

                        if( std::abs(xi) < 0.5f && std::abs(xr) < 0.5f && std::abs(xc) < 0.5f )
                            break;
                        
                        Matx31f dD2((img.at<uchar>(r, c+1) - img.at<uchar>(r, c-1))*deriv_scale,
                                   (img.at<uchar>(r+1, c) - img.at<uchar>(r-1, c))*deriv_scale,
                                   (next.at<uchar>(r, c) - prev.at<uchar>(r, c))*deriv_scale);
                        float t = dD2.dot(Matx31f(xc, xr, xi));
                        contr = img.at<uchar>(r,c)*img_scale + t*0.5f;
                        if( abs( contr ) * octaveLayer < 0.03 )
                            break;
                        cout << abs( contr ) * octaveLayer << endl;
                        
                        double edgeThreshold = 10.0;
                        float tr = dxx + dyy;
                        float det = dxx * dyy - dxy * dxy;
                        if( det <= 0 || tr*tr*edgeThreshold >= (edgeThreshold + 1)*(edgeThreshold + 1)*det )
                            break;

                        // cout << xi << "  "<< xr << "  "<< xc <<endl;
                        // if(std::abs(xi) > 1 || std::abs(xr) > 1 || std::abs(xc) < 1)
                        //     break;
                        tmpKp.pt.x += cvRound(xc);
                        tmpKp.pt.y += cvRound(xr);
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
                        tmpKp.keypoint_bin.resize(36);
                        tmpKp.cell_bin.resize(8);
                        tangentC = (double)image.at<uchar>(tmpKp.pt.y, tmpKp.pt.x+1) - (double)image.at<uchar>(tmpKp.pt.y, tmpKp.pt.x-1);
                        tangentR = (double)image.at<uchar>(tmpKp.pt.y+1, tmpKp.pt.x) - (double)image.at<uchar>(tmpKp.pt.y-1, tmpKp.pt.x);
                        angle = calAngle(tangentC, tangentR);
                        tmpKp.angle = angle;
                        tmpKp.magnitude = pow(tangentC * tangentC + tangentR * tangentR, 0.5);
                        tmpKp.keypoint_bin = keypoint_Orientation(image, tmpKp.pt, 1.5 * pow(2,(tmpKp.layer-1)/2.0 + tmpKp.octave));

                        for(unsigned row_cell = 0; row_cell < block_width; row_cell++){
                            for(unsigned col_cell = 0; col_cell < block_width; col_cell++){
                                vector<double> tmpCell;
                                tmpCell = calcOrientationHist(image, Point2f(tmpKp.pt.x + col_cell*block_width - 7, tmpKp.pt.y + row_cell*block_width - 7),
                                                                tmpKp.pt, Max(tmpKp.keypoint_bin).second * 10);
                                tmpKp.NeighborCells.push_back(tmpCell);
                            }
                        }
                        KP.push_back(tmpKp);
                    }
                }
            }
        }
    }
}

vector<extrema> SIFTDescript(const Mat& image, unsigned& n_pixels){
    vector<vector<Mat> > scaleSpace;
    vector<vector<Mat> > dogScaleSpace;
    vector<extrema> KP;
    buildGaussianPyramid(image, scaleSpace);
    buildDoGPyramid(scaleSpace, dogScaleSpace);
    findScaleSpaceExtrema(KP, dogScaleSpace, image, n_pixels);
    return KP;
}

void findMatches(const vector<extrema>& ex1, const vector<extrema>& ex2, vector<Point>& matchKP1, vector<Point>& matchKP2, unsigned& cnt_match){
    double min_dist = 1000;
    for(const auto& i:ex1){
        for(const auto& j:ex2){
            double dist = 0;
            for(int k = 0; k < j.NeighborCells.size(); k++){
                for(int l = 0; l < j.NeighborCells[k].size(); l++){
                    dist += pow(i.NeighborCells[k][l] - j.NeighborCells[k][l], 2);
                }
            }
            if(dist < min_dist) min_dist = dist;
        }
    }

    vector<vector<double> > everyDist;
    for(int i = 0; i < ex1.size(); i++){
        vector<double> tmpEveryDist;
        for(int j = 0; j < ex2.size(); j++){
            double tmpSum = 0;
            for(int k = 0; k < ex2[j].NeighborCells.size(); k++){
                for(int l = 0; l < ex2[j].NeighborCells[k].size(); l++){
                    tmpSum += pow(ex1[i].NeighborCells[k][l] - ex2[j].NeighborCells[k][l], 2);
                }
            }
            tmpEveryDist.push_back(tmpSum);
        }
        everyDist.push_back(tmpEveryDist);
    }

    vector<double> everyMinDist, everySecondMinDist;
    vector<int> MinIdx;
    for(int i = 0; i < everyDist.size(); i++){
        everyMinDist.push_back(Min(everyDist[i]).first);
        MinIdx.push_back(Min(everyDist[i]).second);
    }
    vector<int> SecondMinIdx;
    for(int i = 0; i < everyDist.size(); i++){
        everySecondMinDist.push_back(secondMin(everyDist[i], MinIdx[i]).first);
        SecondMinIdx.push_back(secondMin(everyDist[i], MinIdx[i]).second);
    }
    for(int i = 0; i < everyDist.size(); i++){
        if(everyMinDist[i]/double(everySecondMinDist[i]) < match_ratio || everyMinDist[i] < 1 ){
            cout << "(" << ex1[i].pt.x << ", " << ex1[i].pt.y << ") match ("
                << ex2[MinIdx[i]].pt.x << ", " << ex2[MinIdx[i]].pt.y << ")" << endl;
            matchKP1.push_back(ex1[i].pt);
            matchKP2.push_back(ex2[MinIdx[i]].pt);
            cnt_match++;
        }
    }
}

int main( int argc, char** argv )
{
    vector<extrema> ex1, ex2;
    Mat image, colorfulImage;
    unsigned n_pixels1, n_pixels2;
    colorfulImage = imread("/Users/hsjimwang/Desktop/SIFT/cookbeef1.jpg");
    cvtColor(colorfulImage, image, COLOR_BGR2GRAY);
    rescale(colorfulImage, image);
    ex1 = SIFTDescript(image, n_pixels1);
    cout << "From " << n_pixels1 << " point, ";
    cout << "we extracted " << ex1.size() << " interest points." << endl;

    Mat image2, colorfulImage2;
    colorfulImage2 = imread("/Users/hsjimwang/Desktop/SIFT/cookbeef2.jpg");
    cvtColor(colorfulImage2, image2, COLOR_BGR2GRAY);
    rescale(colorfulImage2, image2);
    ex2 = SIFTDescript(image2, n_pixels2);
    cout << "From " << n_pixels2 << " point, ";
    cout << "we extracted " << ex2.size() << " interest points." << endl;
    for(auto e:ex2){
        circle(colorfulImage2, e.pt, 3, Scalar(0,255,255));
    }

    for(const auto& e:ex1){
        circle(colorfulImage, e.pt, 3, Scalar(0,255,255));
    }

    imshow("show", colorfulImage);
    waitKey(0);
    imshow("show", colorfulImage2);
    waitKey(0);

    vector<Point> matchKP1, matchKP2;
    unsigned cnt_match = 0;
    findMatches(ex1, ex2, matchKP1, matchKP2, cnt_match);
    cout << "We matched " << cnt_match << " pairs." << endl;

    // Show the matches
    int height = colorfulImage.rows;
    if(colorfulImage.rows < colorfulImage2.rows) height = colorfulImage2.rows;
     Mat matches(height, colorfulImage.cols + colorfulImage2.cols, CV_8UC3);
    for(int r = 0; r < colorfulImage.rows; r++){
        for(int c = 0; c < colorfulImage.cols; c++)
            matches.at<Vec3b>(r,c) = colorfulImage.at<Vec3b>(r,c);
    }
    for(int r = 0; r < colorfulImage2.rows; r++){
        for(int c = 0; c < colorfulImage2.cols; c++)
            matches.at<Vec3b>(r,c+colorfulImage.cols) = colorfulImage2.at<Vec3b>(r,c);
    }
    for(int i=0; i<matchKP1.size(); i++){
        line(matches,matchKP1[i],Point(matchKP2[i].x+colorfulImage.cols, matchKP2[i].y),Scalar(200,200,0));
    }
    imshow("ShowMatches", matches);
    waitKey(0);
    return 0;
}
