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

struct extrema
{
    int octave;
    Point pt;
    double angle;
    double magnitude;
    vector<double> OrientationBin;
    vector<vector<double> > NeighborCells;
};

pair<double, int> Max(vector<double> a) {
    double maximum = 1e+8;
    int mxIdx = 0;
    for(int i=0; i<a.size(); i++){
        if(a[i] > maximum){
            maximum = a[i];
            mxIdx = i;
        }
    }
    return pair<double, int>(maximum, mxIdx);
}

pair<double, int> Min(vector<double> a) {
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

pair<double, int> secondMin(vector<double> a, int index){
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

double calMagnitude(const vector<double>& a){
    double sumMag = 0;
    for(const auto& e:a){
        sumMag += e*e;
    }
    return sumMag;
}

void rotateDescriptor(vector<vector<double> >& a){
    vector<double> vecMag;
    for(auto e:a)
        vecMag.push_back(calMagnitude(e));
    int maxIndex = Max(vecMag).second;
    vector<vector<double> > tmp;
    for(int i=0; i<a.size(); i++){
        if(i+maxIndex > a.size()-1)
            tmp.push_back(a[i+maxIndex-a.size()]);
        else
            tmp.push_back(a[i+maxIndex]);
    }
    a.clear();
    a.assign(tmp.begin(), tmp.end());
}

void sharpen(const Mat &src, Mat &dst){
    src.copyTo(dst);
    const int nChannels = src.channels();
    int heightLimit = src.rows - 1;
    int widthLimit = nChannels * (src.cols-1);
    for(int iH=1; iH<heightLimit; iH++){
        const uchar *prePtr = src.ptr<const uchar>(iH-1);
        const uchar *curPtr = src.ptr<const uchar>(iH);
        const uchar *nextPtr = src.ptr<const uchar>(iH+1);
        uchar *dstPtr = dst.ptr<uchar>(iH);
        for(int iW=nChannels; iW<widthLimit; iW++){
            dstPtr[iW] = saturate_cast<uchar>(5*curPtr[iW]-curPtr[iW-nChannels]-curPtr[iW+nChannels]-prePtr[iW]-nextPtr[iW]);
        }
    }
}

void sharpen2(const Mat &src, Mat &dst){
    Mat kernel(3,3,CV_32F,Scalar(0));
    kernel.at<float>(1,1) = 5.0;
    kernel.at<float>(0,1) = -1.0;
    kernel.at<float>(2,1) = -1.0;
    kernel.at<float>(1,0) = -1.0;
    kernel.at<float>(1,2) = -1.0;
    filter2D(src,dst,src.depth(),kernel);
}

bool findExtrema(vector<vector<Mat> > & dogScaleSpace, double val, unsigned i, unsigned j, unsigned a, unsigned b, double threshold){
    if( val > threshold &&
    ((val >= (double)dogScaleSpace[i][j].at<uchar>(a-1,b-1)    && val >= (double)dogScaleSpace[i][j].at<uchar>(a,b-1)       && val >= (unsigned)dogScaleSpace[i][j].at<uchar>(a+1,b-1)    &&
    val >= (double)dogScaleSpace[i][j].at<uchar>(a-1,b)      && val >= (double)dogScaleSpace[i][j].at<uchar>(a+1,b)       &&
    val >= (double)dogScaleSpace[i][j].at<uchar>(a-1,b+1)    && val >= (double)dogScaleSpace[i][j].at<uchar>(a,b+1)       && val >= (double)dogScaleSpace[i][j].at<uchar>(a+1,b+1)    &&
    val >= (double)dogScaleSpace[i][j-1].at<uchar>(a-1,b-1)  && val >= (double)dogScaleSpace[i][j-1].at<uchar>(a,b-1)     && val >= (double)dogScaleSpace[i][j-1].at<uchar>(a+1,b-1)  &&
    val >= (double)dogScaleSpace[i][j-1].at<uchar>(a-1,b)    && val >= (double)dogScaleSpace[i][j-1].at<uchar>(a,b)       && val >= (double)dogScaleSpace[i][j-1].at<uchar>(a+1,b)    &&
    val >= (double)dogScaleSpace[i][j-1].at<uchar>(a-1,b+1)  && val >= (double)dogScaleSpace[i][j-1].at<uchar>(a,b+1)     && val >= (double)dogScaleSpace[i][j-1].at<uchar>(a+1,b+1)  &&
    val >= (double)dogScaleSpace[i][j+1].at<uchar>(a-1,b-1)  && val >= (double)dogScaleSpace[i][j+1].at<uchar>(a,b-1)     && val >= (double)dogScaleSpace[i][j+1].at<uchar>(a+1,b-1)  &&
    val >= (double)dogScaleSpace[i][j+1].at<uchar>(a-1,b)    && val >= (double)dogScaleSpace[i][j+1].at<uchar>(a,b)       && val >= (double)dogScaleSpace[i][j+1].at<uchar>(a+1,b)    &&
    val >= (double)dogScaleSpace[i][j+1].at<uchar>(a-1,b+1)  && val >= (double)dogScaleSpace[i][j+1].at<uchar>(a,b+1)     && val >= (double)dogScaleSpace[i][j+1].at<uchar>(a+1,b+1)) ||
    // Smaller than other 26 pixels
    (val <= (double)dogScaleSpace[i][j].at<uchar>(a-1,b-1)    && val <= (double)dogScaleSpace[i][j].at<uchar>(a,b-1)       && val <= (double)dogScaleSpace[i][j].at<uchar>(a+1,b-1)     &&
    val <= (double)dogScaleSpace[i][j].at<uchar>(a-1,b)      && val <= (double)dogScaleSpace[i][j].at<uchar>(a+1,b)       &&
    val <= (double)dogScaleSpace[i][j].at<uchar>(a-1,b+1)    && val <= (double)dogScaleSpace[i][j].at<uchar>(a,b+1)       && val <= (double)dogScaleSpace[i][j].at<uchar>(a+1,b+1)     &&
    val <= (double)dogScaleSpace[i][j-1].at<uchar>(a-1,b-1)  && val <= (double)dogScaleSpace[i][j-1].at<uchar>(a,b-1)     && val <= (double)dogScaleSpace[i][j-1].at<uchar>(a+1,b-1)   &&
    val <= (double)dogScaleSpace[i][j-1].at<uchar>(a-1,b)    && val <= (double)dogScaleSpace[i][j-1].at<uchar>(a,b)       && val <= (double)dogScaleSpace[i][j-1].at<uchar>(a+1,b)     &&
    val <= (double)dogScaleSpace[i][j-1].at<uchar>(a-1,b+1)  && val <= (double)dogScaleSpace[i][j-1].at<uchar>(a,b+1)     && val <= (double)dogScaleSpace[i][j-1].at<uchar>(a+1,b+1)   &&
    val <= (double)dogScaleSpace[i][j+1].at<uchar>(a-1,b-1)  && val <= (double)dogScaleSpace[i][j+1].at<uchar>(a,b-1)     && val <= (double)dogScaleSpace[i][j+1].at<uchar>(a+1,b-1)   &&
    val <= (double)dogScaleSpace[i][j+1].at<uchar>(a-1,b)    && val <= (double)dogScaleSpace[i][j+1].at<uchar>(a,b)       && val <= (double)dogScaleSpace[i][j+1].at<uchar>(a+1,b)     &&
    val <= (double)dogScaleSpace[i][j+1].at<uchar>(a-1,b+1)  && val <= (double)dogScaleSpace[i][j+1].at<uchar>(a,b+1)     && val <= (double)dogScaleSpace[i][j+1].at<uchar>(a+1,b+1))))
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
        angle = 66666;
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

void createFilter(vector<vector<double> >& gKernel)
{
    gKernel.resize(17);
    for(auto& e:gKernel)
        e.resize(17);
    // set standard deviation to 1.6
    double sigma = 3;
    double r, s = 2.0 * sigma * sigma;
    // sum is for normalization
    double sum = 0.0;
    // generate 5x5 kernel
    for (int x = -8; x <= 8; x++){
        for(int y = -8; y <= 8; y++){
            r = sqrt(x*x + y*y);
            gKernel[x + 8][y + 8] = (exp(-(r*r)/s))/(M_PI * s);
            sum += gKernel[x + 8][y + 8];
        }
    }
    // normalize the Kernel
    for(int i = 0; i < 17; ++i)
        for(int j = 0; j < 17; ++j)
            gKernel[i][j] /= sum;
}


vector<double> calcOrientationHist(Mat& image, Point point1, Point point2, Point center, vector<vector<double> > smallFilter)
{
    vector<double> tmp(8);
    for(int x=point1.x, gi=0; x<point2.x, gi<4; x++, gi++){
        for(int y=point1.y, gj=0; y<point2.y, gj<4; y++, gj++){
            double neighborX, neighborY, neighborAngle, neighborParam, neighborMag, d;
            neighborX = (double)image.at<uchar>(x+1,y) - (double)image.at<uchar>(x-1,y);
            neighborY = (double)image.at<uchar>(x,y+1) - (double)image.at<uchar>(x-1,y-1);
            neighborAngle = calAngle(neighborX, neighborY);
            
            neighborMag = pow(neighborX * neighborX + neighborY * neighborY, 0.5);
            d = pow((x-center.x)*(x-center.x) + (y-center.y)*(y-center.y), 0.5) / pow(4*4+4*4, 0.5);
            if( 0 <= int(neighborAngle/45) && int(neighborAngle/45) <= 9){
                int bin = neighborAngle/45;
                double proportion = (neighborAngle-bin*45)/45.0 - 0.5;
                tmp[bin] += (1-proportion) * neighborMag * smallFilter[gi][gj];
                if(proportion > 0){
                    if(bin < tmp.size()-1)
                        tmp[bin+1] += proportion * neighborMag * smallFilter[gi][gj];
                    else
                        tmp[0] += proportion * neighborMag * smallFilter[gi][gj];
                }
                else{
                    if(bin>0)
                        tmp[bin-1] += proportion * neighborMag * smallFilter[gi][gj];
                    else
                        tmp[tmp.size()-1] += proportion * neighborMag * smallFilter[gi][gj];
                }
            }
        }
    }

    // Normalize to unit length
    double sum = 0.0;
    double mean = 0.0;
    double variance = 0.0;
    double sd = 0.0;
    for(const auto& e:tmp)
        sum += e;
    mean = sum/double(tmp.size());
    for(const auto& e:tmp)
        variance += pow(e-mean, 2) / double(tmp.size());
    sd = sqrt(variance);
    for(auto &e:tmp)
      e /= double(sd);

    // // Check if any element > 0.2, if yes, then normalize again
    // bool check = false;
    // unsigned maxIndex = 0;
    // for(int i=0; i<tmp.size(); i++)
    // {
    //     if(tmp[i] > 0.2)
    //         check = true;
    //     if(tmp[i] >= maxIndex);
    //         maxIndex = i;
    // }
    // if(check)
    // {
    //     double divide = tmp[maxIndex] / 0.2;
    //     for(auto& e:tmp)
    //     {
    //         e /= divide;
    //     }
    // }
    return tmp;
}

vector<extrema> SIFT_DESCRIPTOR(Mat& colorfulImage, Mat& image)
{
    double sigma = 1.6;
    unsigned octave = 4;
    unsigned octaveLayer = 5;
    vector<vector<Mat> > scaleSpace;
    vector<vector<Mat> > dogScaleSpace;
    cvtColor(colorfulImage, image, COLOR_BGR2GRAY);
    // Shrink the image size
    while(image.rows > 600){
        resize(image, image, Size(image.cols/2,image.rows/2)); 
        resize(colorfulImage, colorfulImage, Size(colorfulImage.cols/2,colorfulImage.rows/2));
    }

    // if(! image.data )                              // Check for invalid input
    // {
    //     cout <<  "Could not open or find the image" << endl ;
    //     return -1;
    // }

    for(int i=0; i<octave; i++){
        vector<Mat > tmpScale;
        for(int j=0; j<octaveLayer; j++){
            Mat copyImage = image.clone();
            if(i > 0)
                resize(copyImage, copyImage, Size(copyImage.cols/pow(2,(i+1)), copyImage.rows/pow(2,(i+1))));
            GaussianBlur(copyImage, copyImage, Size(), pow(2,(j-1)/2.0 + i));
            tmpScale.push_back(copyImage);
        }
        scaleSpace.push_back(tmpScale);
    }

    // Create DOG scale space
    for(int i=0; i<octave; i++){
        vector<Mat> tmpDogScale;
        for(int j=0; j<octaveLayer-1; j++){
            Mat tmp = scaleSpace[i][j+1] - scaleSpace[i][j];
            // sharpen2(tmp,tmp);
            tmpDogScale.push_back(tmp);
        }
        dogScaleSpace.push_back(tmpDogScale);
    }

    // //Display DOG
    // for(auto i:dogScaleSpace)
    // {
    //     for(auto j:i)
    //     {
    //         imshow("DOG", j);
    //         waitKey(0);
    //     }
    // }

    unsigned cnt = 0;
    vector<extrema> KP;
    
    for(unsigned i=0; i<octave; i++){
        for(unsigned j=1; j<dogScaleSpace[i].size()-1; j++){
            for(unsigned a=10; a<dogScaleSpace[i][j].rows-10; a++){
                for(unsigned b=10; b<dogScaleSpace[i][j].cols-10; b++){
                    cnt ++;
                    double val = (double)dogScaleSpace[i][j].at<uchar>(a,b);
                    if( a>0 && b>0 && a<dogScaleSpace[i][j].rows-1 && b<dogScaleSpace[i][j].cols-1){
                        // Larger than other 26 pixels
                       if( findExtrema(dogScaleSpace, val, i, j, a, b, 0)){
                            extrema tmpKp;
                            double tangentX, tangentY, angle;
                            tmpKp.octave = i;
                            
                            const float img_scale = 1.f/(255);
                            const float deriv_scale = img_scale*0.5f;
                            const float second_deriv_scale = img_scale;
                            const float cross_deriv_scale = img_scale*0.25f;

                            const Mat& img = dogScaleSpace[i][j];
                            const Mat& prev = dogScaleSpace[i][j-1];
                            const Mat& next = dogScaleSpace[i][j+1];
                            float xi, xr, xc, contr;

                            Vec3f dD((img.at<uchar>(b+1,a) - img.at<uchar>(b-1,a))*deriv_scale,
                                     (img.at<uchar>(b,a+1) - img.at<uchar>(b,a-1))*deriv_scale,
                                     (next.at<uchar>(b,a) - prev.at<uchar>(b,a))*deriv_scale);

                            int nextb = b / (double)img.cols * next.cols;
                            int prevb = b / (double)img.cols * prev.cols;
                            int nexta = a / (double)img.cols * next.cols;
                            int preva = a / (double)img.cols * prev.cols;

                            float v2 = (double)img.at<uchar>(b,a) * 2;
                            float dxx = ((double)img.at<uchar>(b+1,a) + (double)img.at<uchar>(b-1,a) - v2)*second_deriv_scale;
                            float dyy = ((double)img.at<uchar>(b,a+1) + (double)img.at<uchar>(b,a-1) - v2)*second_deriv_scale;
                            float dss = ((double)next.at<uchar>(nextb, nexta) + (double)prev.at<uchar>(prevb, preva) - v2)*second_deriv_scale;
                            float dxy = ((double)img.at<uchar>(b+1,a+1) - (double)img.at<uchar>(b-1,a) -
                                         (double)img.at<uchar>(b+1,a-1) + (double)img.at<uchar>(b-1,a-1))*cross_deriv_scale;
                            float dxs = ((double)next.at<uchar>(nextb+1, nexta) - (double)next.at<uchar>(nextb-1, nexta) -
                                         (double)prev.at<uchar>(prevb+1, preva) + (double)prev.at<uchar>(prevb-1, preva))*cross_deriv_scale;
                            float dys = ((double)next.at<uchar>(nextb, nexta+1) - (double)next.at<uchar>(nextb, nexta-1) -
                                         (double)prev.at<uchar>(prevb, preva+1) + (double)prev.at<uchar>(prevb, preva-1))*cross_deriv_scale;

                            Matx33f H(dxx, dxy, dxs,
                                      dxy, dyy, dys,
                                      dxs, dys, dss);
                            Vec3f X = H.solve(dD, DECOMP_LU);
                            xi = -X[2];
                            xr = -X[1];
                            xc = -X[0];

                            if( abs(xi) < 0.5f && abs(xr) < 0.5f && abs(xc) < 0.5f )
                                break;
                            cout << xi << "  "<< xr << "  "<< xc <<endl;

                            Matx31f dD2((img.at<uchar>(b+1, a) - img.at<uchar>(b-1, a))*deriv_scale,
                                        (img.at<uchar>(b, a+1) - img.at<uchar>(b, a-1))*deriv_scale,
                                        (next.at<uchar>(nextb, nexta) - prev.at<uchar>(prevb, preva))*deriv_scale);
                            float t = dD2.dot(Matx31f(xc, xr, xi));

                            contr = img.at<uchar>(b,a)*img_scale + t*0.5f;
                            if( abs( contr ) * octaveLayer < 0.03 )
                                break;

                            // Eliminating edge responses
                            double edgeThreshold = 10.0;
                            float tr = dxx + dyy;
                            float det = dxx * dyy - dxy * dxy;
                            if( det <= 0 || tr*tr*edgeThreshold >= (edgeThreshold + 1)*(edgeThreshold + 1)*det )
                                break;
                            
                            tmpKp.pt.x = b / (double)img.cols * (double)image.cols + cvRound(xc);
                            tmpKp.pt.y = a / (double)img.rows * (double)image.rows + cvRound(xr);
                            tmpKp.OrientationBin.resize(8);
                            
                            vector<vector<double> > gaussianFilter;
                            createFilter(gaussianFilter);

                            for(int Y_cell=0; Y_cell<4; Y_cell++){
                                for(int X_cell=0; X_cell<4; X_cell++){
                                    vector<double> tmpCell;
                                    vector<vector<double> > smallFilter;
                                    for(int si=0; si<4; si++){
                                        vector<double> tmpfilter;
                                        for(int sj=0; sj<4; sj++)
                                            tmpfilter.push_back(gaussianFilter[X_cell*4+si+1][Y_cell*4+sj+1]);
                                        smallFilter.push_back(tmpfilter);
                                    }
                                    tmpCell = calcOrientationHist(image, Point(tmpKp.pt.x - X_cell*4 - 7, tmpKp.pt.y - Y_cell*4 - 7), 
                                                            Point(tmpKp.pt.x - X_cell*4 - 3, tmpKp.pt.y - Y_cell*4 - 3), tmpKp.pt, smallFilter);
                                    tmpKp.NeighborCells.push_back(tmpCell);
                                }
                            }

                            rotateDescriptor(tmpKp.NeighborCells);
                            tangentX = (double)image.at<uchar>(tmpKp.pt.x+1, tmpKp.pt.y) - (double)image.at<uchar>(tmpKp.pt.x-1, tmpKp.pt.y);
                            tangentY = (double)image.at<uchar>(tmpKp.pt.x, tmpKp.pt.y+1) - (double)image.at<uchar>(tmpKp.pt.x, tmpKp.pt.y-1);
                            angle = calAngle(tangentX, tangentY);
                            tmpKp.angle = angle;
                            tmpKp.magnitude = pow(tangentX * tangentX + tangentY * tangentY, 0.5);

                            // cout << tmpKp.pt.x << ", " << tmpKp.pt.y << endl << endl; 
                            // cout << angle << " = " << tangentY << " / " << tangentX << "\t\t" << tmpKp.magnitude << "\t\t" << tan(angle/180.0*PI) << endl;
                            KP.push_back(tmpKp);
                        }
                    }
                }
            }
        }
    }

    // Draw the keypoints
    for(auto e:KP)
    {
        // cout << e.pt << " " << e.octave << " " << e.angle << endl;
        circle(colorfulImage, e.pt, 2, Scalar(0,255,255));
        if(e.angle != 66666){
            if(abs(e.magnitude) < 100 && abs(e.magnitude * tan(e.angle/180.0*PI)) < 100)
                line(colorfulImage, e.pt, Point(e.pt.x + e.magnitude, e.pt.y + e.magnitude * tan(e.angle/180.0*PI)), Scalar(0,255,255), 1);

            else if(abs(e.magnitude) >= 100 && abs(e.magnitude * tan(e.angle/180.0*PI)) < 100)
                line(colorfulImage, e.pt, Point(e.pt.x + 100, e.pt.y + 100 * tan(e.angle/180.0*PI)), Scalar(0,255,255), 1);
            else if(abs(e.magnitude) < 100 && abs(e.magnitude * tan(e.angle/180.0*PI)) >= 100)
                line(colorfulImage, e.pt, Point(e.pt.x + 100 / double(tan(e.angle/180.0*PI)), e.pt.y + 100), Scalar(0,255,255), 1);
            else{
                if(abs(e.magnitude) > abs(e.magnitude * tan(e.angle/180.0*PI)))
                    line(colorfulImage, e.pt, Point(e.pt.x + 100, e.pt.y + 100 * tan(e.angle/180.0*PI)), Scalar(0,255,255), 1);
                else
                    line(colorfulImage, e.pt, Point(e.pt.x + 100 / double(tan(e.angle/180.0*PI)), e.pt.y + 100), Scalar(0,255,255), 1);
            }
        }
    }

    // // Keypoint Feature Descriptor
    // for(auto e:KP)
    // {
    //     for(auto f:e.NeighborCells)
    //     {
    //         cout << "[ ";
    //         for(auto g:f)
    //             cout << g << "\t";
    //         cout << "]" << endl;
    //     }
    //     cout << endl;
    // }

    cout << "From " << cnt << " point, ";
    cout << "we extracted " << KP.size() << " interest points." << endl;

    namedWindow( "Display window", WINDOW_AUTOSIZE );    // Create a window for display.
    imshow( "Display window", colorfulImage);                   // Show our image inside it.
    waitKey(0);                                          // Wait for a keystroke in the window
    return KP;
}

int main( int argc, char** argv )
{
    vector<extrema> ex1, ex2;
    Mat image, colorfulImage;
    colorfulImage = imread("/Users/hsjimwang/Desktop/SIFT/plat1.jpg");      // Read the file
    if(! colorfulImage.data ){
        cout <<  "Could not open or find the image" << endl ;
        return -1;
    }
    ex1 = SIFT_DESCRIPTOR(colorfulImage, image);

    Mat image2, colorfulImage2;
    colorfulImage2 = imread("/Users/hsjimwang/Desktop/SIFT/plat1.jpg");
    if(! colorfulImage2.data ){
        cout <<  "Could not open or find the image" << endl ;
        return -1;
    }
    ex2 = SIFT_DESCRIPTOR(colorfulImage2, image);
    
    vector<Point> matchKP1, matchKP2;
    int CNT = 0;
    double min_dist = 1000;

    for(const auto& i:ex1){
        for(auto j:ex2){
            double dist = 0;
            for(int k=0; k<j.NeighborCells.size(); k++){
                for(int l=0; l<j.NeighborCells[k].size(); l++){
                    dist += pow(i.NeighborCells[k][l] - j.NeighborCells[k][l], 2);
                }
            }
            if(dist < min_dist) min_dist = dist;
        }
    }

    vector<vector<double> > everyDist; 

    for(int i=0; i<ex1.size(); i++){
        vector<double> tmpEveryDist;
        for(int j=0; j<ex2.size(); j++){
            double tmpSum = 0;
            for(int k=0; k<ex2[j].NeighborCells.size(); k++){
                for(int l=0; l<ex2[j].NeighborCells[k].size(); l++){
                    tmpSum += pow(ex1[i].NeighborCells[k][l] - ex2[j].NeighborCells[k][l], 2);
                }
            }
            tmpEveryDist.push_back(tmpSum);
            // if (tmpSum < min_dist*1.5){
            //     cout << "(" << ex1[i].pt.x << ", " << ex1[i].pt.y << ") match ("
            //         << ex2[j].pt.x << ", " << ex2[j].pt.y << ")" << endl;
            //     matchKP1.push_back(ex1[i].pt);
            //     matchKP2.push_back(ex2[j].pt);
            //     CNT++;
            // }
        }
        everyDist.push_back(tmpEveryDist);
    }

////////////////////////////////////////////////////////////////////////////////////////////////////////
    vector<double> everyMinDist, everySecondMinDist;

    vector<int> iIdx;
    for(int i=0; i<everyDist.size(); i++){
        everyMinDist.push_back(Min(everyDist[i]).first);
        iIdx.push_back(Min(everyDist[i]).second);
    }

    vector<int> jIdx;
    for(int i=0; i<everyDist.size(); i++){
        everySecondMinDist.push_back(secondMin(everyDist[i], iIdx[i]).first);
        jIdx.push_back(secondMin(everyDist[i], iIdx[i]).second);
    }

    for(int i=0; i<everyDist.size(); i++){
        if(everyMinDist[i]/double(everySecondMinDist[i]) < 0.9 || everyMinDist[i] < 1 ){
            cout << "(" << ex1[i].pt.x << ", " << ex1[i].pt.y << ") match ("
                << ex2[iIdx[i]].pt.x << ", " << ex2[iIdx[i]].pt.y << ")" << endl;
            matchKP1.push_back(ex1[i].pt);
            matchKP2.push_back(ex2[iIdx[i]].pt);
            CNT++;
        }
    }
////////////////////////////////////////////////////////////////////////////////////////////////////////
    // for(int i=0; i<everyMinDist.size(); i++)
    // //     cout << everyMinDist[i] << "    " << everySecondMinDist[i] << endl;
    // cout << iIdx[i] << "    " << everyMinDist[i] << "    "  << jIdx[i] << "    " << everySecondMinDist[i] << endl;

    cout << "min_dist = " << min_dist << endl; 
    cout << "We matched " << CNT << " pairs." << endl;

    for(auto e:matchKP1)
        circle(colorfulImage, e, 10, Scalar(0,0,255));
    for(auto e:matchKP2)
        circle(colorfulImage2, e, 10, Scalar(0,0,255));

    Mat matches(colorfulImage.rows*1.2,colorfulImage.cols*2,CV_8UC3);

    for(int y=0; y<colorfulImage.rows-105; y++){
        for(int x=0; x<colorfulImage.cols+100; x++){
            matches.at<Vec3b>(x,y) = colorfulImage.at<Vec3b>(x,y);
        }
    }

    for(int y=0; y<colorfulImage.rows-105; y++){
        for(int x=0; x<colorfulImage.cols+100; x++){
            matches.at<Vec3b>(x,y+colorfulImage.rows-105) = colorfulImage2.at<Vec3b>(x,y);
        }
    }

    for(int i=0; i<matchKP1.size(); i++){
        line(matches,matchKP1[i],Point(matchKP2[i].x+colorfulImage.rows-105, matchKP2[i].y),Scalar(0,0,255));
    }
    imshow("ShowMatches", matches);
    waitKey(0);
    return 0;
}
