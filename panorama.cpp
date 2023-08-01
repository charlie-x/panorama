#include <iostream>
#include <math.h>
#include <algorithm>
#include <string>
#include <cmath>

#include "opencv2/opencv.hpp"
using namespace cv;

// getopt license doesn't permit static link
#include "getopt.h"

#ifndef _WIN32
#include <unistd.h>
#endif

//#include <xmmintrin.h>
//#include <pmmintrin.h>

// Multithreading
#include <tbb/parallel_for.h>
#include <tbb/blocked_range.h>
using namespace tbb;

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#ifndef M_PI_2
#define M_PI_2 1.57079632679489661923
#endif

// Input parameters
int iflag, oflag, hflag, rflag;
char *ivalue, *ovalue;
int rvalue=4096;

/**
 **	Parse input parameters
 **/
int parseParameters(int argc, char *argv[]) {
    iflag = oflag = hflag = rflag = 0;
    ivalue = ovalue = NULL;
    int c;
    opterr = 0;
    
    while ((c = getopt (argc, argv, "i:o:r:")) != -1)
        switch (c) {
            case 'i':
                // input file
                iflag = 1;
                ivalue = optarg;
                break;
            case 'o':
                oflag = 1;
                ovalue = optarg;
                break;
            case 'r':
                rflag = 1;
                rvalue = std::stoi(optarg);
                break;
            case '?':
                if (optopt == 'i' || optopt == 'o' || optopt == 'r')
                    fprintf (stderr, "Option -%c requires an argument.\n", optopt);
                else if (isprint (optopt))
                    fprintf (stderr, "Unknown option `-%c'.\n", optopt);
                else
                    fprintf (stderr, "Unknown option character `\\x%x'.\n", optopt);
                return 1;
            default:
                abort ();
        }
    
    if (iflag==0 || oflag == 0) {
        std::cout << "No inputs or outputs specified: "<< iflag << "/" << oflag <<"\n";
        abort ();
        return 1;
    }
    return 0;
}

template <typename Coordinate>
struct Vec3_ {
    Coordinate x, y, z;
    
    Vec3_(Coordinate x = {}, Coordinate y = {}, Coordinate z = {})
        : x{x}, y{y}, z{z}
    {}
    
    template <typename Other>
    Vec3_(const Vec3_<Other> &other)
        : x{static_cast<Coordinate>(other.x)}
        , y{static_cast<Coordinate>(other.y)}
        , z{static_cast<Coordinate>(other.z)}
    {}
    
    inline Vec3_ operator +(const Vec3_ &other) const {
        return {x + other.x, y + other.y, z + other.z};
    }
    inline Vec3_ operator -(const Vec3_ &other) const {
        return {x - other.x, y - other.y, z - other.z};
    }
    inline Vec3_ operator *(Coordinate c) const {
        return {x * c, y * c, z * c};
    }
};
using Vec3fa = Vec3_<double>; // though not f(float) and not a(aligned)
using Vec3uc = Vec3_<unsigned char>;
struct PixelRange {int start, end; };

/** get x,y,z coords from out image pixels coords
 **	i,j are pixel coords
 **	face is face number
 **	edge is edge length
 **/
Vec3f outImgToXYZ(int, int, int, int);
Vec3b interpolateXYZtoColor(Vec3f xyz, const Mat& imgIn);

void convertBack(Mat& imgIn, std::vector<Mat>& imgOut);

/**
 **	Convert panorama using an inverse pixel transformation
 **/

int main (int argc, char *argv[]) {
    std::cout << "PeakVisor panorama translator...\n";
    
    parseParameters(argc, argv);
    
    std::cout << "  convert equirectangular panorama: [" << ivalue << "] into cube faces: ["<< ovalue << "] of " << rvalue <<" pixels in dimension\n";
    
	// Input image
	auto imgIn = imread(ivalue);
	if (imgIn.empty() ) {
		std::cerr << "Error: Could not open the input image: " << ivalue << std::endl;
		return 1;
	}
    
    // Create output images
	std::vector < Mat> imgOut;
	imgOut.resize(6);

	for (auto &u : imgOut) {
		u = cv::Mat(imgIn.cols / 6, imgIn.cols / 6, CV_8UC3);

		u.setTo(cv::Scalar(255, 0, 0)); // Blue color (BGR order)

	}

    // Convert panorama
    convertBack(imgIn,imgOut);
    
    // Write output images
    for (int i=0; i<6; ++i){
        std::string fname = std::string(ovalue) + "_" + std::to_string(i) + ".jpg";
		imwrite(fname.c_str(), imgOut[i]);
    }
    
    std::cout << "  conversion finished successfully\n";

    return 0;
}

/**
 **	Convert panorama using an inverse pixel transformation
 **/

void convertBack(Mat& imgIn, std::vector<Mat>& imgOut) {
	int rvalue = imgIn.cols / 6; // Assuming the input cube map is divided equally into six faces
	int edge = rvalue; // the length of each edge in pixels

	// Look around cube faces
	for (int face = 0; face < 6; ++face) {
		for (int i = 0; i < edge; ++i) {
			for (int j = 0; j < edge; ++j) {
				Vec3f xyz = outImgToXYZ(i, j, face, edge); // Assuming outImgToXYZ returns a Vec3f
				Vec3b clr = interpolateXYZtoColor(xyz, imgIn); // Assuming interpolateXYZtoColor returns a Vec3b
				imgOut[face].at<Vec3b>(j, i) = clr;
			}
		}
	}
}

// Given i,j pixel coordinates on a given face in range (0,edge), 
// find the corresponding x,y,z coords in range (-1.0,1.0)
Vec3f outImgToXYZ(int i, int j, int face, int edge) {

    float a = (2.0f*i)/edge - 1.0f;
    float b = (2.0f*j)/edge - 1.0f;
    Vec3f res;
    if (face==0) { // back
        res = {-1.0f, -a, -b};
    } else if (face==1) { // left
        res = {a, -1.0f, -b};
    } else if (face==2) { // front
        res = {1.0f, a,  -b};
    } else if (face==3) { // right
        res = {-a, 1.0f, -b};
    } else if (face==4) { // top
        res = {b, a, 1.0f};
    } else if (face==5) { // bottom
        res = {-b, a, -1.0f};
    }
    else {
        printf("face %d\n",face);    
    }
    return res;
}

template <typename T>
static inline T clamp(const T &n, const T &lower, const T &upper) {
    return std::min(std::max(n, lower), upper);
}

template <typename T>
static inline T safeIndex(const T n, const T size) {
    return clamp(n, {}, size - 1);
}

template <typename T, typename Scalar>
static inline T mix(const T &one, const T &other, const Scalar &c) {
    return one + (other - one) * c;
}

cv::Vec3b interpolateXYZtoColor(cv::Vec3f xyz, const cv::Mat& imgIn)
{
	int _sw = imgIn.cols, _sh = imgIn.rows;

	auto theta = std::atan2(xyz[1], xyz[0]), r = std::hypot(xyz[0], xyz[1]);
	auto phi = std::atan2(xyz[2], r);

	auto uf = (theta + CV_PI) / CV_PI * _sh;
	auto vf = (CV_PI / 2 - phi) / CV_PI * _sh;

	int ui = std::clamp(static_cast<int>(std::floor(uf)), 0, _sw - 1);
	int vi = std::clamp(static_cast<int>(std::floor(vf)), 0, _sh - 1);
	int u2 = std::clamp(ui + 1, 0, _sw - 1);
	int v2 = std::clamp(vi + 1, 0, _sh - 1);

	double mu = uf - ui, nu = vf - vi;
	mu = nu = 0; // Note: This line resets mu and nu to 0, effectively disabling interpolation.

	auto A = imgIn.at<cv::Vec3b>(vi, ui);
	auto B = imgIn.at<cv::Vec3b>(vi, u2);
	auto C = imgIn.at<cv::Vec3b>(v2, ui);
	auto D = imgIn.at<cv::Vec3b>(v2, u2);

	cv::Vec3b value;
	for (int i = 0; i < 3; i++) {
		value[i] = (1 - nu) * ((1 - mu) * A[i] + mu * B[i]) + nu * ((1 - mu) * C[i] + mu * D[i]);
	}
	return value;
}
