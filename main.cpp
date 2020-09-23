#include "facetracking.hpp"
#include <stdio.h>
#include <iostream>
#include <MTCNN/mtcnn_opencv.hpp>

using namespace std;
using namespace cv;

int main() {
    //load model
    FR_MFN_Deploy deploy_rec(prefix);
    RetinaFaceDeploy deploy_track(prefix);
    MTCNN detector(prefix);
//    MTCNNTracking(detector, deploy_rec);
    RetinaFaceTracking(deploy_track, deploy_rec);

}