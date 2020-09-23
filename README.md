
# FaceRecognitionCPU


## Introduction:

FaceRecognitionCPU uses Fast-MTCNN, Arcface, RetinaFace, and the TVM stack. The facial recognition camera
pipline can be ran on CPU (2.8 GHZ is enough for real time) for near real-time facial recognition.
In addition, FaceAttendance includes AutoTVM capabilties for further model optimization and tuning.




**Dependencies:**

* OpenCV4
* TVM 


**Setup:** 

* OpenCV
```
brew install opencv
brew link opencv
brew install pkg-config
pkg-config --cflags --libs /usr/local/Cellar/opencv/<version_number>/lib/pkgconfig/opencv.pc

```
If the pkg-config cannot find the opencv.pc file, if you are using OpenCV4, change the filename in the command to opencv4.pc


* TVM

```
git clone --recursive https://github.com/dmlc/tvm

mkdir build
cp cmake/config.cmake build
cd build
cmake ..
make -j4

```


* Compiler

Use the compiler to compile your pretrained MXNet model. The default model is mobilefacenet-arcface. 


* CMakeLists.txt

Change the TVM path in the file to your own.


* Prefix

Set the prefix model path to your own.


* Ground Truth Recording

create a new directory by the name "img" and set record to "1" to record ground truth image for face recognition




## Run:

Running the project will activate your camera.

```
mkdir build
cd build
cmake ..
make -j4
./FaceRecognitionCpp
```




## TODO:

1. Test production deployment through Flask 
2. Optimize further for large scale facial recognition
3. Set up frontend web server to connect facial recognition system with a SQL database?
4. Set up cross compiling and RPC tracker (maybe look into android device deployment)






## Citation:

```
@inproceedings{imistyrain2018MTCNN,
title={Fast-MTCNN https://github.com/imistyrain/MTCNN/tree/master/Fast-MTCNN},
author={Jack Yu},
}

@inproceedings{RetinaFace-TVM,
title={RetinaFace-TVM https://github.com/Howave/RetinaFace-TVM},
author={Howave},
}

@inproceedings{deng2019retinaface,
title={RetinaFace: Single-stage Dense Face Localisation in the Wild},
author={Deng, Jiankang and Guo, Jia and Yuxiang, Zhou and Jinke Yu and Irene Kotsia and Zafeiriou, Stefanos},
booktitle={arxiv},
year={2019}
}

@inproceedings{guo2018stacked,
  title={Stacked Dense U-Nets with Dual Transformers for Robust Face Alignment},
  author={Guo, Jia and Deng, Jiankang and Xue, Niannan and Zafeiriou, Stefanos},
  booktitle={BMVC},
  year={2018}
}

@article{deng2018menpo,
  title={The Menpo benchmark for multi-pose 2D and 3D facial landmark localisation and tracking},
  author={Deng, Jiankang and Roussos, Anastasios and Chrysos, Grigorios and Ververas, Evangelos and Kotsia, Irene and Shen, Jie and Zafeiriou, Stefanos},
  journal={IJCV},
  year={2018}
}

@inproceedings{deng2018arcface,
title={ArcFace: Additive Angular Margin Loss for Deep Face Recognition},
author={Deng, Jiankang and Guo, Jia and Niannan, Xue and Zafeiriou, Stefanos},
booktitle={CVPR},
year={2019}
}

```
