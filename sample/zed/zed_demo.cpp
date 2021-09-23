/*
Copyright 2016 Fixstars Corporation

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http ://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

#include <stdlib.h>
#include <cstdint>
#include <iostream>
#include <iomanip>
#include <string>
#include <chrono>

#include <cuda_runtime.h>
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <sl/Camera.hpp>

#include <libsgm.h>

template <class... Args>
static std::string format_string(const char* fmt, Args... args)
{
	const int BUF_SIZE = 1024;
	char buf[BUF_SIZE];
	std::snprintf(buf, BUF_SIZE, fmt, args...);
	return std::string(buf);
}

struct device_buffer {
	device_buffer() : data(nullptr) {}
	device_buffer(size_t count) { allocate(count); }
	void allocate(size_t count) { cudaMalloc(&data, count); }
	~device_buffer() { cudaFree(data); }
	void* data;
};

int getOCVtype(sl::MAT_TYPE type) {
    int cv_type = -1;
    switch (type) {
        case sl::MAT_TYPE::F32_C1: cv_type = CV_32FC1; break;
        case sl::MAT_TYPE::F32_C2: cv_type = CV_32FC2; break;
        case sl::MAT_TYPE::F32_C3: cv_type = CV_32FC3; break;
        case sl::MAT_TYPE::F32_C4: cv_type = CV_32FC4; break;
        case sl::MAT_TYPE::U8_C1: cv_type = CV_8UC1; break;
        case sl::MAT_TYPE::U8_C2: cv_type = CV_8UC2; break;
        case sl::MAT_TYPE::U8_C3: cv_type = CV_8UC3; break;
        case sl::MAT_TYPE::U8_C4: cv_type = CV_8UC4; break;
        default: break;
    }
    return cv_type;
}

cv::Mat slMat2cvMat(sl::Mat& input) {
    // Since cv::Mat data requires a uchar* pointer, we get the uchar1 pointer from sl::Mat (getPtr<T>())
    // cv::Mat and sl::Mat will share a single memory structure
    return cv::Mat(input.getHeight(), input.getWidth(), getOCVtype(input.getDataType()), 
                   input.getPtr<sl::uchar1>(sl::MEM::CPU), input.getStepBytes(sl::MEM::CPU));
}

void parse_args(int argc, char **argv,sl::InitParameters& param)
{
    if (argc > 1 && std::string(argv[1]).find(".svo")!=std::string::npos) {
        // SVO input mode
        param.input.setFromSVOFile(argv[1]);
        param.svo_real_time_mode=true;

        std::cout<<"[Sample] Using SVO File input: "<<argv[1]<<std::endl;
    } else if (argc > 1 && std::string(argv[1]).find(".svo")==std::string::npos) {
        std::string arg = std::string(argv[1]);
        unsigned int a,b,c,d,port;
        if (sscanf(arg.c_str(),"%u.%u.%u.%u:%d", &a, &b, &c, &d,&port) == 5) {
            // Stream input mode - IP + port
            std::string ip_adress = std::to_string(a)+"."+std::to_string(b)+"."+std::to_string(c)+"."+std::to_string(d);
            param.input.setFromStream(sl::String(ip_adress.c_str()),port);
            std::cout<<"[Sample] Using Stream input, IP : "<<ip_adress<<", port : "<<port<< std::endl;
        }
        else  if (sscanf(arg.c_str(),"%u.%u.%u.%u", &a, &b, &c, &d) == 4) {
            // Stream input mode - IP only
            param.input.setFromStream(sl::String(argv[1]));
            std::cout<<"[Sample] Using Stream input, IP : "<<argv[1]<<std::endl;
        }
        else if (arg.find("HD2K")!=std::string::npos) {
            param.camera_resolution = sl::RESOLUTION::HD2K;
            std::cout<<"[Sample] Using Camera in resolution HD2K"<<std::endl;
        } else if (arg.find("HD1080")!=std::string::npos) {
            param.camera_resolution = sl::RESOLUTION::HD1080;
            std::cout<<"[Sample] Using Camera in resolution HD1080"<<std::endl;
        } else if (arg.find("HD720")!=std::string::npos) {
            param.camera_resolution = sl::RESOLUTION::HD720;
            std::cout<<"[Sample] Using Camera in resolution HD720"<<std::endl;
        } else if (arg.find("VGA")!=std::string::npos) {
            param.camera_resolution = sl::RESOLUTION::VGA;
            std::cout<<"[Sample] Using Camera in resolution VGA"<<std::endl;
        }
    } else {
        // Default
    }
}

int main(int argc, char* argv[]) {	
	
	const int disp_size = 128;
	
	sl::Camera zed;
	sl::InitParameters initParameters;
	initParameters.camera_resolution = sl::RESOLUTION::VGA;

    parse_args(argc, argv, initParameters);

	auto err = zed.open(initParameters);

	if (err != sl::ERROR_CODE::SUCCESS) {
		std::cout << toString(err) << std::endl;
		zed.close();
		return 1;
	}

    sl::CameraInformation camera_infos = zed.getCameraInformation();
    auto resolution = camera_infos.camera_configuration.resolution;

	const int width = resolution.width;
	const int height = resolution.height;

	sl::Mat d_zed_image_l(resolution, sl::MAT_TYPE::U8_C1, sl::MEM::GPU);
	sl::Mat d_zed_image_r(resolution, sl::MAT_TYPE::U8_C1, sl::MEM::GPU);
	sl::Mat zed_image_l(resolution, sl::MAT_TYPE::U8_C1, sl::MEM::CPU);

	const int input_depth = 8;
	const int output_depth = 8;
	const int output_bytes = output_depth * width * height / 8;

	CV_Assert(d_zed_image_l.getStep(sl::MEM::GPU) == d_zed_image_r.getStep(sl::MEM::GPU));
	sgm::StereoSGM sgm(width, height, disp_size, input_depth, output_depth, static_cast<int>(d_zed_image_l.getStep(sl::MEM::GPU)), width, sgm::EXECUTE_INOUT_CUDA2CUDA);

	cv::Mat disparity(height, width, CV_8U);
	cv::Mat disparity_8u, disparity_color;

	device_buffer d_disparity(output_bytes);
	while (1) {
		if (zed.grab() == sl::ERROR_CODE::SUCCESS) {
			zed.retrieveImage(d_zed_image_l, sl::VIEW::LEFT_GRAY, sl::MEM::GPU);
			zed.retrieveImage(d_zed_image_r, sl::VIEW::RIGHT_GRAY, sl::MEM::GPU);

			zed.retrieveImage(zed_image_l, sl::VIEW::LEFT_GRAY, sl::MEM::CPU);
		} else continue;

		const auto t1 = std::chrono::system_clock::now();

		sgm.execute(d_zed_image_l.getPtr<uchar>(sl::MEM::GPU), d_zed_image_r.getPtr<uchar>(sl::MEM::GPU), d_disparity.data);
		cudaDeviceSynchronize();

		const auto t2 = std::chrono::system_clock::now();
		const auto duration = std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count();
		const double fps = 1e6 / duration;

        auto image_ocv = slMat2cvMat(zed_image_l);

		cudaMemcpy(disparity.data, d_disparity.data, output_bytes, cudaMemcpyDeviceToHost);

		disparity.convertTo(disparity_8u, CV_8U, 255. / disp_size);
		cv::applyColorMap(disparity_8u, disparity_color, cv::COLORMAP_JET);
		disparity_color.setTo(cv::Scalar(0, 0, 0), disparity == static_cast<uint8_t>(sgm.get_invalid_disparity()));
		cv::putText(disparity_color, format_string("sgm execution time: %4.1f[msec] %4.1f[FPS]", 1e-3 * duration, fps),
			cv::Point(50, 50), 2, 0.75, cv::Scalar(255, 255, 255));

		cv::imshow("left", image_ocv);
		cv::imshow("disparity", disparity_color);
		const char c = cv::waitKey(1);
		if (c == 27) // ESC
			break;
	}
	zed.close();
	return 0;
}
