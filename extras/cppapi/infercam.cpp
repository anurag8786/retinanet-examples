#include <iostream>
#include <stdexcept>
#include <fstream>
#include <vector>
#include <chrono>

#include <opencv2/opencv.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

#include <cuda_runtime.h>

#include "../../csrc/engine.h"

using namespace std;
using namespace cv;

int main(int argc, char *argv[]) {
	if (argc != 2) {
		cerr << "Usage: " << argv[0] << " engine.plan input.mov [Optional]threshold between 0-1"  << endl;
		return 1;
	}

	cout << "Loading engine..." << endl;
	auto engine = retinanet::Engine(argv[1]);
	VideoCapture src(0);

	cout << "Enter Threshold [0-1] : "<< endl;
	double threshold;
	cin >> threshold;

	if (!src.isOpened()){
		cerr << "Could not read " << argv[2] << endl;
		return 1;
	}
	
	auto fh=src.get(CAP_PROP_FRAME_HEIGHT);
	auto fw=src.get(CAP_PROP_FRAME_WIDTH);
	auto fps=src.get(CAP_PROP_FPS);
	auto nframes=src.get(CAP_PROP_FRAME_COUNT);

	clock_t start, end;
	Mat frame;
	Mat resized_frame;
	Mat inferred_frame;
	int count=1;

	auto inputSize = engine.getInputSize();
	//inputSize[0] = inputSize[0]/2;
	//inputSize[1] = inputSize[1]/2;

	// Create device buffers
	void *data_d, *scores_d, *boxes_d, *classes_d;
	auto num_det = engine.getMaxDetections();
	cudaMalloc(&data_d, 3 * inputSize[0] * inputSize[1] * sizeof(float));
	cudaMalloc(&scores_d, num_det * sizeof(float));
	cudaMalloc(&boxes_d, num_det * 4 * sizeof(float));
	cudaMalloc(&classes_d, num_det * sizeof(float));

	unique_ptr<float[]> scores(new float[num_det]);
	unique_ptr<float[]> boxes(new float[num_det * 4]);
	unique_ptr<float[]> classes(new float[num_det]);

	vector<float> mean {0.485, 0.456, 0.406};
	vector<float> std {0.229, 0.224, 0.225};

	vector<uint8_t> blues {0,63,127,191,255,0}; //colors for bonuding boxes
	vector<uint8_t> greens {0,255,191,127,63,0};
	vector<uint8_t> reds {191,255,0,0,63,127};

	int channels = 3;
	vector<float> img;
	vector<float> data (channels * inputSize[0] * inputSize[1]);

	cout <<"Resolution" << inputSize[0] << "*" << inputSize[1] << endl;
	String window_name = "Output Stream";

	namedWindow(window_name, WINDOW_NORMAL);
	//Mat image(inputSize[0],inputSize[1], CV_8UC3, Scalar(0,0,0));

	while(1)
	{
		start = clock();
		src >> frame;
		//frame = image;
		if (frame.empty()){
			cout << "Finished inference!" << endl;
			break;
		}

		//cv::resize(frame, resized_frame, Size(512,512));
		cv::resize(frame, resized_frame, Size(inputSize[1], inputSize[0]));
		cv::Mat pixels;
		resized_frame.convertTo(pixels, CV_32FC3, 1.0 / 255, 0);

		img.assign((float*)pixels.datastart, (float*)pixels.dataend);

		for (int c = 0; c < channels; c++) {
			for (int j = 0, hw = inputSize[0] * inputSize[1]; j < hw; j++) {
				data[c * hw + j] = (img[channels * j + 2 - c] - mean[c]) / std[c];
			}
		}

		// Copy image to device
		size_t dataSize = data.size() * sizeof(float);
		cudaMemcpy(data_d, data.data(), dataSize, cudaMemcpyHostToDevice);

		//Do inference
		cout << "Inferring on frame: " << count <<"/" << nframes << endl;
		count++;
		vector<void *> buffers = { data_d, scores_d, boxes_d, classes_d };
		engine.infer(buffers);

		cudaMemcpy(scores.get(), scores_d, sizeof(float) * num_det, cudaMemcpyDeviceToHost);
		cudaMemcpy(boxes.get(), boxes_d, sizeof(float) * num_det * 4, cudaMemcpyDeviceToHost);
		cudaMemcpy(classes.get(), classes_d, sizeof(float) * num_det, cudaMemcpyDeviceToHost);

		// Get back the bounding boxes
		for (int i = 0; i < num_det; i++) {
			// Show results over confidence threshold
			if (scores[i] >= threshold) {
				float x1 = boxes[i*4+0];
				float y1 = boxes[i*4+1];
				float x2 = boxes[i*4+2];
				float y2 = boxes[i*4+3];
				int cls=classes[i];
				// Draw bounding box on image
				cv::rectangle(resized_frame, Point(x1, y1), Point(x2, y2), cv::Scalar(blues[cls], greens[cls], reds[cls]));
			}
		}
		cv::resize(resized_frame, inferred_frame, Size(fw, fh));
		imshow(window_name, inferred_frame);
		if (waitKey(10) ==27){
			cout << "Esc key is presses. Stopping the VIdeo" << endl;
			break;
		}
		end = clock();
		double t_t = double(end-start) / double(CLOCKS_PER_SEC);
		double out_fps = 1/t_t;
		cout << "FPS of Output Stream : " << fixed << out_fps <<setprecision(2) << endl;
	}
	src.release();
	cudaFree(data_d);
	cudaFree(scores_d);
	cudaFree(boxes_d);
	cudaFree(classes_d);
	return 0;
}
