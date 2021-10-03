#include <iostream>
#include <stdexcept>
#include <fstream>
#include <vector>
#include <chrono>
#include <string>

#include <opencv2/opencv.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

#include <cuda_runtime.h>

#include "../../csrc/engine.h"

using namespace std;
using namespace cv;

int main(int argc, char *argv[]) {
	if (argc != 3) {
		cerr << "Usage: " << argv[0] << " engine.plan input.mov" << endl;
		return 1;
	}

	cout << "Loading engine..." << endl;
	auto engine = retinanet::Engine(argv[1]);
	VideoCapture src(argv[2]);
	if (!src.isOpened()){
		cerr << "Could not read " << argv[2] << endl;
		return 1;
	}
	auto fh=src.get(CAP_PROP_FRAME_HEIGHT);
	auto fw=src.get(CAP_PROP_FRAME_WIDTH);
	auto fps=src.get(CAP_PROP_FPS);
	auto nframes=src.get(CAP_PROP_FRAME_COUNT);

	Mat frame;
	Mat resized_frame;
	Mat inferred_frame;
	int count=1;

	auto inputSize = engine.getInputSize();
	// Create device buffers
	void *data_d, *scores_d, *boxes_d, *classes_d;
	auto num_det = engine.getMaxDetections();
	cudaMalloc(&data_d, 3 * inputSize[0] * inputSize[1] * sizeof(float));
	cudaMalloc(&scores_d, num_det * sizeof(float));
	cudaMalloc(&boxes_d, num_det * 4 * sizeof(float));
	cudaMalloc(&classes_d, num_det * sizeof(float));

	auto scores = new float[num_det];
	auto boxes = new float[num_det * 4];
	auto classes = new float[num_det];

	vector<float> mean {0.485, 0.456, 0.406};
	vector<float> std {0.229, 0.224, 0.225};

	vector<uint8_t> blues {0,63,127,191,255,0}; //colors for bonuding boxes
	vector<uint8_t> greens {0,255,191,127,63,0};
	vector<uint8_t> reds {191,255,0,0,63,127};

	int channels = 3;
	vector<float> img;
	vector<float> data (channels * inputSize[0] * inputSize[1]);
	


	double threshold;
	cout << "Threshold [0-1] : ";
	cin >> threshold;
	
	ofstream savefile;

	string name = "frame_dir/Frame";
	string ext = ".txt";
	string jpg = ".jpg";

	while (1){
		src >> frame;
		if (frame.empty()){
			cout << "Finished inference!" << endl;
			break;
		}

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
		engine.infer(buffers, 1);

		cudaMemcpy(scores, scores_d, sizeof(float) * num_det, cudaMemcpyDeviceToHost);
		cudaMemcpy(boxes, boxes_d, sizeof(float) * num_det * 4, cudaMemcpyDeviceToHost);
		cudaMemcpy(classes, classes_d, sizeof(float) * num_det, cudaMemcpyDeviceToHost);
		
		string strcount = to_string(count);
		savefile.open(name+strcount+ext, ios_base::app);
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
				savefile << cls <<" "<< x1 <<" "<< y1 <<" "<< x2 <<" "<< y2 << endl;
			}
		}
		cv::resize(resized_frame, inferred_frame, Size(fw, fh));
		imwrite(name+strcount+jpg, resized_frame);
		savefile.close();
		
	}
	src.release();
	return 0;
}
