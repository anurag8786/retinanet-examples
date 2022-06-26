### Device Setup

Flash Avermedia Jetson NX with Jetpack Version: 4.5.1 by downloading the BSP NX image from the following link https://www.avermedia.com/epaper/file_https.php?file=http://ftp2.avermedia.com/EN715/EN715-NX-R1.0.7.4.5.1.zip

To confirm the Jetpack version execute the following command

```bash
$ sudo apt-cache show nvidia-jetpack
#Output
Package: nvidia-jetpack
Version: 4.5.1-b17
Architecture: arm64
...
```

### Installations

After the first boot, remove unnecessary packages from software window such as: Games, LibreOffice & Thunderbird Mail, to free up the space from the device. Then perform the following installations

#### Installing Nvidia Packages

```bash
$ sudo apt-get update
$ sudo apt install nvidia-cuda
$ sudo apt install nvidia-tenssorrt
$ sudo apt install nvidia-opencv
$ sudo apt install cmake
```

#### Installing Deepstream 5.0

Install the following dependencies before installing deepstream

```bash
$ sudo apt-get install \
    libssl1.0.0 \
    libgstreamer1.0-0 \
    gstreamer1.0-tools \
    gstreamer1.0-plugins-good \
    gstreamer1.0-plugins-bad \
    gstreamer1.0-plugins-ugly \
    gstreamer1.0-libav \
    gstreamer1.0-alsa \
    libgstrtspserver-1.0-0 \
    libjansson4
```

```bash
$ sudo apt-get install libgstreamer-plugins-base1.0-dev libgstreamer1.0-dev \
   libgstrtspserver-1.0-dev libx11-dev libjson-glib-dev
```

Install Deepstream

```bash
$ sudo apt install deepstream-5.0
```

### Setting up retinanet-examples repository

Clone the retinanet-examples repository in home directory

```bash
$ git clone https://github.com/anurag8786/retinanet-examples.git
```

```bash
$ cd retinanet-examples
$ git checkout retina-adverto
$ cd extras/cppapi/
```

#### Cpp API

The C++ API allows you to build a TensorRT engine for inference using the ONNX export of a core model.

The following shows how to build and run code samples for exporting an ONNX core model (from RetinaNet or other toolkit supporting the same sort of core model structure) to a TensorRT engine and doing inference on images.

```bash
$ mkdir build && cd build
$ cmake -DCMAKE_CUDA_FLAGS="--expt-extended-lambda -std=c++14" ..
$ sudo make
```

Download the onnx model file in same folder (~/retinanet-examples/extras/cppapi/build)

```bash

```

Convert the onnx model to plan engine (optimised model) file, this will take around 10-15 mins

```bash
$ ./export final.onnx final.plan
```

#### Deepstream API

```bash
$ cd /home/nvidia/retinanet-examples/extras/deepstream/deepstream-sample
$ mkdir build && cd build
$ cmake -DDeepStream_DIR=/opt/nvidia/deepstream/deepstream-5.0 .. && make -j
```

### Setting up deepstream

To save the frames of inferred images through deepstream, following changes need to be made

```bash
$ sudo cp /home/nvidia/retinanet-examples/deepstream-updates/deepstream_dsexample.h /opt/nvidia/deepstream/deepstream-5.0/sources/apps/apps-common/includes/

$ sudo cp /home/nvidia/retinanet-examples/deepstream-updates/deepstream_config_file_parser.c /opt/nvidia/deepstream/deepstream-5.0/sources/apps/apps-common/src/

$ sudo cp /home/nvidia/retinanet-examples/deepstream-updates/deepstream_dsexample.c /opt/nvidia/deepstream/deepstream-5.0/sources/apps/apps-common/src/

$ sudo cp /home/nvidia/retinanet-examples/deepstream-updates/deepstream_app.c /opt/nvidia/deepstream/deepstream-5.0/sources/apps/sample_apps/deepstream-app/

$ sudo cp /home/nvidia/retinanet-examples/deepstream-updates/gstdsexample.cpp /opt/nvidia/deepstream/deepstream-5.0/sources/gst-plugins/gst-dsexample/

$ sudo cp /home/nvidia/retinanet-examples/deepstream-updates/Makefile /opt/nvidia/deepstream/deepstream-5.0/sources/gst-plugins/gst-dsexample/
```

```bash
$ cd /opt/nvidia/deepstream/deepstream-5.0/sources/gst-plugins/gst-dsexample
$ sudo make
$ sudo make install
```

```bash
$ cd /opt/nvidia/deepstream/deepstream-5.0/sources/gst-plugins/gst-dsexample/dsexample_lib
$ sudo make
```

```bash
$ cd /opt/nvidia/deepstream/deepstream-5.0/sources/apps/sample_apps/deepstream-app
$ sudo make
$ sudo make install
```

### Inference

```bash
LD_PRELOAD=/home/nvidia/retinanet-examples/extras/deepstream/deepstream-sample/build/libnvdsparsebbox_retinanet.so /opt/nvidia/deepstream/deepstream-5.0/sources/apps/sample_apps/deepstream-app/deepstream-app -c /home/nvidia/retinanet-examples/extras/deepstream/deepstream-sample/ds_config_1vid.txt
```





