#  CUDA Image Processing

Project Overview This project implements CUDA-based parallel programming for image processing, focusing on convolution and morphological operations. Each operation is executed in separate CUDA kernels for optimized performance.

---

##  Features

- Efficient **convolution filters**
- Fast **morphological operations** (dilation, erosion, etc.)
- Modular design with separate CUDA kernels for each operation
- Supports input/output via OpenCV (C++)

---

##  System Requirements

###  Hardware
- **GPU**: NVIDIA GPU with CUDA Compute Capability **5.0 or higher**
- **RAM**: Minimum **8GB** (16GB recommended)
- **Storage**: At least **2GB** of free disk space

###  Software
- **OS**: Windows 10/11 (64-bit)
- **CUDA Toolkit**: Version **11.0 or later**
- **NVIDIA Drivers**: Latest compatible with your CUDA version
- **OpenCV (C++)**: Version **4.5+**
- **Visual Studio**: Version **2019/2022** (Community or Professional)

---

##  Setup Instructions

1️. Install CUDA Toolkit

- Download the CUDA Toolkit from NVIDIA's official site: [CUDA Toolkit](https://developer.nvidia.com/cuda-toolkit)
- During installation:
  - Enable Visual Studio Integration.
  - Ensure nvcc compiler is added to the system PATH.

To verify:
```
nvcc --version
```
This should print CUDA version information.

2. Install NVIDIA Drivers

 i. Ensure you have the latest NVIDIA GPU drivers from NVIDIA's driver page.
ii. Restart your computer after installation.


3. Install OpenCV for C++ Method 1: Using Prebuilt Binaries (Recommended) 1. Download OpenCV from OpenCV's official site. 2. Extract it to a location (e.g., C:\opencv). 3. Set the environment variables:
```
 	o	Include Path: C:\opencv\build\include

 	o	Library Path: C:\opencv\build\x64\vc16\lib

 	o	Binary Path: C:\opencv\build\x64\vc16\bin (add this to PATH)
```
Method 2: Using vcpkg

i. Install vcpkg (if not already installed)

```
 git clone https://github.com/microsoft/vcpkg.git
 cd vcpkg
 bootstrap-vcpkg.bat
```
ii. Install OpenCV using vcpkg:
```
vcpkg install opencv[core,imgproc,highgui]:x64-windows
```

iii. Run:
```
 vcpkg integrate install
```

Compilation and Execution

Step 1: Navigate to the Project Directory

```cd C:\Users\Reliance Digital\Documents\Parallel Programming\CUDA_project```

Step 2: Compile with nvcc Run the following command to compile the project:

```
nvcc -o image_processing.exe main.cu convolution.cu morphology.cu utils.cu ^
-I"C:\opencv\build\include" ^
-L"C:\opencv\build\x64\vc16\lib" ^
-lopencv_world480
```
(Note: Replace opencv_world480 with your actual OpenCV version.)

Step 3: Run the Executable

```
image_processing.exe input_image.jpg output_image.jpg
```

Step 4: Verify Output Check output_image.jpg to see the processed result.
