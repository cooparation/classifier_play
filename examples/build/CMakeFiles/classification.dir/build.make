# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 2.8

#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:

# Remove some rules from gmake that .SUFFIXES does not remove.
SUFFIXES =

.SUFFIXES: .hpux_make_needs_suffix_list

# Suppress display of executed commands.
$(VERBOSE).SILENT:

# A target that is always out of date.
cmake_force:
.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/bin/cmake

# The command to remove a file.
RM = /usr/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The program to use to edit the cache.
CMAKE_EDIT_COMMAND = /usr/bin/ccmake

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/liusj/dl/classifier_play/examples

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/liusj/dl/classifier_play/examples/build

# Include any dependencies generated for this target.
include CMakeFiles/classification.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/classification.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/classification.dir/flags.make

CMakeFiles/classification.dir/test/main.cpp.o: CMakeFiles/classification.dir/flags.make
CMakeFiles/classification.dir/test/main.cpp.o: ../test/main.cpp
	$(CMAKE_COMMAND) -E cmake_progress_report /home/liusj/dl/classifier_play/examples/build/CMakeFiles $(CMAKE_PROGRESS_1)
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Building CXX object CMakeFiles/classification.dir/test/main.cpp.o"
	/usr/bin/c++   $(CXX_DEFINES) $(CXX_FLAGS) -o CMakeFiles/classification.dir/test/main.cpp.o -c /home/liusj/dl/classifier_play/examples/test/main.cpp

CMakeFiles/classification.dir/test/main.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/classification.dir/test/main.cpp.i"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_FLAGS) -E /home/liusj/dl/classifier_play/examples/test/main.cpp > CMakeFiles/classification.dir/test/main.cpp.i

CMakeFiles/classification.dir/test/main.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/classification.dir/test/main.cpp.s"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_FLAGS) -S /home/liusj/dl/classifier_play/examples/test/main.cpp -o CMakeFiles/classification.dir/test/main.cpp.s

CMakeFiles/classification.dir/test/main.cpp.o.requires:
.PHONY : CMakeFiles/classification.dir/test/main.cpp.o.requires

CMakeFiles/classification.dir/test/main.cpp.o.provides: CMakeFiles/classification.dir/test/main.cpp.o.requires
	$(MAKE) -f CMakeFiles/classification.dir/build.make CMakeFiles/classification.dir/test/main.cpp.o.provides.build
.PHONY : CMakeFiles/classification.dir/test/main.cpp.o.provides

CMakeFiles/classification.dir/test/main.cpp.o.provides.build: CMakeFiles/classification.dir/test/main.cpp.o

# Object files for target classification
classification_OBJECTS = \
"CMakeFiles/classification.dir/test/main.cpp.o"

# External object files for target classification
classification_EXTERNAL_OBJECTS =

classification: CMakeFiles/classification.dir/test/main.cpp.o
classification: CMakeFiles/classification.dir/build.make
classification: libclassify.so
classification: /usr/local/lib/libopencv_videostab.so.2.4.13
classification: /usr/local/lib/libopencv_ts.a
classification: /usr/local/lib/libopencv_superres.so.2.4.13
classification: /usr/local/lib/libopencv_stitching.so.2.4.13
classification: /usr/local/lib/libopencv_contrib.so.2.4.13
classification: /usr/local/lib/libopencv_nonfree.so.2.4.13
classification: /usr/local/lib/libopencv_ocl.so.2.4.13
classification: /usr/local/lib/libopencv_gpu.so.2.4.13
classification: /usr/local/lib/libopencv_photo.so.2.4.13
classification: /usr/local/lib/libopencv_objdetect.so.2.4.13
classification: /usr/local/lib/libopencv_legacy.so.2.4.13
classification: /usr/local/lib/libopencv_video.so.2.4.13
classification: /usr/local/lib/libopencv_ml.so.2.4.13
classification: /usr/local/cuda-8.0/lib64/libcufft.so
classification: /usr/local/lib/libopencv_calib3d.so.2.4.13
classification: /usr/local/lib/libopencv_features2d.so.2.4.13
classification: /usr/local/lib/libopencv_flann.so.2.4.13
classification: /home/liusj/dl/caffe/build/lib/libcaffe.so.1.0.0
classification: /usr/local/lib/libopencv_highgui.so.2.4.13
classification: /usr/local/lib/libopencv_imgproc.so.2.4.13
classification: /usr/local/lib/libopencv_core.so.2.4.13
classification: /usr/local/cuda-8.0/lib64/libcudart.so
classification: /usr/local/cuda-8.0/lib64/libnppc.so
classification: /usr/local/cuda-8.0/lib64/libnppi.so
classification: /usr/local/cuda-8.0/lib64/libnpps.so
classification: /home/liusj/dl/caffe/build/lib/libcaffeproto.a
classification: /usr/local/lib/libboost_system.so
classification: /usr/local/lib/libboost_thread.so
classification: /usr/local/lib/libboost_filesystem.so
classification: /usr/local/lib/libglog.so
classification: /usr/local/lib/libgflags.a
classification: /usr/local/lib/libprotobuf.so
classification: /usr/lib64/libhdf5_hl.so
classification: /usr/lib64/libhdf5.so
classification: /usr/lib64/libhdf5_hl.so
classification: /usr/lib64/libhdf5.so
classification: /usr/local/lib/liblmdb.so
classification: /usr/lib64/libleveldb.so
classification: /usr/local/cuda-8.0/lib64/libcudart.so
classification: /usr/local/cuda-8.0/lib64/libcurand.so
classification: /usr/local/cuda-8.0/lib64/libcublas.so
classification: /usr/local/cuda-8.0/lib64/libcudnn.so
classification: /usr/lib64/libopenblas.so
classification: /usr/local/lib/libboost_python.so
classification: CMakeFiles/classification.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --red --bold "Linking CXX executable classification"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/classification.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/classification.dir/build: classification
.PHONY : CMakeFiles/classification.dir/build

CMakeFiles/classification.dir/requires: CMakeFiles/classification.dir/test/main.cpp.o.requires
.PHONY : CMakeFiles/classification.dir/requires

CMakeFiles/classification.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/classification.dir/cmake_clean.cmake
.PHONY : CMakeFiles/classification.dir/clean

CMakeFiles/classification.dir/depend:
	cd /home/liusj/dl/classifier_play/examples/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/liusj/dl/classifier_play/examples /home/liusj/dl/classifier_play/examples /home/liusj/dl/classifier_play/examples/build /home/liusj/dl/classifier_play/examples/build /home/liusj/dl/classifier_play/examples/build/CMakeFiles/classification.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/classification.dir/depend

