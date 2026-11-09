# flatbuffers 编译
cd 3rd_party/flatbuffers/tmp
cmake ..
cmake --build . --target  flatc

bash ./schema/generate.sh 


mkdir build
cd build
cmake .. -DMNN_BUILD_TEST=ON -DMNN_BUILD_CONVERTER=ON
make -j8