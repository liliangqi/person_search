# build nms
cd nms/src/cuda
echo "Compiling nms kernels by nvcc..."
nvcc -c -o roi_pooling_kernel.cu.o roi_pooling_kernel.cu -x cu -Xcompiler -fPIC -arch=sm_61
cd ../..
python build.py
cd ..