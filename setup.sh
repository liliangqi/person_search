# build nms
cd nms/src/cuda
echo "Compiling nms kernels by nvcc..."
nvcc -c -o nms_kernel.cu.o nms_kernel.cu -x cu -Xcompiler -fPIC -arch=sm_61
cd ../..
python build.py
cd ..

# test on my own computer
ln -s ../person_search_caffe/data/psdb data
cd data
mkdir pretrained
cp /home/liliangqi/res50.pth pretrained/
cd ..