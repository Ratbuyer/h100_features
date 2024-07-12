sm_version=90
NVCC=/usr/local/cuda-12.4/bin/nvcc
INCLUDES=-I./headers/device/ -I./headers/host/
OPTIMIZATION=-O0
LINKS=-lcudart -lcuda
OUTPUT=bins/bin


test:
	${NVCC} -arch=sm_${sm_version} ${OPTIMIZATION} ${INCLUDES} ${LINKS} -o ${OUTPUT} examples/test.cu

dense:
	${NVCC} -arch=sm_${sm_version} ${OPTIMIZATION} ${INCLUDES} ${LINKS} -o ${OUTPUT} examples/wgmma_dense.cu

sparse:
	${NVCC} -arch=sm_${sm_version} ${OPTIMIZATION} ${INCLUDES} ${LINKS} -o ${OUTPUT} examples/wgmma_sparse.cu

overlap:
	${NVCC} -arch=sm_${sm_version} ${OPTIMIZATION} ${INCLUDES} ${LINKS} -o ${OUTPUT} examples/overlap.cu

gemm:
	${NVCC} -arch=sm_${sm_version} ${OPTIMIZATION} ${INCLUDES} ${LINKS} -o ${OUTPUT} examples/gemm.cu

tma_1d:
	${NVCC} -arch=sm_${sm_version} ${OPTIMIZATION} ${INCLUDES} ${LINKS} -o ${OUTPUT} examples/tma_1d.cu

tma_2d:
	${NVCC} -arch=sm_${sm_version} ${OPTIMIZATION} ${INCLUDES} ${LINKS} -o ${OUTPUT} examples/tma_2d.cu

push:
	git add .
	git commit -m "update"
	git push

all:
	make test
	make run

run:
	./bins/bin

clean:
	rm -rf bins/*