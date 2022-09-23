CUDA_PATH=/usr/local/cuda
HOST_COMPILER ?= g++
NVCC=${CUDA_PATH}/bin/nvcc -ccbin ${HOST_COMPILER}
TARGET=cuSpeckle3D

INCLUDES= -I${CUDA_PATH}/samples/common/inc -I${CUDA_PATH}/include
INCLUDES+= -Iinclude 
NVCC_FLAGS=-m64 -lineinfo

IS_CUDA_11:=${shell expr `$(NVCC) --version | grep compilation | grep -Eo -m 1 '[0-9]+.[0-9]' | head -1` \>= 11.0}

# Gencode argumentes
SMS = 35 37 50 52 60 61 70 75
ifeq "$(IS_CUDA_11)" "1"
SMS = 52 60 61 70 75 80
endif
$(foreach sm, ${SMS}, $(eval GENCODE_FLAGS += -gencode arch=compute_$(sm),code=sm_$(sm)))

LIBS=-lcurand 

SOURCES=src/* lib/*

CXX_FLAGS=-O3 -lpng

DEBUG_FLAGS=-g -G

${TARGET}:$(SOURCES)
	${NVCC} ${INCLUDES} ${ALL_CCFLAGS} ${CXX_FLAGS} ${GENCODE_FLAGS} $(SOURCES) $(LIBS) -o ${TARGET}

debug:
	${NVCC} ${INCLUDES} ${ALL_CCFLAGS} ${CXX_FLAGS} ${GENCODE_FLAGS} ${DEBUG_FLAGS} $(SOURCES) $(LIBS) -o ${TARGET}

clean:
	rm -f ${TARGET}