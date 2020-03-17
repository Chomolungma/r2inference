/* Copyright (C) 2020 RidgeRun, LLC (http://www.ridgerun.com)
 * All Rights Reserved.
 *
 * The contents of this software are proprietary and confidential to RidgeRun,
 * LLC.  No part of this program may be photocopied, reproduced or translated
 * into another programming language without prior written consent of
 * RidgeRun, LLC.  The user is free to modify the source code after obtaining
 * a software license from RidgeRun.  All source code changes must be provided
 * back to RidgeRun without any encumbrance.
*/

#include <r2i/r2i.h>
#include <r2i/tensorrt/prediction.h>
#include <fstream>
#include <cstring>
#include <memory>

#include <CppUTest/CommandLineTestRunner.h>
#include <CppUTest/TestHarness.h>

#include <cuda.h>
#include <cuda_runtime.h>

#define INPUTS 3

bool cudaMemCpyError = false;
bool mallocError = false;

__host__ cudaError_t CUDARTAPI cudaMemcpy(void *dst, const void *src,
    size_t count, enum cudaMemcpyKind kind) {
  if (!cudaMemCpyError) {
    memcpy ( dst, src, count );
    return cudaSuccess;
  } else {
    return cudaErrorInvalidValue;
  }
}

void *malloc (size_t size) {
  if (!mallocError) {
    return calloc (1, size);;
  } else {
    return nullptr;
  }
}

TEST_GROUP (TensorRTPrediction) {
  r2i::RuntimeError error;

  r2i::tensorrt::Prediction prediction;
  float matrix[INPUTS] = {0.2, 0.4, 0.6};

  int64_t raw_input_dims[1] = {INPUTS};

  void setup () {
    error.Clean();
    cudaMemCpyError = false;
  }

  void teardown () {
  }
};

TEST (TensorRTPrediction, SetResultBufferSuccess) {
  error = prediction.SetResultBuffer(matrix, INPUTS * sizeof(float));
  LONGS_EQUAL (r2i::RuntimeError::Code::EOK, error.GetCode());
}

TEST (TensorRTPrediction, SetNullResultBuffer) {
  error = prediction.SetResultBuffer(nullptr, INPUTS * sizeof(float));
  LONGS_EQUAL (r2i::RuntimeError::Code::NULL_PARAMETER, error.GetCode());
}

TEST (TensorRTPrediction, Prediction) {
  double result;

  error = prediction.SetResultBuffer(matrix, INPUTS * sizeof(float));
  LONGS_EQUAL (r2i::RuntimeError::Code::EOK, error.GetCode());

  result = prediction.At (0, error);
  LONGS_EQUAL (r2i::RuntimeError::Code::EOK, error.GetCode());
  DOUBLES_EQUAL (matrix[0], result, 0.05);
}

TEST (TensorRTPrediction, PredictionGetResultSize) {
  error = prediction.SetResultBuffer(matrix, INPUTS * sizeof(float));
  LONGS_EQUAL (r2i::RuntimeError::Code::EOK, error.GetCode());

  LONGS_EQUAL (INPUTS * sizeof(float), prediction.GetResultSize());
}

TEST (TensorRTPrediction, PredictionNoTensor) {
  prediction.At (0, error);
  LONGS_EQUAL (r2i::RuntimeError::Code::INVALID_FRAMEWORK_PARAMETER,
               error.GetCode());
}

TEST (TensorRTPrediction, PredictionNonExistentIndex) {
  error = prediction.SetResultBuffer(matrix, INPUTS * sizeof(float));
  LONGS_EQUAL (r2i::RuntimeError::Code::EOK, error.GetCode());

  prediction.At (5, error);
  LONGS_EQUAL (r2i::RuntimeError::Code::MEMORY_ERROR, error.GetCode());
}

TEST (TensorRTPrediction, PredictionCudaMemCpyError) {
  cudaMemCpyError = true;
  error = prediction.SetResultBuffer(matrix, INPUTS * sizeof(float));
  LONGS_EQUAL (r2i::RuntimeError::Code::MEMORY_ERROR, error.GetCode());
}

TEST (TensorRTPrediction, PredictionMallocError) {
  cudaMemCpyError = true;
  error = prediction.SetResultBuffer(matrix, INPUTS * sizeof(float));
  LONGS_EQUAL (r2i::RuntimeError::Code::MEMORY_ERROR, error.GetCode());
}

int main (int ac, char **av) {
  //MemoryLeakWarningPlugin::turnOffNewDeleteOverloads();

  return CommandLineTestRunner::RunAllTests (ac, av);
}
