/* Copyright (C) 2018-2020 RidgeRun, LLC (http://www.ridgerun.com)
 * All Rights Reserved.
 *
 * The contents of this software are proprietary and confidential to RidgeRun,
 * LLC.  No part of this program may be photocopied, reproduced or translated
 * into another programming language without prior written consent of
 * RidgeRun, LLC.  The user is free to modify the source code after obtaining
 * a software license from RidgeRun.  All source code changes must be provided
 * back to RidgeRun without any encumbrance.
*/

// #include <fstream>
// #include <iostream>
// #include <vector>

#include "r2i/tensorrt/model.h"

namespace r2i {
namespace tensorrt {

Model::Model () {
}

RuntimeError Model::Start (const std::string &name) {
  RuntimeError error;

  return error;
}

RuntimeError Model::Set (std::shared_ptr<nvinfer1::ICudaEngine> tensorrtmodel) {
  RuntimeError error;

  if (nullptr == tensorrtmodel) {
    error.Set (RuntimeError::Code::NULL_PARAMETER,
               "Trying to set model with null model pointer");
    return error;
  }

  this->engine = tensorrtmodel;

  return error;
}

std::shared_ptr<nvinfer1::ICudaEngine> Model::GetTREngineModel () {
  return this->engine;
}

} // namespace tensorrt
} // namespace r2i
