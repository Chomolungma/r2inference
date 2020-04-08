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

#include "edgetpuengine.h"

#include "r2i/tflite/prediction.h"
#include "r2i/tflite/frame.h"
#include <tensorflow/lite/model.h>
#include <tensorflow/lite/kernels/register.h>
#include <tensorflow/lite/string_util.h>

namespace r2i {
namespace edgetpu {

EdgeTPUEngine::EdgeTPUEngine () {
  this->state = r2i::tflite::Engine::State::STOPPED;
  this->model = nullptr;
  this->number_of_threads = 0;
  this->allow_fp16 = 0;
}

RuntimeError EdgeTPUEngine::Start ()  {
  RuntimeError error;

  return error;
}

EdgeTPUEngine::~EdgeTPUEngine () {
  r2i::tflite::Engine::Stop();
}

} //namespace edgetpu
} //namepsace r2i
