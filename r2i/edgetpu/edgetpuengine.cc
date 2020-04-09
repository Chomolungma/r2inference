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

#include <r2i/tflite/prediction.h>
#include <r2i/tflite/frame.h>

#include <iostream>

namespace r2i {
namespace edgetpu {

// TODO: Add error handlinds
// TODO: Add verifications of TPUs
// TODO: Do not pass this variables as arguments (mode and context)
std::unique_ptr<::tflite::Interpreter> EdgeTPUEngine::BuildEdgeTpuInterpreter(
  const ::tflite::FlatBufferModel &model, ::edgetpu::EdgeTpuContext *context) {

  ::tflite::ops::builtin::BuiltinOpResolver resolver;
  resolver.AddCustom(::edgetpu::kCustomOp, ::edgetpu::RegisterCustomOp());

  std::unique_ptr<::tflite::Interpreter> interpreter;
  ::tflite::InterpreterBuilder interpreter_builder(model.GetModel(), resolver);
  if (interpreter_builder(&interpreter) != kTfLiteOk) {
    std::cerr << "Error in interpreter initialization." << std::endl;
    return nullptr;
  }

  interpreter->SetExternalContext(kTfLiteEdgeTpuContext, context);
  interpreter->SetNumThreads(1);
  if (interpreter->AllocateTensors() != kTfLiteOk) {
    std::cerr << "Failed to allocate tensors." << std::endl;
    return nullptr;
  }

  return interpreter;
}

EdgeTPUEngine::EdgeTPUEngine () {
  this->state = r2i::tflite::Engine::State::STOPPED;
  this->model = nullptr;
  this->number_of_threads = 0;
  this->allow_fp16 = 0;
}

RuntimeError EdgeTPUEngine::Start ()  {
  RuntimeError error;

  if (r2i::tflite::Engine::State::STARTED == this->state) {
    error.Set (RuntimeError::Code::WRONG_ENGINE_STATE,
               "Engine already started");
    return error;
  }

  if (nullptr == this->model) {
    error.Set (RuntimeError::Code:: NULL_PARAMETER,
               "Model not set yet");
    return error;
  }

  if (!this->interpreter) {
    // Get EdgeTPU context
    std::shared_ptr<::edgetpu::EdgeTpuContext> edgetpu_context
      = ::edgetpu::EdgeTpuManager::GetSingleton()->OpenDevice();

    std::unique_ptr<::tflite::Interpreter> interpreter =
      this->BuildEdgeTpuInterpreter(*this->model->GetTfliteModel(),
                                    edgetpu_context.get());

    if (!interpreter) {
      error.Set (RuntimeError::Code::FRAMEWORK_ERROR,
                 "Failed to construct interpreter");
      return error;
    }

    std::shared_ptr<::tflite::Interpreter> tflite_interpreter_shared{std::move(interpreter)};

    this->interpreter = tflite_interpreter_shared;
  }

  this->state = r2i::tflite::Engine::State::STARTED;

  return error;
}

EdgeTPUEngine::~EdgeTPUEngine () {
  r2i::tflite::Engine::Stop();
}

} //namespace edgetpu
} //namepsace r2i
