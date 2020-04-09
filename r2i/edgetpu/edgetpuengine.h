/* Copyright (C) 2018 RidgeRun, LLC (http://www.ridgerun.com)
 * All Rights Reserved.
 *
 * The contents of this software are proprietary and confidential to RidgeRun,
 * LLC.  No part of this program may be photocopied, reproduced or translated
 * into another programming language without prior written consent of
 * RidgeRun, LLC.  The user is free to modify the source code after obtaining
 * a software license from RidgeRun.  All source code changes must be provided
 * back to RidgeRun without any encumbrance.
*/

#ifndef R2I_EDGETPU_ENGINE_H
#define R2I_EDGETPU_ENGINE_H

#include <memory>

#include <r2i/tflite/engine.h>
#include <r2i/tflite/model.h>

// TODO: Verify that all the includes are still valid
#include <tensorflow/lite/model.h>
#include <tensorflow/lite/kernels/register.h>
#include <tensorflow/lite/string_util.h>
#include <edgetpu.h>

namespace r2i {
namespace edgetpu {

class EdgeTPUEngine : public r2i::tflite::Engine {
 public:
  EdgeTPUEngine ();

  r2i::RuntimeError Start () override;

  ~EdgeTPUEngine ();

 private:
  std::unique_ptr<::tflite::Interpreter> BuildEdgeTpuInterpreter(
    const ::tflite::FlatBufferModel &model, ::edgetpu::EdgeTpuContext *context);
};

}
}
#endif //R2I_EDGETPU_ENGINE_H
