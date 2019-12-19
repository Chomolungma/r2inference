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

#include "r2i/tensorflowlite/model.h"

namespace r2i {
namespace tensorflowlite {

Model::Model () {
  this->tflemodel = nullptr;
}

RuntimeError Model::Start (const std::string &name) {
  RuntimeError error;

  return error;
}

RuntimeError Model::Set (std::shared_ptr<TfLiteModel> tfltmodel) {
  RuntimeError error;

  if (nullptr != tfltmodel) {
    error.Set (RuntimeError::Code::NULL_PARAMETER,
               "Trying to set model with null model pointer");
    return error;
  }

  this->tflemodel = tfltmodel;

  return error;
}

} // namespace tensorflowlite
} // namespace r2i
