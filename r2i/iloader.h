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

#ifndef R2I_ILOADER_H
#define R2I_ILOADER_H

#include <r2i/imodel.h>
#include <r2i/runtime_error.h>

#include <memory>
#include <string>

/**
 * R2Inference Namespace
 */
namespace r2i
{
/**
 *  Implements the interface to validate a IModel implementation
 *  for an IEngine implementation
 */
class ILoader
{

public:
  /**
   * \brief Checks consistency of a trained model.
   * \param in_path A string of the absolute path to a model for evaluation.
   * \param error [out] RuntimeError with a description of an error.
   * \return A validated IModel for an IEngine or nullptr in case of error.
   */
  virtual std::unique_ptr<r2i::IModel> Load (const std::string &in_path,
    r2i::RuntimeError &error) = 0;
};

}

#endif // R2I_ILOADER_H
