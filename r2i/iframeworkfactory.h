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

#ifndef R2I_IFRAMEWORKFACTORY_H
#define R2I_IFRAMEWORKFACTORY_H

#include <r2i/iengine.h>
#include <r2i/iloader.h>
#include <r2i/imodel.h>
#include <r2i/iparameters.h>
#include <r2i/frameworkmeta.h>

#include <memory>
#include <string>

/**
 * R2Inference Namespace
 */
namespace r2i {
/**
 *  Implements the interface to abstract factory to create framework related
 * objects.
 */
class IFrameworkFactory {

 public:
  /**
   * Numerical codes identifying supported frameworks. Not that not
   * all frameworks will be available at runtime. For example, some of
   * them may be disabled at configure by the user, automatically if
   * no development where found or if the system doesn't seem to have
   * to appropriate hardware.
   */
  enum Code {
    /**
     * Intel Movidius Neural Compute software developer kit
     */
    NCSDK,

    /**
     * Number of supported frameworks, mostly for testing purposes.
     */
    MAX_FRAMEWORK
  };

  static std::unique_ptr<IFrameworkFactory> MakeFactory (Code code,
      RuntimeError &error);

  /**
   * \brief Creates an ILoader based on a particular Framework
   * \param error [out] RuntimeError with a description of an error.
   * \return a valid ILoader for the framework or nullptr in case of error.
   */
  virtual std::unique_ptr<r2i::ILoader> MakeLoader (RuntimeError &error) = 0;

  /**
   * \brief Creates a valid IEngine for a particular Framework
   * \param error [out] RuntimeError with a description of an error.
   * \return a valid IEngine for the framework or nullptr in case of error.
   */
  virtual std::unique_ptr<r2i::IEngine> MakeEngine (RuntimeError &error) = 0;

  /**
   * \brief Creates a valid IParameters on a particular Framework
   * \param error [out] RuntimeError with a description of an error.
   * \return a valid IParameters of a framework or nullptr in case of error.
   */
  virtual std::unique_ptr<r2i::IParameters> MakeParameters (
    RuntimeError &error) = 0;

  /**
   * \brief Creates the FrameworkMetadata of particular Framework
   * \param error [out] RuntimeError with a description of an error.
   * \return A FrameworkMetadata of a framework which is valid only if
   * no error was set.
   */
  virtual r2i::FrameworkMeta GetDescription (RuntimeError &error) = 0;

  /**
   * \brief Default destructor
   */
  virtual ~IFrameworkFactory () {};
};

}

#endif // R2I_IFRAMEWORKFACTORY_H
