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

#include "r2i/imageformat.h"

#include <string>
#include <unordered_map>

namespace r2i {

static std::unordered_map<int, std::pair<const std::string, int>>
format_descriptors ({
  {ImageFormat::Code::RGB, {"RGB", 3}},
  {ImageFormat::Code::BGR, {"BGR", 3}},
  {ImageFormat::Code::GRAY, {"Grayscale", 1}},
  {ImageFormat::Code::UNKNOWN_FORMAT, {"Unknown format", 0}}
});

ImageFormat::ImageFormat ()
  : code(Code::UNKNOWN_FORMAT) {
}

ImageFormat::ImageFormat (Code code)
  : code(code) {
}

ImageFormat::Code ImageFormat::GetCode () {
  return this->code;
}

std::pair<const std::string, int> search (ImageFormat::Code code) {
  auto search = format_descriptors.find (code);
  if (format_descriptors.end () == search) {
    search = format_descriptors.find (ImageFormat::Code::UNKNOWN_FORMAT);
  }
  return search->second;  
}

const std::string ImageFormat::GetDescription () {
  std::pair<const std::string, int> descriptor = search(this->code);
  return descriptor.first;
}

int ImageFormat::GetNumPlanes () {
  std::pair<const std::string, int> descriptor = search(this->code);
  return descriptor.second;
}
}
