// SPDX-License-Identifier: MIT
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
#include "rocm_ops.hpp"
#include "aiter_stream.h"
#include "epx.h"

PYBIND11_MODULE(AITER_EXTENSION_NAME, m)
{
      EPX_PYBIND;
}
