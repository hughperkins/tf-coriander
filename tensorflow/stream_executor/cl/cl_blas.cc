/* Copyright 2015 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

// Include CLBlast headers early, and then set EIGEN_HAS_CL_FP16
// if we have new enough CL (which we will only know after including
// cl.h). This ensures that Eigen's Half.h does not attempt to make its own
// __half typedef if CL has already defined one (and conversely, that we do
// not include <cl_fp16.h> after Half.h has made its typedef).
#include "cuda.h"
// #include "cl/include/cublas_v2.h"

// #if CL_VERSION >= 7050
// #define EIGEN_HAS_CL_FP16
// #endif

// #if CL_VERSION >= 8000
// #define SE_CL_DATA_HALF CL_R_16F
// #else
// #define SE_CL_DATA_HALF CUBLAS_DATA_HALF
// #endif  

#include "tensorflow/stream_executor/cl/cl_blas.h"

#include <complex>

#include "tensorflow/stream_executor/cl/cl_activation.h"
#include "tensorflow/stream_executor/cl/cl_gpu_executor.h"
#include "tensorflow/stream_executor/cl/cl_helpers.h"
#include "tensorflow/stream_executor/cl/cl_platform_id.h"
#include "tensorflow/stream_executor/cl/cl_stream.h"
#include "tensorflow/stream_executor/device_memory.h"
// #include "tensorflow/stream_executor/dso_loader.h"
#include "tensorflow/stream_executor/lib/env.h"
#include "tensorflow/stream_executor/lib/initialize.h"
#include "tensorflow/stream_executor/lib/status.h"
#include "tensorflow/stream_executor/lib/status_macros.h"
#include "tensorflow/stream_executor/lib/strcat.h"
#include "tensorflow/stream_executor/lib/stringprintf.h"
#include "tensorflow/stream_executor/platform/logging.h"
#include "tensorflow/stream_executor/platform/port.h"
#include "tensorflow/stream_executor/plugin_registry.h"
#include "tensorflow/stream_executor/scratch_allocator.h"
#include "tensorflow/stream_executor/stream_executor.h"

#include <iostream>

namespace perftools {
namespace gputools {
namespace cl {

PLUGIN_REGISTRY_DEFINE_PLUGIN_ID(kClBlasPlugin);

namespace dynload {

// #define PERFTOOLS_GPUTOOLS_CUBLAS_WRAP(__name)                              \
//   struct DynLoadShim__##__name {                                            \
//     static const char *kName;                                               \
//     using FuncPointerT = std::add_pointer<decltype(::__name)>::type;        \
//     static void *GetDsoHandle() {                                           \
//       static auto status = internal::CachedDsoLoader::GetCublasDsoHandle(); \
//       return status.ValueOrDie();                                           \
//     }                                                                       \
//     static FuncPointerT LoadOrDie() {                                       \
//       void *f;                                                              \
//       port::Status s = port::Env::Default()->GetSymbolFromLibrary(          \
//           GetDsoHandle(), kName, &f);                                       \
//       CHECK(s.ok()) << "could not find " << kName                           \
//                     << " in CLBlast DSO; dlerror: " << s.error_message();   \
//       return reinterpret_cast<FuncPointerT>(f);                             \
//     }                                                                       \
//     static FuncPointerT DynLoad() {                                         \
//       static FuncPointerT f = LoadOrDie();                                  \
//       return f;                                                             \
//     }                                                                       \
//     template <typename... Args>                                             \
//     cublasStatus_t operator()(CLExecutor *parent, Args... args) {         \
//       cl::ScopedActivateExecutorContext sac{parent};                      \
//       return DynLoad()(args...);                                            \
//     }                                                                       \
//   } __name;                                                                 \
//   const char *DynLoadShim__##__name::kName = #__name;

// #define PERFTOOLS_GPUTOOLS_CUBLAS_V2_WRAP(__name) \
//   PERFTOOLS_GPUTOOLS_CUBLAS_WRAP(__name)

//#define CUBLAS_BLAS_ROUTINE_EACH(__macro) 
//   __macro(cublasSgemm)                    
//   __macro(cublasSnrm2)                    
//   __macro(cublasDnrm2)                    
//   __macro(cublasScnrm2)                   
//   __macro(cublasDznrm2)                   
//   __macro(cublasSdot)                     
//   __macro(cublasDdot)                     
//   __macro(cublasCdotu)                    
//   __macro(cublasCdotc)                    
//   __macro(cublasZdotu)                    
//   __macro(cublasZdotc)                    
//   __macro(cublasSscal)                    
//   __macro(cublasDscal)                    
//   __macro(cublasCscal)                    
//   __macro(cublasCsscal)                   
//   __macro(cublasZscal)                    
//   __macro(cublasZdscal)                   
//   __macro(cublasSaxpy)                    
//   __macro(cublasDaxpy)                    
//   __macro(cublasCaxpy)                    
//   __macro(cublasZaxpy)                    
//   __macro(cublasScopy)                    
//   __macro(cublasDcopy)                    
//   __macro(cublasCcopy)                    
//   __macro(cublasZcopy)                    
//   __macro(cublasSswap)                    
//   __macro(cublasDswap)                    
//   __macro(cublasCswap)                    
//   __macro(cublasZswap)                    
//   __macro(cublasIsamax)                   
//   __macro(cublasIdamax)                   
//   __macro(cublasIcamax)                   
//   __macro(cublasIzamax)                   
//   __macro(cublasIsamin)                   
//   __macro(cublasIdamin)                   
//   __macro(cublasIcamin)                   
//   __macro(cublasIzamin)                   
//   __macro(cublasSasum)                    
//   __macro(cublasDasum)                    
//   __macro(cublasScasum)                   
//   __macro(cublasDzasum)                   
//   __macro(cublasSrot)                     
//   __macro(cublasDrot)                     
//   __macro(cublasCrot)                     
//   __macro(cublasCsrot)                    
//   __macro(cublasZrot)                     
//   __macro(cublasZdrot)                    
//   __macro(cublasSrotg)                    
//   __macro(cublasDrotg)                    
//   __macro(cublasCrotg)                    
//   __macro(cublasZrotg)                    
//   __macro(cublasSrotm)                    
//   __macro(cublasDrotm)                    
//   __macro(cublasSrotmg)                   
//   __macro(cublasDrotmg)                   
//   __macro(cublasSgemv)                    
//   __macro(cublasDgemv)                    
//   __macro(cublasCgemv)                    
//   __macro(cublasZgemv)                    
//   __macro(cublasSgbmv)                    
//   __macro(cublasDgbmv)                    
//   __macro(cublasCgbmv)                    
//   __macro(cublasZgbmv)                    
//   __macro(cublasStrmv)                    
//   __macro(cublasDtrmv)                    
//   __macro(cublasCtrmv)                    
//   __macro(cublasZtrmv)                    
//   __macro(cublasStbmv)                    
//   __macro(cublasDtbmv)                    
//   __macro(cublasCtbmv)                    
//   __macro(cublasZtbmv)                    
//   __macro(cublasStpmv)                    
//   __macro(cublasDtpmv)                    
//   __macro(cublasCtpmv)                    
//   __macro(cublasZtpmv)                    
//   __macro(cublasStrsv)                    
//   __macro(cublasDtrsv)                    
//   __macro(cublasCtrsv)                    
//   __macro(cublasZtrsv)                    
//   __macro(cublasStpsv)                    
//   __macro(cublasDtpsv)                    
//   __macro(cublasCtpsv)                    
//   __macro(cublasZtpsv)                    
//   __macro(cublasStbsv)                    
//   __macro(cublasDtbsv)                    
//   __macro(cublasCtbsv)                    
//   __macro(cublasZtbsv)                    
//   __macro(cublasSsymv)                    
//   __macro(cublasDsymv)                    
//   __macro(cublasCsymv)                    
//   __macro(cublasZsymv)                    
//   __macro(cublasChemv)                    
//   __macro(cublasZhemv)                    
//   __macro(cublasSsbmv)                    
//   __macro(cublasDsbmv)                    
//   __macro(cublasChbmv)                    
//   __macro(cublasZhbmv)                    
//   __macro(cublasSspmv)                    
//   __macro(cublasDspmv)                    
//   __macro(cublasChpmv)                    
//   __macro(cublasZhpmv)                    
//   __macro(cublasSger)                     
//   __macro(cublasDger)                     
//   __macro(cublasCgeru)                    
//   __macro(cublasCgerc)                    
//   __macro(cublasZgeru)                    
//   __macro(cublasZgerc)                    
//   __macro(cublasSsyr)                     
//   __macro(cublasDsyr)                     
//   __macro(cublasCsyr)                     
//   __macro(cublasZsyr)                     
//   __macro(cublasCher)                     
//   __macro(cublasZher)                     
//   __macro(cublasSspr)                     
//   __macro(cublasDspr)                     
//   __macro(cublasChpr)                     
//   __macro(cublasZhpr)                     
//   __macro(cublasSsyr2)                    
//   __macro(cublasDsyr2)                    
//   __macro(cublasCsyr2)                    
//   __macro(cublasZsyr2)                    
//   __macro(cublasCher2)                    
//   __macro(cublasZher2)                    
//   __macro(cublasSspr2)                    
//   __macro(cublasDspr2)                    
//   __macro(cublasChpr2)                    
//   __macro(cublasZhpr2)                    
//   __macro(cublasDgemm)                    
//   __macro(cublasCgemm)                    
//   __macro(cublasZgemm)                    
//   __macro(cublasSsyrk)                    
//   __macro(cublasDsyrk)                    
//   __macro(cublasCsyrk)                    
//   __macro(cublasZsyrk)                    
//   __macro(cublasCherk)                    
//   __macro(cublasZherk)                    
//   __macro(cublasSsyr2k)                   
//   __macro(cublasDsyr2k)                   
//   __macro(cublasCsyr2k)                   
//   __macro(cublasZsyr2k)                   
//   __macro(cublasCher2k)                   
//   __macro(cublasZher2k)                   
//   __macro(cublasSsyrkx)                   
//   __macro(cublasDsyrkx)                   
//   __macro(cublasCsyrkx)                   
//   __macro(cublasZsyrkx)                   
//   __macro(cublasCherkx)                   
//   __macro(cublasZherkx)                   
//   __macro(cublasSsymm)                    
//   __macro(cublasDsymm)                    
//   __macro(cublasCsymm)                    
//   __macro(cublasZsymm)                    
//   __macro(cublasChemm)                    
//   __macro(cublasZhemm)                    
//   __macro(cublasStrsm)                    
//   __macro(cublasDtrsm)                    
//   __macro(cublasCtrsm)                    
//   __macro(cublasZtrsm)                    
//   __macro(cublasStrmm)                    
//   __macro(cublasDtrmm)                    
//   __macro(cublasCtrmm)                    
//   __macro(cublasZtrmm)                    
//   __macro(cublasSgeam)                    
//   __macro(cublasDgeam)                    
//   __macro(cublasCgeam)                    
//   __macro(cublasZgeam)                    
//   __macro(cublasSdgmm)                    
//   __macro(cublasDdgmm)                    
//   __macro(cublasCdgmm)                    
//   __macro(cublasZdgmm)

// PERFTOOLS_GPUTOOLS_CUBLAS_V2_WRAP(cublasCreate)
// PERFTOOLS_GPUTOOLS_CUBLAS_V2_WRAP(cublasDestroy)
// PERFTOOLS_GPUTOOLS_CUBLAS_V2_WRAP(cublasSetStream)
// PERFTOOLS_GPUTOOLS_CUBLAS_V2_WRAP(cublasSetPointerMode)
// PERFTOOLS_GPUTOOLS_CUBLAS_V2_WRAP(cublasGetPointerMode)
// PERFTOOLS_GPUTOOLS_CUBLAS_WRAP(cublasSgemmBatched)
// PERFTOOLS_GPUTOOLS_CUBLAS_WRAP(cublasDgemmBatched)
// PERFTOOLS_GPUTOOLS_CUBLAS_WRAP(cublasCgemmBatched)
// PERFTOOLS_GPUTOOLS_CUBLAS_WRAP(cublasZgemmBatched)
// CUBLAS_BLAS_ROUTINE_EACH(PERFTOOLS_GPUTOOLS_CUBLAS_V2_WRAP)

// #if CL_VERSION >= 7050
// PERFTOOLS_GPUTOOLS_CUBLAS_WRAP(cublasSgemmEx)
// #endif

}  // namespace dynload

static string ToString(cublasStatus_t status) {
  switch (status) {
    case CUBLAS_STATUS_SUCCESS:
      return "CUBLAS_STATUS_SUCCESS";
    case CUBLAS_STATUS_NOT_INITIALIZED:
      return "CUBLAS_STATUS_NOT_INITIALIZED";
    case CUBLAS_STATUS_ALLOC_FAILED:
      return "CUBLAS_STATUS_ALLOC_FAILED";
    case CUBLAS_STATUS_INVALID_VALUE:
      return "CUBLAS_STATUS_INVALID_VALUE";
    case CUBLAS_STATUS_ARCH_MISMATCH:
      return "CUBLAS_STATUS_ARCH_MISMATCH";
    case CUBLAS_STATUS_MAPPING_ERROR:
      return "CUBLAS_STATUS_MAPPING_ERROR";
    case CUBLAS_STATUS_EXECUTION_FAILED:
      return "CUBLAS_STATUS_EXECUTION_FAILED";
    case CUBLAS_STATUS_INTERNAL_ERROR:
      return "CUBLAS_STATUS_INTERNAL_ERROR";
    default:
      return port::StrCat("<invalid cublas status: ", status, ">");
  }
}

// CLBlast has interfaces that permit pointers to be passed from either the host
// memory space or the device memory space; however, you must instruct it as to
// which address space those pointers are in with cublasSetPointerMode.
//
// This helper sets the CLBlast pointer mode to a desired value for a CLBlast call
// you are about to perform in a given scope.
//
// The prior CLBlast pointer mode is retained and restored when this object goes
// out of scope.
class ScopedCublasPointerMode {
 public:
  // Note that, because the setting of the cublas pointer mode is fallible,
  // construction of this scoped datatype must be paired with a call to
  // Init().
  //
  // Parameters:
  //  handle: The cublas library handle to act upon in setting the pointer mode.
  explicit ScopedCublasPointerMode(CLExecutor *parent, cublasHandle_t handle)
      : parent_(parent), handle_(handle), ok_(false) {}

  // Attempts the switch to the requested scoped pointer mode, new_mode.
  //  
  // Note that when false is returned, an appropriate error has already been
  // logged.
  bool Init(cublasPointerMode_t new_mode) {
    cublasStatus_t ret =
        cublasGetPointerMode_v2(handle_, &old_mode_);
    if (ret != CUBLAS_STATUS_SUCCESS) {
      LOG(ERROR) << "failed to get old cublas pointer mode: " << ToString(ret);
      return ok_ = false;
    }

    ret = cublasSetPointerMode_v2(handle_, new_mode);
    if (ret != CUBLAS_STATUS_SUCCESS) {
      LOG(ERROR) << "failed to set new cublas pointer mode: " << ToString(ret);
      return ok_ = false;
    }

    return ok_ = true;
  }

  // Switches back to the prior pointer mode, if the switch operation was
  // successful in the first place.
  ~ScopedCublasPointerMode() {
    if (ok_) {
      cublasStatus_t ret =
          cublasSetPointerMode_v2(handle_, old_mode_);
      if (ret != CUBLAS_STATUS_SUCCESS) {
        LOG(ERROR) << "failed to set former cublas pointer mode: "
                   << ToString(ret);
      }
    }
  }

 private:
  CLExecutor *parent_;   // Executor establishing this pointer mode for.
  cublasHandle_t handle_;  // Handle to the CLBlast instance of interest.
  cublasPointerMode_t old_mode_;  // Prior CLBlast pointer mode, to be restored.
  bool ok_;                       // Whether the change was successful.
};

bool CLBlas::Init() {
  cublasStatus_t ret = cublasCreate_v2(&blas_);
  if (ret != CUBLAS_STATUS_SUCCESS) {
    LOG(ERROR) << "failed to create cublas handle: " << ToString(ret);
    return false;
  }

  return true;
}

CLBlas::CLBlas(cl::CLExecutor *parent)
    : parent_(CHECK_NOTNULL(parent)), blas_(nullptr) {
      // std::cout << "CLBlast()" << std::endl;
    }

CLBlas::~CLBlas() {
      // std::cout << "~CLBlast()" << std::endl;
  if (blas_ != nullptr) {
    cublasDestroy_v2(blas_);
  }
}

bool CLBlas::SetStream(Stream *stream) {
      // std::cout << "CLBlas::SetStream()" << std::endl;
  CHECK(stream != nullptr);
  CHECK(AsCLStreamValue(stream) != nullptr);
  CHECK(blas_ != nullptr);
  cublasStatus_t ret =
      cublasSetStream_v2(blas_, AsCLStreamValue(stream));
  if (ret != CUBLAS_STATUS_SUCCESS) {
    LOG(ERROR) << "failed to set stream for CLBlast calls: " << ToString(ret);
    return false;
  }

  return true;
}

namespace {

// Helper functions transforming blas arguments into CLBlast arguments.

cublasOperation_t CLBlasTranspose(blas::Transpose trans) {
      // std::cout << "CLBlas::CLBlasTranspoe()" << std::endl;
  switch (trans) {
    case blas::Transpose::kNoTranspose:
      return CUBLAS_OP_N;
    case blas::Transpose::kTranspose:
      return CUBLAS_OP_T;
    case blas::Transpose::kConjugateTranspose:
      return CUBLAS_OP_C;
    default:
      LOG(FATAL) << "Invalid value of blas::Transpose.";
  }
}

cublasFillMode_t CLBlasUpperLower(blas::UpperLower uplo) {
      // std::cout << "CLBlas::CLBlasUpperLower()" << std::endl;
  switch (uplo) {
    case blas::UpperLower::kUpper:
      return CUBLAS_FILL_MODE_UPPER;
    case blas::UpperLower::kLower:
      return CUBLAS_FILL_MODE_LOWER;
    default:
      LOG(FATAL) << "Invalid value of blas::UpperLower.";
  }
}

cublasDiagType_t CLBlasDiagonal(blas::Diagonal diag) {
      // std::cout << "CLBlas::CLBlasDiagonal()" << std::endl;
  switch (diag) {
    case blas::Diagonal::kUnit:
      return CUBLAS_DIAG_UNIT;
    case blas::Diagonal::kNonUnit:
      return CUBLAS_DIAG_NON_UNIT;
    default:
      LOG(FATAL) << "Invalid value of blas::Diagonal.";
  }
}

cublasSideMode_t CLBlasSide(blas::Side side) {
      // std::cout << "CLBlas::CLBlasSide()" << std::endl;
  switch (side) {
    case blas::Side::kLeft:
      return CUBLAS_SIDE_LEFT;
    case blas::Side::kRight:
      return CUBLAS_SIDE_RIGHT;
    default:
      LOG(FATAL) << "Invalid value of blas::Side.";
  }
}

}  // namespace

template <typename FuncT, typename... Args>
bool CLBlas::DoBlasInternal(FuncT cublas_func, Stream *stream,
                              bool pointer_mode_host, Args... args) {
      // std::cout << "CLBlas::DoBlasInternal()" << std::endl;
  mutex_lock lock{mu_};

  CHECK(blas_ != nullptr);
  if (!SetStream(stream)) {
    return false;
  }

  ScopedCublasPointerMode pointer_mode{parent_, blas_};
  if (!pointer_mode.Init(pointer_mode_host ? CUBLAS_POINTER_MODE_HOST
                                           : CUBLAS_POINTER_MODE_DEVICE)) {
    return false;
  }

  cublasStatus_t ret = cublas_func(blas_, args...);
  if (ret != CUBLAS_STATUS_SUCCESS) {
    // LOG(ERROR) << "failed to run CLBlast routine " << cublas_func.kName << ": "
    //            << ToString(ret);
    LOG(ERROR) << "failed to run CLBlast routine " << ": "
               << ToString(ret);
    return false;
  }

  return true;
}

bool CLBlas::DoBlasGemm(Stream *stream, blas::Transpose transa,
                          blas::Transpose transb, uint64 m, uint64 n, uint64 k,
                          float alpha, const DeviceMemory<float> &a, int lda,
                          const DeviceMemory<float> &b, int ldb, float beta,
                          DeviceMemory<float> *c, int ldc) {
      // std::cout << "CLBlas::DoBlasGemm()" << std::endl;
  VLOG(1) << port::Printf(
      "doing CLBlast SGEMM: at=%d bt=%d m=%llu n=%llu "
      "k=%llu alpha=%f a=%p lda=%d b=%p ldb=%d beta=%f "
      "c=%p ldc=%d",
      static_cast<int>(transa), static_cast<int>(transb), m, n, k, alpha,
      a.opaque(), lda, b.opaque(), ldb, beta, c->opaque(), ldc);
  if (transa == blas::Transpose::kNoTranspose) {
    if (lda < static_cast<int64>(m)) {
      LOG(WARNING) << "GEMM lda was smaller than m (no transpose case); "
                      "precondition violation";
    }
  } else {
    if (lda < static_cast<int64>(k)) {
      LOG(WARNING) << "GEMM lda (" << lda << ") was smaller than k (" << k
                   << ") (transpose case); precondition violation";
    }
  }
  if (transb == blas::Transpose::kNoTranspose) {
    if (ldb < static_cast<int64>(k)) {
      LOG(WARNING) << "GEMM ldb (" << ldb << ") was smaller than k (" << k
                   << ") (no transpose case); precondition violation";
    }
  } else {
    if (ldb < static_cast<int64>(n)) {
      LOG(WARNING) << "GEMM ldb was smaller than n (transpose case); "
                      "precondition violation";
    }
  }
  return DoBlasInternal(
      cublasSgemm, stream, true /* = pointer_mode_host */,
      CLBlasTranspose(transa), CLBlasTranspose(transb), m, n, k, &alpha,
      CUDAMemory(a), lda, CUDAMemory(b), ldb, &beta, CUDAMemoryMutable(c), ldc);
}

bool CLBlas::DoBlasAsum(Stream *stream, uint64 elem_count,
                          const DeviceMemory<float> &x, int incx,
                          DeviceMemory<float> *result) {
  // cout << "CLBlas::DoBlasAsum" << endl;
  return false;
  // return DoBlasInternal(cublasSasum, stream,
  //                       false  = pointer_mode_host , elem_count,
  //                       CLMemory(x), incx, CLMemoryMutable(result));
}

bool CLBlas::DoBlasAsum(Stream *stream, uint64 elem_count,
                          const DeviceMemory<double> &x, int incx,
                          DeviceMemory<double> *result) {
  return false;
  // return DoBlasInternal(cublasDasum, stream,
  //                       false  = pointer_mode_host , elem_count,
  //                       CLMemory(x), incx, CLMemoryMutable(result));
}

bool CLBlas::DoBlasAsum(Stream *stream, uint64 elem_count,
                          const DeviceMemory<std::complex<float>> &x, int incx,
                          DeviceMemory<float> *result) {
  return false;
  // return DoBlasInternal(
  //     cublasScasum, stream, false  = pointer_mode_host ,
  //     elem_count, CLComplex(CLMemory(x)), incx, CLMemoryMutable(result));
}

bool CLBlas::DoBlasAsum(Stream *stream, uint64 elem_count,
                          const DeviceMemory<std::complex<double>> &x, int incx,
                          DeviceMemory<double> *result) {
  return false;
  // return DoBlasInternal(
  //     cublasDzasum, stream, false  = pointer_mode_host ,
  //     elem_count, CLComplex(CLMemory(x)), incx, CLMemoryMutable(result));
}

bool CLBlas::DoBlasAxpy(Stream *stream, uint64 elem_count, float alpha,
                          const DeviceMemory<float> &x, int incx,
                          DeviceMemory<float> *y, int incy) {
  return false;
  // return DoBlasInternal(cublasSaxpy, stream,
  //                       true  = pointer_mode_host , elem_count, &alpha,
  //                       CLMemory(x), incx, CLMemoryMutable(y), incy);
}

bool CLBlas::DoBlasAxpy(Stream *stream, uint64 elem_count, double alpha,
                          const DeviceMemory<double> &x, int incx,
                          DeviceMemory<double> *y, int incy) {
  return false;
  // return DoBlasInternal(cublasDaxpy, stream,
  //                       true  = pointer_mode_host , elem_count, &alpha,
  //                       CLMemory(x), incx, CLMemoryMutable(y), incy);
}

bool CLBlas::DoBlasAxpy(Stream *stream, uint64 elem_count,
                          std::complex<float> alpha,
                          const DeviceMemory<std::complex<float>> &x, int incx,
                          DeviceMemory<std::complex<float>> *y, int incy) {
  return false;
  // return DoBlasInternal(cublasCaxpy, stream,
  //                       true /* = pointer_mode_host */, elem_count,
  //                       CLComplex(&alpha), CLComplex(CLMemory(x)), incx,
  //                       CLComplex(CLMemoryMutable(y)), incy);
}

bool CLBlas::DoBlasAxpy(Stream *stream, uint64 elem_count,
                          std::complex<double> alpha,
                          const DeviceMemory<std::complex<double>> &x, int incx,
                          DeviceMemory<std::complex<double>> *y, int incy) {
  return false;
  // return DoBlasInternal(cublasZaxpy, stream,
  //                       true /* = pointer_mode_host */, elem_count,
  //                       CLComplex(&alpha), CLComplex(CLMemory(x)), incx,
  //                       CLComplex(CLMemoryMutable(y)), incy);
}

bool CLBlas::DoBlasCopy(Stream *stream, uint64 elem_count,
                          const DeviceMemory<float> &x, int incx,
                          DeviceMemory<float> *y, int incy) {
      // std::cout << "CLBlas::DoBlasCopy()" << std::endl;
  return false;
  // return DoBlasInternal(cublasScopy, stream,
  //                       true  = pointer_mode_host , elem_count,
  //                       CLMemory(x), incx, CLMemoryMutable(y), incy);
}

bool CLBlas::DoBlasCopy(Stream *stream, uint64 elem_count,
                          const DeviceMemory<double> &x, int incx,
                          DeviceMemory<double> *y, int incy) {
  // cout << "CLBlas::DoBlasCopy" << endl;
  return false;
  // return DoBlasInternal(cublasDcopy, stream,
  //                       true  = pointer_mode_host , elem_count,
  //                       CLMemory(x), incx, CLMemoryMutable(y), incy);
}

bool CLBlas::DoBlasCopy(Stream *stream, uint64 elem_count,
                          const DeviceMemory<std::complex<float>> &x, int incx,
                          DeviceMemory<std::complex<float>> *y, int incy) {
  // cout << "CLBlas::DoBlasCopy" << endl;
  return false;
  // return DoBlasInternal(cublasCcopy, stream,
  //                       true /* = pointer_mode_host */, elem_count,
  //                       CLComplex(CLMemory(x)), incx,
  //                       CLComplex(CLMemoryMutable(y)), incy);
}

bool CLBlas::DoBlasCopy(Stream *stream, uint64 elem_count,
                          const DeviceMemory<std::complex<double>> &x, int incx,
                          DeviceMemory<std::complex<double>> *y, int incy) {
  // cout << "CLBlas::DoBlasCopy" << endl;
  return false;
  // return DoBlasInternal(cublasZcopy, stream,
  //                       true /* = pointer_mode_host */, elem_count,
  //                       CLComplex(CLMemory(x)), incx,
  //                       CLComplex(CLMemoryMutable(y)), incy);
}

bool CLBlas::DoBlasDot(Stream *stream, uint64 elem_count,
                         const DeviceMemory<float> &x, int incx,
                         const DeviceMemory<float> &y, int incy,
                         DeviceMemory<float> *result) {
  return false;
  // return DoBlasInternal(
  //     cublasSdot, stream, false /* = pointer_mode_host */, elem_count,
  //     CLMemory(x), incx, CLMemory(y), incy, CLMemoryMutable(result));
}

bool CLBlas::DoBlasDot(Stream *stream, uint64 elem_count,
                         const DeviceMemory<double> &x, int incx,
                         const DeviceMemory<double> &y, int incy,
                         DeviceMemory<double> *result) {
  return false;
  // return DoBlasInternal(
  //     cublasDdot, stream, false /* = pointer_mode_host */, elem_count,
  //     CLMemory(x), incx, CLMemory(y), incy, CLMemoryMutable(result));
}

bool CLBlas::DoBlasDotc(Stream *stream, uint64 elem_count,
                          const DeviceMemory<std::complex<float>> &x, int incx,
                          const DeviceMemory<std::complex<float>> &y, int incy,
                          DeviceMemory<std::complex<float>> *result) {
  return false;
  // return DoBlasInternal(
  //     cublasCdotc, stream, false /* = pointer_mode_host */, elem_count,
  //     CLComplex(CLMemory(x)), incx, CLComplex(CLMemory(y)), incy,
  //     CLComplex(CLMemoryMutable(result)));
}

bool CLBlas::DoBlasDotc(Stream *stream, uint64 elem_count,
                          const DeviceMemory<std::complex<double>> &x, int incx,
                          const DeviceMemory<std::complex<double>> &y, int incy,
                          DeviceMemory<std::complex<double>> *result) {
  return false;
  // return DoBlasInternal(
  //     cublasZdotc, stream, false /* = pointer_mode_host */, elem_count,
  //     CLComplex(CLMemory(x)), incx, CLComplex(CLMemory(y)), incy,
  //     CLComplex(CLMemoryMutable(result)));
}

bool CLBlas::DoBlasDotu(Stream *stream, uint64 elem_count,
                          const DeviceMemory<std::complex<float>> &x, int incx,
                          const DeviceMemory<std::complex<float>> &y, int incy,
                          DeviceMemory<std::complex<float>> *result) {
  return false;
  // return DoBlasInternal(
  //     cublasCdotu, stream, false /* = pointer_mode_host */, elem_count,
  //     CLComplex(CLMemory(x)), incx, CLComplex(CLMemory(y)), incy,
  //     CLComplex(CLMemoryMutable(result)));
}

bool CLBlas::DoBlasDotu(Stream *stream, uint64 elem_count,
                          const DeviceMemory<std::complex<double>> &x, int incx,
                          const DeviceMemory<std::complex<double>> &y, int incy,
                          DeviceMemory<std::complex<double>> *result) {
  return false;
  // return DoBlasInternal(
  //     cublasZdotu, stream, false /* = pointer_mode_host */, elem_count,
  //     CLComplex(CLMemory(x)), incx, CLComplex(CLMemory(y)), incy,
  //     CLComplex(CLMemoryMutable(result)));
}

bool CLBlas::DoBlasNrm2(Stream *stream, uint64 elem_count,
                          const DeviceMemory<float> &x, int incx,
                          DeviceMemory<float> *result) {
  return false;
  // return DoBlasInternal(cublasSnrm2, stream,
  //                       false  = pointer_mode_host , elem_count,
  //                       CLMemory(x), incx, CLMemoryMutable(result));
}

bool CLBlas::DoBlasNrm2(Stream *stream, uint64 elem_count,
                          const DeviceMemory<double> &x, int incx,
                          DeviceMemory<double> *result) {
  return false;
  // return DoBlasInternal(cublasDnrm2, stream,
  //                       false  = pointer_mode_host , elem_count,
  //                       CLMemory(x), incx, CLMemoryMutable(result));
}

bool CLBlas::DoBlasNrm2(Stream *stream, uint64 elem_count,
                          const DeviceMemory<std::complex<float>> &x, int incx,
                          DeviceMemory<float> *result) {
  return false;
  // return DoBlasInternal(
  //     cublasScnrm2, stream, false  = pointer_mode_host ,
  //     elem_count, CLComplex(CLMemory(x)), incx, CLMemoryMutable(result));
}

bool CLBlas::DoBlasNrm2(Stream *stream, uint64 elem_count,
                          const DeviceMemory<std::complex<double>> &x, int incx,
                          DeviceMemory<double> *result) {
  return false;
  // return DoBlasInternal(
  //     cublasDznrm2, stream, false  = pointer_mode_host ,
  //     elem_count, CLComplex(CLMemory(x)), incx, CLMemoryMutable(result));
}

bool CLBlas::DoBlasRot(Stream *stream, uint64 elem_count,
                         DeviceMemory<float> *x, int incx,
                         DeviceMemory<float> *y, int incy, float c, float s) {
  return false;
  // return DoBlasInternal(
  //     cublasSrot, stream, true /* = pointer_mode_host */, elem_count,
  //     CLMemoryMutable(x), incx, CLMemoryMutable(y), incy, &c, &s);
}

bool CLBlas::DoBlasRot(Stream *stream, uint64 elem_count,
                         DeviceMemory<double> *x, int incx,
                         DeviceMemory<double> *y, int incy, double c,
                         double s) {
  return false;
  // return DoBlasInternal(
  //     cublasDrot, stream, true /* = pointer_mode_host */, elem_count,
  //     CLMemoryMutable(x), incx, CLMemoryMutable(y), incy, &c, &s);
}

bool CLBlas::DoBlasRot(Stream *stream, uint64 elem_count,
                         DeviceMemory<std::complex<float>> *x, int incx,
                         DeviceMemory<std::complex<float>> *y, int incy,
                         float c, float s) {
  return false;
  // return DoBlasInternal(cublasCsrot, stream,
  //                       true /* = pointer_mode_host */, elem_count,
  //                       CLComplex(CLMemoryMutable(x)), incx,
  //                       CLComplex(CLMemoryMutable(y)), incy, &c, &s);
}

bool CLBlas::DoBlasRot(Stream *stream, uint64 elem_count,
                         DeviceMemory<std::complex<double>> *x, int incx,
                         DeviceMemory<std::complex<double>> *y, int incy,
                         double c, double s) {
  return false;
  // return DoBlasInternal(cublasZdrot, stream,
  //                       true /* = pointer_mode_host */, elem_count,
  //                       CLComplex(CLMemoryMutable(x)), incx,
  //                       CLComplex(CLMemoryMutable(y)), incy, &c, &s);
}

bool CLBlas::DoBlasRotg(Stream *stream, DeviceMemory<float> *a,
                          DeviceMemory<float> *b, DeviceMemory<float> *c,
                          DeviceMemory<float> *s) {
  return false;
  // return DoBlasInternal(cublasSrotg, stream,
  //                       false /* = pointer_mode_host */, CLMemoryMutable(a),
  //                       CLMemoryMutable(b), CLMemoryMutable(c),
  //                       CLMemoryMutable(s));
}

bool CLBlas::DoBlasRotg(Stream *stream, DeviceMemory<double> *a,
                          DeviceMemory<double> *b, DeviceMemory<double> *c,
                          DeviceMemory<double> *s) {
  return false;
  // return DoBlasInternal(cublasDrotg, stream,
  //                       false /* = pointer_mode_host */,
  //                       CLComplex(CLMemoryMutable(a)), CLMemoryMutable(b),
  //                       CLMemoryMutable(c), CLMemoryMutable(s));
}

bool CLBlas::DoBlasRotg(Stream *stream, DeviceMemory<std::complex<float>> *a,
                          DeviceMemory<std::complex<float>> *b,
                          DeviceMemory<float> *c,
                          DeviceMemory<std::complex<float>> *s) {
  return false;
  // return DoBlasInternal(
  //     cublasCrotg, stream, false /* = pointer_mode_host */,
  //     CLComplex(CLMemoryMutable(a)), CLComplex(CLMemoryMutable(b)),
  //     CLComplex(CLMemoryMutable(c)), CLComplex(CLMemoryMutable(s)));
}

bool CLBlas::DoBlasRotg(Stream *stream, DeviceMemory<std::complex<double>> *a,
                          DeviceMemory<std::complex<double>> *b,
                          DeviceMemory<double> *c,
                          DeviceMemory<std::complex<double>> *s) {
  return false;
  // return DoBlasInternal(
  //     cublasZrotg, stream, false /* = pointer_mode_host */,
  //     CLComplex(CLMemoryMutable(a)), CLComplex(CLMemoryMutable(b)),
  //     CLComplex(CLMemoryMutable(c)), CLComplex(CLMemoryMutable(s)));
}

bool CLBlas::DoBlasRotm(Stream *stream, uint64 elem_count,
                          DeviceMemory<float> *x, int incx,
                          DeviceMemory<float> *y, int incy,
                          const DeviceMemory<float> &param) {
  return false;
  // return DoBlasInternal(cublasSrotm, stream,
  //                       false /* = pointer_mode_host */, elem_count,
  //                       CLMemoryMutable(x), incx, CLMemoryMutable(y), incy,
  //                       CLMemory(param));
}

bool CLBlas::DoBlasRotm(Stream *stream, uint64 elem_count,
                          DeviceMemory<double> *x, int incx,
                          DeviceMemory<double> *y, int incy,
                          const DeviceMemory<double> &param) {
  return false;
  // return DoBlasInternal(cublasDrotm, stream,
  //                       false /* = pointer_mode_host */, elem_count,
  //                       CLMemoryMutable(x), incx, CLMemoryMutable(y), incy,
  //                       CLMemory(param));
}

bool CLBlas::DoBlasRotmg(Stream *stream, DeviceMemory<float> *d1,
                           DeviceMemory<float> *d2, DeviceMemory<float> *x1,
                           const DeviceMemory<float> &y1,
                           DeviceMemory<float> *param) {
  return false;
  // return DoBlasInternal(cublasSrotmg, stream,
  //                       false /* = pointer_mode_host */, CLMemoryMutable(d1),
  //                       CLMemoryMutable(d2), CLMemoryMutable(x1),
  //                       CLMemory(y1), CLMemoryMutable(param));
}

bool CLBlas::DoBlasRotmg(Stream *stream, DeviceMemory<double> *d1,
                           DeviceMemory<double> *d2, DeviceMemory<double> *x1,
                           const DeviceMemory<double> &y1,
                           DeviceMemory<double> *param) {
  return false;
  // return DoBlasInternal(cublasDrotmg, stream,
  //                       false /* = pointer_mode_host */, CLMemoryMutable(d1),
  //                       CLMemoryMutable(d2), CLMemoryMutable(x1),
  //                       CLMemory(y1), CLMemoryMutable(param));
}

bool CLBlas::DoBlasScal(Stream *stream, uint64 elem_count, float alpha,
                          DeviceMemory<float> *x, int incx) {
  return false;
  // return DoBlasInternal(cublasSscal, stream,
  //                       true  = pointer_mode_host , elem_count, &alpha,
  //                       CUDAMemoryMutable(x), incx);
}

bool CLBlas::DoBlasScal(Stream *stream, uint64 elem_count, double alpha,
                          DeviceMemory<double> *x, int incx) {
  return false;
  // return DoBlasInternal(cublasDscal, stream,
  //                       true  = pointer_mode_host , elem_count, &alpha,
  //                       CLMemoryMutable(x), incx);
}

bool CLBlas::DoBlasScal(Stream *stream, uint64 elem_count, float alpha,
                          DeviceMemory<std::complex<float>> *x, int incx) {
  return false;
  // return DoBlasInternal(
  //     cublasCsscal, stream, true  = pointer_mode_host , elem_count,
  //     CLComplex(&alpha), CLComplex(CLMemoryMutable(x)), incx);
}

bool CLBlas::DoBlasScal(Stream *stream, uint64 elem_count, double alpha,
                          DeviceMemory<std::complex<double>> *x, int incx) {
  return false;
  // return DoBlasInternal(
  //     cublasZdscal, stream, true  = pointer_mode_host , elem_count,
  //     CLComplex(&alpha), CLComplex(CLMemoryMutable(x)), incx);
}

bool CLBlas::DoBlasScal(Stream *stream, uint64 elem_count,
                          std::complex<float> alpha,
                          DeviceMemory<std::complex<float>> *x, int incx) {
  return false;
  // return DoBlasInternal(
  //     cublasCscal, stream, true  = pointer_mode_host , elem_count,
  //     CLComplex(&alpha), CLComplex(CLMemoryMutable(x)), incx);
}

bool CLBlas::DoBlasScal(Stream *stream, uint64 elem_count,
                          std::complex<double> alpha,
                          DeviceMemory<std::complex<double>> *x, int incx) {
  return false;
  // return DoBlasInternal(
  //     cublasZscal, stream, true  = pointer_mode_host , elem_count,
  //     CLComplex(&alpha), CLComplex(CLMemoryMutable(x)), incx);
}

bool CLBlas::DoBlasSwap(Stream *stream, uint64 elem_count,
                          DeviceMemory<float> *x, int incx,
                          DeviceMemory<float> *y, int incy) {
  // std::cout << "CLBlas::DoBlasSwap" << endl;
  return false;
  // return DoBlasInternal(cublasSswap, stream,
  //                       true  = pointer_mode_host , elem_count,
  //                       CLMemoryMutable(x), incx, CLMemoryMutable(y), incy);
}

bool CLBlas::DoBlasSwap(Stream *stream, uint64 elem_count,
                          DeviceMemory<double> *x, int incx,
                          DeviceMemory<double> *y, int incy) {
  // cout << "CLBlas::DoBlasSwap" << endl;
  return false;
  // return DoBlasInternal(cublasDswap, stream,
  //                       true  = pointer_mode_host , elem_count,
  //                       CLMemoryMutable(x), incx, CLMemoryMutable(y), incy);
}

bool CLBlas::DoBlasSwap(Stream *stream, uint64 elem_count,
                          DeviceMemory<std::complex<float>> *x, int incx,
                          DeviceMemory<std::complex<float>> *y, int incy) {
  // cout << "CLBlas::DoBlasSwap" << endl;
  return false;
  // return DoBlasInternal(cublasCswap, stream,
  //                       true /* = pointer_mode_host */, elem_count,
  //                       CLComplex(CLMemoryMutable(x)), incx,
  //                       CLComplex(CLMemoryMutable(y)), incy);
}

bool CLBlas::DoBlasSwap(Stream *stream, uint64 elem_count,
                          DeviceMemory<std::complex<double>> *x, int incx,
                          DeviceMemory<std::complex<double>> *y, int incy) {
  // cout << "CLBlas::DoBlasSwap" << endl;
  return false;
  // return DoBlasInternal(cublasZswap, stream,
  //                       true /* = pointer_mode_host */, elem_count,
  //                       CLComplex(CLMemoryMutable(x)), incx,
  //                       CLComplex(CLMemoryMutable(y)), incy);
}

bool CLBlas::DoBlasIamax(Stream *stream, uint64 elem_count,
                           const DeviceMemory<float> &x, int incx,
                           DeviceMemory<int> *result) {
  return false;
  // return DoBlasInternal(cublasIsamax, stream,
  //                       false  = pointer_mode_host , elem_count,
  //                       CLMemory(x), incx, CLMemoryMutable(result));
}

bool CLBlas::DoBlasIamax(Stream *stream, uint64 elem_count,
                           const DeviceMemory<double> &x, int incx,
                           DeviceMemory<int> *result) {
  return false;
  // return DoBlasInternal(cublasIdamax, stream,
  //                       false  = pointer_mode_host , elem_count,
  //                       CLMemory(x), incx, CLMemoryMutable(result));
}

bool CLBlas::DoBlasIamax(Stream *stream, uint64 elem_count,
                           const DeviceMemory<std::complex<float>> &x, int incx,
                           DeviceMemory<int> *result) {
  return false;
  // return DoBlasInternal(
  //     cublasIcamax, stream, false  = pointer_mode_host ,
  //     elem_count, CLComplex(CLMemory(x)), incx, CLMemoryMutable(result));
}

bool CLBlas::DoBlasIamax(Stream *stream, uint64 elem_count,
                           const DeviceMemory<std::complex<double>> &x,
                           int incx, DeviceMemory<int> *result) {
  return false;
  // return DoBlasInternal(
  //     cublasIzamax, stream, false  = pointer_mode_host ,
  //     elem_count, CLComplex(CLMemory(x)), incx, CLMemoryMutable(result));
}

bool CLBlas::DoBlasIamin(Stream *stream, uint64 elem_count,
                           const DeviceMemory<float> &x, int incx,
                           DeviceMemory<int> *result) {
  return false;
  // return DoBlasInternal(
  //     cublasIsamin, stream, false  = pointer_mode_host ,
  //     elem_count, CLComplex(CLMemory(x)), incx, CLMemoryMutable(result));
}

bool CLBlas::DoBlasIamin(Stream *stream, uint64 elem_count,
                           const DeviceMemory<double> &x, int incx,
                           DeviceMemory<int> *result) {
  return false;
  // return DoBlasInternal(
  //     cublasIdamin, stream, false  = pointer_mode_host ,
  //     elem_count, CLComplex(CLMemory(x)), incx, CLMemoryMutable(result));
}

bool CLBlas::DoBlasIamin(Stream *stream, uint64 elem_count,
                           const DeviceMemory<std::complex<float>> &x, int incx,
                           DeviceMemory<int> *result) {
  return false;
  // return DoBlasInternal(
  //     cublasIcamin, stream, false  = pointer_mode_host ,
  //     elem_count, CLComplex(CLMemory(x)), incx, CLMemoryMutable(result));
}

bool CLBlas::DoBlasIamin(Stream *stream, uint64 elem_count,
                           const DeviceMemory<std::complex<double>> &x,
                           int incx, DeviceMemory<int> *result) {
  return false;
  // return DoBlasInternal(
  //     cublasIzamin, stream, false  = pointer_mode_host ,
  //     elem_count, CLComplex(CLMemory(x)), incx, CLMemoryMutable(result));
}

bool CLBlas::DoBlasGbmv(Stream *stream, blas::Transpose trans, uint64 m,
                          uint64 n, uint64 kl, uint64 ku, float alpha,
                          const DeviceMemory<float> &a, int lda,
                          const DeviceMemory<float> &x, int incx, float beta,
                          DeviceMemory<float> *y, int incy) {
  return false;
  // return DoBlasInternal(
  //     cublasSgbmv, stream, true /* = pointer_mode_host */,
  //     CLBlasTranspose(trans), m, n, kl, ku, &alpha, CLMemory(a), lda,
  //     CLMemory(x), incx, &beta, CLMemoryMutable(y), incy);
}

bool CLBlas::DoBlasGbmv(Stream *stream, blas::Transpose trans, uint64 m,
                          uint64 n, uint64 kl, uint64 ku, double alpha,
                          const DeviceMemory<double> &a, int lda,
                          const DeviceMemory<double> &x, int incx, double beta,
                          DeviceMemory<double> *y, int incy) {
  return false;
  // return DoBlasInternal(
  //     cublasDgbmv, stream, true /* = pointer_mode_host */,
  //     CLBlasTranspose(trans), m, n, kl, ku, &alpha, CLMemory(a), lda,
  //     CLMemory(x), incx, &beta, CLMemoryMutable(y), incy);
}

bool CLBlas::DoBlasGbmv(Stream *stream, blas::Transpose trans, uint64 m,
                          uint64 n, uint64 kl, uint64 ku,
                          std::complex<float> alpha,
                          const DeviceMemory<std::complex<float>> &a, int lda,
                          const DeviceMemory<std::complex<float>> &x, int incx,
                          std::complex<float> beta,
                          DeviceMemory<std::complex<float>> *y, int incy) {
  return false;
  // return DoBlasInternal(
  //     cublasCgbmv, stream, true /* = pointer_mode_host */,
  //     CLBlasTranspose(trans), m, n, kl, ku, CLComplex(&alpha),
  //     CLComplex(CLMemory(a)), lda, CLComplex(CLMemory(x)), incx,
  //     CLComplex(&beta), CLComplex(CLMemoryMutable(y)), incy);
}

bool CLBlas::DoBlasGbmv(Stream *stream, blas::Transpose trans, uint64 m,
                          uint64 n, uint64 kl, uint64 ku,
                          std::complex<double> alpha,
                          const DeviceMemory<std::complex<double>> &a, int lda,
                          const DeviceMemory<std::complex<double>> &x, int incx,
                          std::complex<double> beta,
                          DeviceMemory<std::complex<double>> *y, int incy) {
  return false;
  // return DoBlasInternal(
  //     cublasZgbmv, stream, true /* = pointer_mode_host */,
  //     CLBlasTranspose(trans), m, n, kl, ku, CLComplex(&alpha),
  //     CLComplex(CLMemory(a)), lda, CLComplex(CLMemory(x)), incx,
  //     CLComplex(&beta), CLComplex(CLMemoryMutable(y)), incy);
}

bool CLBlas::DoBlasGemv(Stream *stream, blas::Transpose trans, uint64 m,
                          uint64 n, float alpha, const DeviceMemory<float> &a,
                          int lda, const DeviceMemory<float> &x, int incx,
                          float beta, DeviceMemory<float> *y, int incy) {
  return false;
  // return DoBlasInternal(
  //     cublasSgemv, stream, true /* = pointer_mode_host */,
  //     CLBlasTranspose(trans), m, n, &alpha, CLMemory(a), lda, CLMemory(x),
  //     incx, &beta, CLMemoryMutable(y), incy);
}

bool CLBlas::DoBlasGemv(Stream *stream, blas::Transpose trans, uint64 m,
                          uint64 n, double alpha, const DeviceMemory<double> &a,
                          int lda, const DeviceMemory<double> &x, int incx,
                          double beta, DeviceMemory<double> *y, int incy) {
  return false;
  // return DoBlasInternal(
  //     cublasDgemv, stream, true /* = pointer_mode_host */,
  //     CLBlasTranspose(trans), m, n, &alpha, CLMemory(a), lda, CLMemory(x),
  //     incx, &beta, CLMemoryMutable(y), incy);
}

bool CLBlas::DoBlasGemv(Stream *stream, blas::Transpose trans, uint64 m,
                          uint64 n, std::complex<float> alpha,
                          const DeviceMemory<std::complex<float>> &a, int lda,
                          const DeviceMemory<std::complex<float>> &x, int incx,
                          std::complex<float> beta,
                          DeviceMemory<std::complex<float>> *y, int incy) {
  return false;
  // return DoBlasInternal(
  //     cublasCgemv, stream, true /* = pointer_mode_host */,
  //     CLBlasTranspose(trans), m, n, CLComplex(&alpha),
  //     CLComplex(CLMemory(a)), lda, CLComplex(CLMemory(x)), incx,
  //     CLComplex(&beta), CLComplex(CLMemoryMutable(y)), incy);
}

bool CLBlas::DoBlasGemv(Stream *stream, blas::Transpose trans, uint64 m,
                          uint64 n, std::complex<double> alpha,
                          const DeviceMemory<std::complex<double>> &a, int lda,
                          const DeviceMemory<std::complex<double>> &x, int incx,
                          std::complex<double> beta,
                          DeviceMemory<std::complex<double>> *y, int incy) {
  return false;
  // return DoBlasInternal(
  //     cublasZgemv, stream, true /* = pointer_mode_host */,
  //     CLBlasTranspose(trans), m, n, CLComplex(&alpha),
  //     CLComplex(CLMemory(a)), lda, CLComplex(CLMemory(x)), incx,
  //     CLComplex(&beta), CLComplex(CLMemoryMutable(y)), incy);
}

bool CLBlas::DoBlasGer(Stream *stream, uint64 m, uint64 n, float alpha,
                         const DeviceMemory<float> &x, int incx,
                         const DeviceMemory<float> &y, int incy,
                         DeviceMemory<float> *a, int lda) {
  return false;
  // return DoBlasInternal(
  //     cublasSger, stream, true /* = pointer_mode_host */, m, n, &alpha,
  //     CLMemory(x), incx, CLMemory(y), incy, CLMemoryMutable(a), lda);
}

bool CLBlas::DoBlasGer(Stream *stream, uint64 m, uint64 n, double alpha,
                         const DeviceMemory<double> &x, int incx,
                         const DeviceMemory<double> &y, int incy,
                         DeviceMemory<double> *a, int lda) {
  return false;
  // return DoBlasInternal(
  //     cublasDger, stream, true /* = pointer_mode_host */, m, n, &alpha,
  //     CLMemory(x), incx, CLMemory(y), incy, CLMemoryMutable(a), lda);
}

bool CLBlas::DoBlasGerc(Stream *stream, uint64 m, uint64 n,
                          std::complex<float> alpha,
                          const DeviceMemory<std::complex<float>> &x, int incx,
                          const DeviceMemory<std::complex<float>> &y, int incy,
                          DeviceMemory<std::complex<float>> *a, int lda) {
  return false;
  // return DoBlasInternal(
  //     cublasCgerc, stream, true /* = pointer_mode_host */, m, n,
  //     CLComplex(&alpha), CLComplex(CLMemory(x)), incx,
  //     CLComplex(CLMemory(y)), incy, CLComplex(CLMemoryMutable(a)), lda);
}

bool CLBlas::DoBlasGerc(Stream *stream, uint64 m, uint64 n,
                          std::complex<double> alpha,
                          const DeviceMemory<std::complex<double>> &x, int incx,
                          const DeviceMemory<std::complex<double>> &y, int incy,
                          DeviceMemory<std::complex<double>> *a, int lda) {
  return false;
  // return DoBlasInternal(
  //     cublasZgerc, stream, true /* = pointer_mode_host */, m, n,
  //     CLComplex(&alpha), CLComplex(CLMemory(x)), incx,
  //     CLComplex(CLMemory(y)), incy, CLComplex(CLMemoryMutable(a)), lda);
}

bool CLBlas::DoBlasGeru(Stream *stream, uint64 m, uint64 n,
                          std::complex<float> alpha,
                          const DeviceMemory<std::complex<float>> &x, int incx,
                          const DeviceMemory<std::complex<float>> &y, int incy,
                          DeviceMemory<std::complex<float>> *a, int lda) {
  return false;
  // return DoBlasInternal(
  //     cublasCgeru, stream, true /* = pointer_mode_host */, m, n,
  //     CLComplex(&alpha), CLComplex(CLMemory(x)), incx,
  //     CLComplex(CLMemory(y)), incy, CLComplex(CLMemoryMutable(a)), lda);
}

bool CLBlas::DoBlasGeru(Stream *stream, uint64 m, uint64 n,
                          std::complex<double> alpha,
                          const DeviceMemory<std::complex<double>> &x, int incx,
                          const DeviceMemory<std::complex<double>> &y, int incy,
                          DeviceMemory<std::complex<double>> *a, int lda) {
  return false;
  // return DoBlasInternal(
  //     cublasZgeru, stream, true /* = pointer_mode_host */, m, n,
  //     CLComplex(&alpha), CLComplex(CLMemory(x)), incx,
  //     CLComplex(CLMemory(y)), incy, CLComplex(CLMemoryMutable(a)), lda);
}

bool CLBlas::DoBlasHbmv(Stream *stream, blas::UpperLower uplo, uint64 n,
                          uint64 k, std::complex<float> alpha,
                          const DeviceMemory<std::complex<float>> &a, int lda,
                          const DeviceMemory<std::complex<float>> &x, int incx,
                          std::complex<float> beta,
                          DeviceMemory<std::complex<float>> *y, int incy) {
  return false;
  // return DoBlasInternal(
  //     cublasChbmv, stream, true /* = pointer_mode_host */,
  //     CLBlasUpperLower(uplo), n, k, CLComplex(&alpha),
  //     CLComplex(CLMemory(a)), lda, CLComplex(CLMemory(x)), incx,
  //     CLComplex(&beta), CLComplex(CLMemoryMutable(y)), incy);
}

bool CLBlas::DoBlasHbmv(Stream *stream, blas::UpperLower uplo, uint64 n,
                          uint64 k, std::complex<double> alpha,
                          const DeviceMemory<std::complex<double>> &a, int lda,
                          const DeviceMemory<std::complex<double>> &x, int incx,
                          std::complex<double> beta,
                          DeviceMemory<std::complex<double>> *y, int incy) {
  return false;
  // return DoBlasInternal(
  //     cublasZhbmv, stream, true /* = pointer_mode_host */,
  //     CLBlasUpperLower(uplo), n, k, CLComplex(&alpha),
  //     CLComplex(CLMemory(a)), lda, CLComplex(CLMemory(x)), incx,
  //     CLComplex(&beta), CLComplex(CLMemoryMutable(y)), incy);
}

bool CLBlas::DoBlasHemv(Stream *stream, blas::UpperLower uplo, uint64 n,
                          std::complex<float> alpha,
                          const DeviceMemory<std::complex<float>> &a, int lda,
                          const DeviceMemory<std::complex<float>> &x, int incx,
                          std::complex<float> beta,
                          DeviceMemory<std::complex<float>> *y, int incy) {
  return false;
  // return DoBlasInternal(
  //     cublasChemv, stream, true /* = pointer_mode_host */,
  //     CLBlasUpperLower(uplo), n, CLComplex(&alpha),
  //     CLComplex(CLMemory(a)), lda, CLComplex(CLMemory(x)), incx,
  //     CLComplex(&beta), CLComplex(CLMemoryMutable(y)), incy);
}

bool CLBlas::DoBlasHemv(Stream *stream, blas::UpperLower uplo, uint64 n,
                          std::complex<double> alpha,
                          const DeviceMemory<std::complex<double>> &a, int lda,
                          const DeviceMemory<std::complex<double>> &x, int incx,
                          std::complex<double> beta,
                          DeviceMemory<std::complex<double>> *y, int incy) {
  return false;
  // return DoBlasInternal(
  //     cublasZhemv, stream, true /* = pointer_mode_host */,
  //     CLBlasUpperLower(uplo), n, CLComplex(&alpha),
  //     CLComplex(CLMemory(a)), lda, CLComplex(CLMemory(x)), incx,
  //     CLComplex(&beta), CLComplex(CLMemoryMutable(y)), incy);
}

bool CLBlas::DoBlasHer(Stream *stream, blas::UpperLower uplo, uint64 n,
                         float alpha,
                         const DeviceMemory<std::complex<float>> &x, int incx,
                         DeviceMemory<std::complex<float>> *a, int lda) {
  return false;
  // return DoBlasInternal(
  //     cublasCher, stream, true /* = pointer_mode_host */,
  //     CLBlasUpperLower(uplo), n, &alpha, CLComplex(CLMemory(x)), incx,
  //     CLComplex(CLMemoryMutable(a)), lda);
}

bool CLBlas::DoBlasHer(Stream *stream, blas::UpperLower uplo, uint64 n,
                         double alpha,
                         const DeviceMemory<std::complex<double>> &x, int incx,
                         DeviceMemory<std::complex<double>> *a, int lda) {
  return false;
  // return DoBlasInternal(
  //     cublasZher, stream, true /* = pointer_mode_host */,
  //     CLBlasUpperLower(uplo), n, &alpha, CLComplex(CLMemory(x)), incx,
  //     CLComplex(CLMemoryMutable(a)), lda);
}

bool CLBlas::DoBlasHer2(Stream *stream, blas::UpperLower uplo, uint64 n,
                          std::complex<float> alpha,
                          const DeviceMemory<std::complex<float>> &x, int incx,
                          const DeviceMemory<std::complex<float>> &y, int incy,
                          DeviceMemory<std::complex<float>> *a, int lda) {
  return false;
  // return DoBlasInternal(
  //     cublasCher2, stream, true /* = pointer_mode_host */,
  //     CLBlasUpperLower(uplo), n, CLComplex(&alpha),
  //     CLComplex(CLMemory(x)), incx, CLComplex(CLMemory(y)), incy,
  //     CLComplex(CLMemoryMutable(a)), lda);
}

bool CLBlas::DoBlasHer2(Stream *stream, blas::UpperLower uplo, uint64 n,
                          std::complex<double> alpha,
                          const DeviceMemory<std::complex<double>> &x, int incx,
                          const DeviceMemory<std::complex<double>> &y, int incy,
                          DeviceMemory<std::complex<double>> *a, int lda) {
  return false;
  // return DoBlasInternal(
  //     cublasZher2, stream, true /* = pointer_mode_host */,
  //     CLBlasUpperLower(uplo), n, CLComplex(&alpha),
  //     CLComplex(CLMemory(x)), incx, CLComplex(CLMemory(y)), incy,
  //     CLComplex(CLMemoryMutable(a)), lda);
}

bool CLBlas::DoBlasHpmv(Stream *stream, blas::UpperLower uplo, uint64 n,
                          std::complex<float> alpha,
                          const DeviceMemory<std::complex<float>> &ap,
                          const DeviceMemory<std::complex<float>> &x, int incx,
                          std::complex<float> beta,
                          DeviceMemory<std::complex<float>> *y, int incy) {
  return false;
  // return DoBlasInternal(
  //     cublasChpmv, stream, true /* = pointer_mode_host */,
  //     CLBlasUpperLower(uplo), n, CLComplex(&alpha),
  //     CLComplex(CLMemory(ap)), CLComplex(CLMemory(x)), incx,
  //     CLComplex(&beta), CLComplex(CLMemoryMutable(y)), incy);
}

bool CLBlas::DoBlasHpmv(Stream *stream, blas::UpperLower uplo, uint64 n,
                          std::complex<double> alpha,
                          const DeviceMemory<std::complex<double>> &ap,
                          const DeviceMemory<std::complex<double>> &x, int incx,
                          std::complex<double> beta,
                          DeviceMemory<std::complex<double>> *y, int incy) {
  return false;
  // return DoBlasInternal(
  //     cublasZhpmv, stream, true /* = pointer_mode_host */,
  //     CLBlasUpperLower(uplo), n, CLComplex(&alpha),
  //     CLComplex(CLMemory(ap)), CLComplex(CLMemory(x)), incx,
  //     CLComplex(&beta), CLComplex(CLMemoryMutable(y)), incy);
}

bool CLBlas::DoBlasHpr(Stream *stream, blas::UpperLower uplo, uint64 n,
                         float alpha,
                         const DeviceMemory<std::complex<float>> &x, int incx,
                         DeviceMemory<std::complex<float>> *ap) {
  return false;
  // return DoBlasInternal(
  //     cublasChpr, stream, true /* = pointer_mode_host */,
  //     CLBlasUpperLower(uplo), n, CLComplex(&alpha),
  //     CLComplex(CLMemory(x)), incx, CLComplex(CLMemoryMutable(ap)));
}

bool CLBlas::DoBlasHpr(Stream *stream, blas::UpperLower uplo, uint64 n,
                         double alpha,
                         const DeviceMemory<std::complex<double>> &x, int incx,
                         DeviceMemory<std::complex<double>> *ap) {
  return false;
  // return DoBlasInternal(
  //     cublasZhpr, stream, true /* = pointer_mode_host */,
  //     CLBlasUpperLower(uplo), n, CLComplex(&alpha),
  //     CLComplex(CLMemory(x)), incx, CLComplex(CLMemoryMutable(ap)));
}

bool CLBlas::DoBlasHpr2(Stream *stream, blas::UpperLower uplo, uint64 n,
                          std::complex<float> alpha,
                          const DeviceMemory<std::complex<float>> &x, int incx,
                          const DeviceMemory<std::complex<float>> &y, int incy,
                          DeviceMemory<std::complex<float>> *ap) {
  return false;
  // return DoBlasInternal(
  //     cublasChpr2, stream, true /* = pointer_mode_host */,
  //     CLBlasUpperLower(uplo), n, CLComplex(&alpha),
  //     CLComplex(CLMemory(x)), incx, CLComplex(CLMemory(y)), incy,
  //     CLComplex(CLMemoryMutable(ap)));
}

bool CLBlas::DoBlasHpr2(Stream *stream, blas::UpperLower uplo, uint64 n,
                          std::complex<double> alpha,
                          const DeviceMemory<std::complex<double>> &x, int incx,
                          const DeviceMemory<std::complex<double>> &y, int incy,
                          DeviceMemory<std::complex<double>> *ap) {
  return false;
  // return DoBlasInternal(
  //     cublasZhpr2, stream, true /* = pointer_mode_host */,
  //     CLBlasUpperLower(uplo), n, CLComplex(&alpha),
  //     CLComplex(CLMemory(x)), incx, CLComplex(CLMemory(y)), incy,
  //     CLComplex(CLMemoryMutable(ap)));
}

bool CLBlas::DoBlasSbmv(Stream *stream, blas::UpperLower uplo, uint64 n,
                          uint64 k, float alpha, const DeviceMemory<float> &a,
                          int lda, const DeviceMemory<float> &x, int incx,
                          float beta, DeviceMemory<float> *y, int incy) {
  return false;
  // return DoBlasInternal(
  //     cublasSsbmv, stream, true /* = pointer_mode_host */,
  //     CLBlasUpperLower(uplo), n, k, &alpha, CLMemory(a), lda, CLMemory(x),
  //     incx, &beta, CLMemoryMutable(y), incy);
}

bool CLBlas::DoBlasSbmv(Stream *stream, blas::UpperLower uplo, uint64 n,
                          uint64 k, double alpha, const DeviceMemory<double> &a,
                          int lda, const DeviceMemory<double> &x, int incx,
                          double beta, DeviceMemory<double> *y, int incy) {
  return false;
  // return DoBlasInternal(
  //     cublasDsbmv, stream, true /* = pointer_mode_host */,
  //     CLBlasUpperLower(uplo), n, k, &alpha, CLMemory(a), lda, CLMemory(x),
  //     incx, &beta, CLMemoryMutable(y), incy);
}

bool CLBlas::DoBlasSpmv(Stream *stream, blas::UpperLower uplo, uint64 n,
                          float alpha, const DeviceMemory<float> &ap,
                          const DeviceMemory<float> &x, int incx, float beta,
                          DeviceMemory<float> *y, int incy) {
  return false;
  // return DoBlasInternal(cublasSspmv, stream,
  //                       true /* = pointer_mode_host */,
  //                       CLBlasUpperLower(uplo), n, &alpha, CLMemory(ap),
  //                       CLMemory(x), incx, &beta, CLMemoryMutable(y), incy);
}

bool CLBlas::DoBlasSpmv(Stream *stream, blas::UpperLower uplo, uint64 n,
                          double alpha, const DeviceMemory<double> &ap,
                          const DeviceMemory<double> &x, int incx, double beta,
                          DeviceMemory<double> *y, int incy) {
  return false;
  // return DoBlasInternal(cublasDspmv, stream,
  //                       true /* = pointer_mode_host */,
  //                       CLBlasUpperLower(uplo), n, &alpha, CLMemory(ap),
  //                       CLMemory(x), incx, &beta, CLMemoryMutable(y), incy);
}

bool CLBlas::DoBlasSpr(Stream *stream, blas::UpperLower uplo, uint64 n,
                         float alpha, const DeviceMemory<float> &x, int incx,
                         DeviceMemory<float> *ap) {
  return false;
  // return DoBlasInternal(cublasSspr, stream,
  //                       true /* = pointer_mode_host */,
  //                       CLBlasUpperLower(uplo), n, &alpha, CLMemory(x),
  //                       incx, CLMemoryMutable(ap));
}

bool CLBlas::DoBlasSpr(Stream *stream, blas::UpperLower uplo, uint64 n,
                         double alpha, const DeviceMemory<double> &x, int incx,
                         DeviceMemory<double> *ap) {
  return false;
  // return DoBlasInternal(cublasDspr, stream,
  //                       true /* = pointer_mode_host */,
  //                       CLBlasUpperLower(uplo), n, &alpha, CLMemory(x),
  //                       incx, CLMemoryMutable(ap));
}

bool CLBlas::DoBlasSpr2(Stream *stream, blas::UpperLower uplo, uint64 n,
                          float alpha, const DeviceMemory<float> &x, int incx,
                          const DeviceMemory<float> &y, int incy,
                          DeviceMemory<float> *ap) {
  return false;
  // return DoBlasInternal(cublasSspr2, stream,
  //                       true /* = pointer_mode_host */,
  //                       CLBlasUpperLower(uplo), n, &alpha, CLMemory(x),
  //                       incx, CLMemory(y), incy, CLMemoryMutable(ap));
}

bool CLBlas::DoBlasSpr2(Stream *stream, blas::UpperLower uplo, uint64 n,
                          double alpha, const DeviceMemory<double> &x, int incx,
                          const DeviceMemory<double> &y, int incy,
                          DeviceMemory<double> *ap) {
  return false;
  // return DoBlasInternal(cublasDspr2, stream,
  //                       true /* = pointer_mode_host */,
  //                       CLBlasUpperLower(uplo), n, &alpha, CLMemory(x),
  //                       incx, CLMemory(y), incy, CLMemoryMutable(ap));
}

bool CLBlas::DoBlasSymv(Stream *stream, blas::UpperLower uplo, uint64 n,
                          float alpha, const DeviceMemory<float> &a, int lda,
                          const DeviceMemory<float> &x, int incx, float beta,
                          DeviceMemory<float> *y, int incy) {
  return false;
  // return DoBlasInternal(cublasSsymv, stream,
  //                       true /* = pointer_mode_host */,
  //                       CLBlasUpperLower(uplo), n, &alpha, CLMemory(a), lda,
  //                       CLMemory(x), incx, &beta, CLMemoryMutable(y), incy);
}

bool CLBlas::DoBlasSymv(Stream *stream, blas::UpperLower uplo, uint64 n,
                          double alpha, const DeviceMemory<double> &a, int lda,
                          const DeviceMemory<double> &x, int incx, double beta,
                          DeviceMemory<double> *y, int incy) {
  return false;
  // return DoBlasInternal(cublasDsymv, stream,
  //                       true /* = pointer_mode_host */,
  //                       CLBlasUpperLower(uplo), n, &alpha, CLMemory(a), lda,
  //                       CLMemory(x), incx, &beta, CLMemoryMutable(y), incy);
}

bool CLBlas::DoBlasSyr(Stream *stream, blas::UpperLower uplo, uint64 n,
                         float alpha, const DeviceMemory<float> &x, int incx,
                         DeviceMemory<float> *a, int lda) {
  return false;
  // return DoBlasInternal(cublasSsyr, stream,
  //                       true /* = pointer_mode_host */,
  //                       CLBlasUpperLower(uplo), n, &alpha, CLMemory(x),
  //                       incx, CLMemoryMutable(a), lda);
}

bool CLBlas::DoBlasSyr(Stream *stream, blas::UpperLower uplo, uint64 n,
                         double alpha, const DeviceMemory<double> &x, int incx,
                         DeviceMemory<double> *a, int lda) {
  return false;
  // return DoBlasInternal(cublasDsyr, stream,
  //                       true /* = pointer_mode_host */,
  //                       CLBlasUpperLower(uplo), n, &alpha, CLMemory(x),
  //                       incx, CLMemoryMutable(a), lda);
}

bool CLBlas::DoBlasSyr2(Stream *stream, blas::UpperLower uplo, uint64 n,
                          float alpha, const DeviceMemory<float> &x, int incx,
                          const DeviceMemory<float> &y, int incy,
                          DeviceMemory<float> *a, int lda) {
  return false;
  // return DoBlasInternal(cublasSsyr2, stream,
  //                       true /* = pointer_mode_host */,
  //                       CLBlasUpperLower(uplo), n, &alpha, CLMemory(x),
  //                       incx, CLMemory(y), incy, CLMemoryMutable(a), lda);
}

bool CLBlas::DoBlasSyr2(Stream *stream, blas::UpperLower uplo, uint64 n,
                          double alpha, const DeviceMemory<double> &x, int incx,
                          const DeviceMemory<double> &y, int incy,
                          DeviceMemory<double> *a, int lda) {
  return false;
  // return DoBlasInternal(cublasDsyr2, stream,
  //                       true /* = pointer_mode_host */,
  //                       CLBlasUpperLower(uplo), n, &alpha, CLMemory(x),
  //                       incx, CLMemory(y), incy, CLMemoryMutable(a), lda);
}

bool CLBlas::DoBlasTbmv(Stream *stream, blas::UpperLower uplo,
                          blas::Transpose trans, blas::Diagonal diag, uint64 n,
                          uint64 k, const DeviceMemory<float> &a, int lda,
                          DeviceMemory<float> *x, int incx) {
  return false;
  // return DoBlasInternal(cublasStbmv, stream,
  //                       true /* = pointer_mode_host */,
  //                       CLBlasUpperLower(uplo), CLBlasTranspose(trans),
  //                       CLBlasDiagonal(diag), n, k, CLMemory(a), lda,
  //                       CLMemoryMutable(x), incx);
}

bool CLBlas::DoBlasTbmv(Stream *stream, blas::UpperLower uplo,
                          blas::Transpose trans, blas::Diagonal diag, uint64 n,
                          uint64 k, const DeviceMemory<double> &a, int lda,
                          DeviceMemory<double> *x, int incx) {
  return false;
  // return DoBlasInternal(cublasDtbmv, stream,
  //                       true /* = pointer_mode_host */,
  //                       CLBlasUpperLower(uplo), CLBlasTranspose(trans),
  //                       CLBlasDiagonal(diag), n, k, CLMemory(a), lda,
  //                       CLMemoryMutable(x), incx);
}

bool CLBlas::DoBlasTbmv(Stream *stream, blas::UpperLower uplo,
                          blas::Transpose trans, blas::Diagonal diag, uint64 n,
                          uint64 k, const DeviceMemory<std::complex<float>> &a,
                          int lda, DeviceMemory<std::complex<float>> *x,
                          int incx) {
  return false;
  // return DoBlasInternal(
  //     cublasCtbmv, stream, true /* = pointer_mode_host */,
  //     CLBlasUpperLower(uplo), CLBlasTranspose(trans),
  //     CLBlasDiagonal(diag), n, k, CLComplex(CLMemory(a)), lda,
  //     CLComplex(CLMemoryMutable(x)), incx);
}

bool CLBlas::DoBlasTbmv(Stream *stream, blas::UpperLower uplo,
                          blas::Transpose trans, blas::Diagonal diag, uint64 n,
                          uint64 k, const DeviceMemory<std::complex<double>> &a,
                          int lda, DeviceMemory<std::complex<double>> *x,
                          int incx) {
  return false;
  // return DoBlasInternal(
  //     cublasZtbmv, stream, true /* = pointer_mode_host */,
  //     CLBlasUpperLower(uplo), CLBlasTranspose(trans),
  //     CLBlasDiagonal(diag), n, k, CLComplex(CLMemory(a)), lda,
  //     CLComplex(CLMemoryMutable(x)), incx);
}

bool CLBlas::DoBlasTbsv(Stream *stream, blas::UpperLower uplo,
                          blas::Transpose trans, blas::Diagonal diag, uint64 n,
                          uint64 k, const DeviceMemory<float> &a, int lda,
                          DeviceMemory<float> *x, int incx) {
  return false;
  // return DoBlasInternal(cublasStbsv, stream,
  //                       true /* = pointer_mode_host */,
  //                       CLBlasUpperLower(uplo), CLBlasTranspose(trans),
  //                       CLBlasDiagonal(diag), n, k, CLMemory(a), lda,
  //                       CLMemoryMutable(x), incx);
}

bool CLBlas::DoBlasTbsv(Stream *stream, blas::UpperLower uplo,
                          blas::Transpose trans, blas::Diagonal diag, uint64 n,
                          uint64 k, const DeviceMemory<double> &a, int lda,
                          DeviceMemory<double> *x, int incx) {
  return false;
  // return DoBlasInternal(cublasDtbsv, stream,
  //                       true /* = pointer_mode_host */,
  //                       CLBlasUpperLower(uplo), CLBlasTranspose(trans),
  //                       CLBlasDiagonal(diag), n, k, CLMemory(a), lda,
  //                       CLMemoryMutable(x), incx);
}

bool CLBlas::DoBlasTbsv(Stream *stream, blas::UpperLower uplo,
                          blas::Transpose trans, blas::Diagonal diag, uint64 n,
                          uint64 k, const DeviceMemory<std::complex<float>> &a,
                          int lda, DeviceMemory<std::complex<float>> *x,
                          int incx) {
  return false;
  // return DoBlasInternal(
  //     cublasCtbsv, stream, true /* = pointer_mode_host */,
  //     CLBlasUpperLower(uplo), CLBlasTranspose(trans),
  //     CLBlasDiagonal(diag), n, k, CLComplex(CLMemory(a)), lda,
  //     CLComplex(CLMemoryMutable(x)), incx);
}

bool CLBlas::DoBlasTbsv(Stream *stream, blas::UpperLower uplo,
                          blas::Transpose trans, blas::Diagonal diag, uint64 n,
                          uint64 k, const DeviceMemory<std::complex<double>> &a,
                          int lda, DeviceMemory<std::complex<double>> *x,
                          int incx) {
  return false;
  // return DoBlasInternal(
  //     cublasZtbsv, stream, true /* = pointer_mode_host */,
  //     CLBlasUpperLower(uplo), CLBlasTranspose(trans),
  //     CLBlasDiagonal(diag), n, k, CLComplex(CLMemory(a)), lda,
  //     CLComplex(CLMemoryMutable(x)), incx);
}

bool CLBlas::DoBlasTpmv(Stream *stream, blas::UpperLower uplo,
                          blas::Transpose trans, blas::Diagonal diag, uint64 n,
                          const DeviceMemory<float> &ap, DeviceMemory<float> *x,
                          int incx) {
  return false;
  // return DoBlasInternal(
  //     cublasStpmv, stream, true /* = pointer_mode_host */,
  //     CLBlasUpperLower(uplo), CLBlasTranspose(trans),
  //     CLBlasDiagonal(diag), n, CLMemory(ap), CLMemoryMutable(x), incx);
}

bool CLBlas::DoBlasTpmv(Stream *stream, blas::UpperLower uplo,
                          blas::Transpose trans, blas::Diagonal diag, uint64 n,
                          const DeviceMemory<double> &ap,
                          DeviceMemory<double> *x, int incx) {
  return false;
  // return DoBlasInternal(
  //     cublasDtpmv, stream, true /* = pointer_mode_host */,
  //     CLBlasUpperLower(uplo), CLBlasTranspose(trans),
  //     CLBlasDiagonal(diag), n, CLMemory(ap), CLMemoryMutable(x), incx);
}

bool CLBlas::DoBlasTpmv(Stream *stream, blas::UpperLower uplo,
                          blas::Transpose trans, blas::Diagonal diag, uint64 n,
                          const DeviceMemory<std::complex<float>> &ap,
                          DeviceMemory<std::complex<float>> *x, int incx) {
  return false;
  // return DoBlasInternal(cublasCtpmv, stream,
  //                       true /* = pointer_mode_host */,
  //                       CLBlasUpperLower(uplo), CLBlasTranspose(trans),
  //                       CLBlasDiagonal(diag), n, CLComplex(CLMemory(ap)),
  //                       CLComplex(CLMemoryMutable(x)), incx);
}

bool CLBlas::DoBlasTpmv(Stream *stream, blas::UpperLower uplo,
                          blas::Transpose trans, blas::Diagonal diag, uint64 n,
                          const DeviceMemory<std::complex<double>> &ap,
                          DeviceMemory<std::complex<double>> *x, int incx) {
  return false;
  // return DoBlasInternal(cublasZtpmv, stream,
  //                       true /* = pointer_mode_host */,
  //                       CLBlasUpperLower(uplo), CLBlasTranspose(trans),
  //                       CLBlasDiagonal(diag), n, CLComplex(CLMemory(ap)),
  //                       CLComplex(CLMemoryMutable(x)), incx);
}

bool CLBlas::DoBlasTpsv(Stream *stream, blas::UpperLower uplo,
                          blas::Transpose trans, blas::Diagonal diag, uint64 n,
                          const DeviceMemory<float> &ap, DeviceMemory<float> *x,
                          int incx) {
  return false;
  // return DoBlasInternal(
  //     cublasStpsv, stream, true /* = pointer_mode_host */,
  //     CLBlasUpperLower(uplo), CLBlasTranspose(trans),
  //     CLBlasDiagonal(diag), n, CLMemory(ap), CLMemoryMutable(x), incx);
}

bool CLBlas::DoBlasTpsv(Stream *stream, blas::UpperLower uplo,
                          blas::Transpose trans, blas::Diagonal diag, uint64 n,
                          const DeviceMemory<double> &ap,
                          DeviceMemory<double> *x, int incx) {
  return false;
  // return DoBlasInternal(
  //     cublasDtpsv, stream, true /* = pointer_mode_host */,
  //     CLBlasUpperLower(uplo), CLBlasTranspose(trans),
  //     CLBlasDiagonal(diag), n, CLMemory(ap), CLMemoryMutable(x), incx);
}

bool CLBlas::DoBlasTpsv(Stream *stream, blas::UpperLower uplo,
                          blas::Transpose trans, blas::Diagonal diag, uint64 n,
                          const DeviceMemory<std::complex<float>> &ap,
                          DeviceMemory<std::complex<float>> *x, int incx) {
  return false;
  // return DoBlasInternal(cublasCtpsv, stream,
  //                       true /* = pointer_mode_host */,
  //                       CLBlasUpperLower(uplo), CLBlasTranspose(trans),
  //                       CLBlasDiagonal(diag), n, CLComplex(CLMemory(ap)),
  //                       CLComplex(CLMemoryMutable(x)), incx);
}

bool CLBlas::DoBlasTpsv(Stream *stream, blas::UpperLower uplo,
                          blas::Transpose trans, blas::Diagonal diag, uint64 n,
                          const DeviceMemory<std::complex<double>> &ap,
                          DeviceMemory<std::complex<double>> *x, int incx) {
  return false;
  // return DoBlasInternal(cublasZtpsv, stream,
  //                       true /* = pointer_mode_host */,
  //                       CLBlasUpperLower(uplo), CLBlasTranspose(trans),
  //                       CLBlasDiagonal(diag), n, CLComplex(CLMemory(ap)),
  //                       CLComplex(CLMemoryMutable(x)), incx);
}

bool CLBlas::DoBlasTrmv(Stream *stream, blas::UpperLower uplo,
                          blas::Transpose trans, blas::Diagonal diag, uint64 n,
                          const DeviceMemory<float> &a, int lda,
                          DeviceMemory<float> *x, int incx) {
  return false;
  // return DoBlasInternal(cublasStrmv, stream,
  //                       true /* = pointer_mode_host */,
  //                       CLBlasUpperLower(uplo), CLBlasTranspose(trans),
  //                       CLBlasDiagonal(diag), n, CLMemory(a), lda,
  //                       CLMemoryMutable(x), incx);
}

bool CLBlas::DoBlasTrmv(Stream *stream, blas::UpperLower uplo,
                          blas::Transpose trans, blas::Diagonal diag, uint64 n,
                          const DeviceMemory<double> &a, int lda,
                          DeviceMemory<double> *x, int incx) {
  return false;
  // return DoBlasInternal(cublasDtrmv, stream,
  //                       true /* = pointer_mode_host */,
  //                       CLBlasUpperLower(uplo), CLBlasTranspose(trans),
  //                       CLBlasDiagonal(diag), n, CLMemory(a), lda,
  //                       CLMemoryMutable(x), incx);
}

bool CLBlas::DoBlasTrmv(Stream *stream, blas::UpperLower uplo,
                          blas::Transpose trans, blas::Diagonal diag, uint64 n,
                          const DeviceMemory<std::complex<float>> &a, int lda,
                          DeviceMemory<std::complex<float>> *x, int incx) {
  return false;
  // return DoBlasInternal(cublasCtrmv, stream,
  //                       true /* = pointer_mode_host */,
  //                       CLBlasUpperLower(uplo), CLBlasTranspose(trans),
  //                       CLBlasDiagonal(diag), n, CLComplex(CLMemory(a)),
  //                       lda, CLComplex(CLMemoryMutable(x)), incx);
}

bool CLBlas::DoBlasTrmv(Stream *stream, blas::UpperLower uplo,
                          blas::Transpose trans, blas::Diagonal diag, uint64 n,
                          const DeviceMemory<std::complex<double>> &a, int lda,
                          DeviceMemory<std::complex<double>> *x, int incx) {
  return false;
  // return DoBlasInternal(cublasZtrmv, stream,
  //                       true /* = pointer_mode_host */,
  //                       CLBlasUpperLower(uplo), CLBlasTranspose(trans),
  //                       CLBlasDiagonal(diag), n, CLComplex(CLMemory(a)),
  //                       lda, CLComplex(CLMemoryMutable(x)), incx);
}

bool CLBlas::DoBlasTrsv(Stream *stream, blas::UpperLower uplo,
                          blas::Transpose trans, blas::Diagonal diag, uint64 n,
                          const DeviceMemory<float> &a, int lda,
                          DeviceMemory<float> *x, int incx) {
  return false;
  // return DoBlasInternal(cublasStrsv, stream,
  //                       true /* = pointer_mode_host */,
  //                       CLBlasUpperLower(uplo), CLBlasTranspose(trans),
  //                       CLBlasDiagonal(diag), n, CLMemory(a), lda,
  //                       CLMemoryMutable(x), incx);
}

bool CLBlas::DoBlasTrsv(Stream *stream, blas::UpperLower uplo,
                          blas::Transpose trans, blas::Diagonal diag, uint64 n,
                          const DeviceMemory<double> &a, int lda,
                          DeviceMemory<double> *x, int incx) {
  return false;
  // return DoBlasInternal(cublasDtrsv, stream,
  //                       true /* = pointer_mode_host */,
  //                       CLBlasUpperLower(uplo), CLBlasTranspose(trans),
  //                       CLBlasDiagonal(diag), n, CLMemory(a), lda,
  //                       CLMemoryMutable(x), incx);
}

bool CLBlas::DoBlasTrsv(Stream *stream, blas::UpperLower uplo,
                          blas::Transpose trans, blas::Diagonal diag, uint64 n,
                          const DeviceMemory<std::complex<float>> &a, int lda,
                          DeviceMemory<std::complex<float>> *x, int incx) {
  return false;
  // return DoBlasInternal(cublasCtrsv, stream,
  //                       true /* = pointer_mode_host */,
  //                       CLBlasUpperLower(uplo), CLBlasTranspose(trans),
  //                       CLBlasDiagonal(diag), n, CLComplex(CLMemory(a)),
  //                       lda, CLComplex(CLMemoryMutable(x)), incx);
}

bool CLBlas::DoBlasTrsv(Stream *stream, blas::UpperLower uplo,
                          blas::Transpose trans, blas::Diagonal diag, uint64 n,
                          const DeviceMemory<std::complex<double>> &a, int lda,
                          DeviceMemory<std::complex<double>> *x, int incx) {
  return false;
  // return DoBlasInternal(cublasZtrsv, stream,
  //                       true /* = pointer_mode_host */,
  //                       CLBlasUpperLower(uplo), CLBlasTranspose(trans),
  //                       CLBlasDiagonal(diag), n, CLComplex(CLMemory(a)),
  //                       lda, CLComplex(CLMemoryMutable(x)), incx);
}

bool CLBlas::DoBlasGemm(
    Stream *stream, blas::Transpose transa,
    blas::Transpose transb, uint64 m, uint64 n, uint64 k,
    float alpha, const DeviceMemory<Eigen::half> &a, int lda,
    const DeviceMemory<Eigen::half> &b, int ldb, float beta,
    DeviceMemory<Eigen::half> *c, int ldc) {
  return false;
// #if CL_VERSION >= 7050
//   VLOG(1) << port::Printf(
//       "doing CLBlast SGEMM: at=%d bt=%d m=%llu n=%llu "
//       "k=%llu alpha=%f a=%p lda=%d b=%p ldb=%d beta=%f "
//       "c=%p ldc=%d",
//       static_cast<int>(transa), static_cast<int>(transb), m, n, k, alpha,
//       a.opaque(), lda, b.opaque(), ldb, beta, c->opaque(), ldc);
//   if (transa == blas::Transpose::kNoTranspose) {
//     if (lda < static_cast<int64>(m)) {
//       LOG(WARNING) << "GEMM lda was smaller than m (no transpose case); "
//                       "precondition violation";
//     }
//   } else {
//     if (lda < static_cast<int64>(k)) {
//       LOG(WARNING) << "GEMM lda (" << lda << ") was smaller than k (" << k
//                    << ") (transpose case); precondition violation";
//     }
//   }
//   if (transb == blas::Transpose::kNoTranspose) {
//     if (ldb < static_cast<int64>(k)) {
//       LOG(WARNING) << "GEMM ldb (" << ldb << ") was smaller than k (" << k
//                    << ") (no transpose case); precondition violation";
//     }
//   } else {
//     if (ldb < static_cast<int64>(n)) {
//       LOG(WARNING) << "GEMM ldb was smaller than n (transpose case); "
//                       "precondition violation";
//     }
//   }
//   // TODO(sesse): Consider supporting the Hgemm interface, which uses half
//   // calculations internally (faster on newer devices, such as Pascal and TX1,
//   // but less precise).
//   return DoBlasInternal(
//       cublasSgemmEx, stream, true /* = pointer_mode_host */,
//       CLBlasTranspose(transa), CLBlasTranspose(transb), m, n, k, &alpha,
//       CLMemory(a), SE_CL_DATA_HALF, lda,
//       CLMemory(b), SE_CL_DATA_HALF, ldb,
//       &beta,
//       CLMemoryMutable(c), SE_CL_DATA_HALF, ldc);
// #else
//   LOG(ERROR) << "fp16 sgemm is not implemented in this CLBlast version "
//              << "(need at least CL 7.5)";
//   return false;
// #endif
}

bool CLBlas::DoBlasGemm(Stream *stream, blas::Transpose transa,
                          blas::Transpose transb, uint64 m, uint64 n, uint64 k,
                          double alpha, const DeviceMemory<double> &a, int lda,
                          const DeviceMemory<double> &b, int ldb, double beta,
                          DeviceMemory<double> *c, int ldc) {
  return false;
  // return DoBlasInternal(
  //     cublasDgemm, stream, true /* = pointer_mode_host */,
  //     CLBlasTranspose(transa), CLBlasTranspose(transb), m, n, k, &alpha,
  //     CLMemory(a), lda, CLMemory(b), ldb, &beta, CLMemoryMutable(c), ldc);
}

bool CLBlas::DoBlasGemm(Stream *stream, blas::Transpose transa,
                          blas::Transpose transb, uint64 m, uint64 n, uint64 k,
                          std::complex<float> alpha,
                          const DeviceMemory<std::complex<float>> &a, int lda,
                          const DeviceMemory<std::complex<float>> &b, int ldb,
                          std::complex<float> beta,
                          DeviceMemory<std::complex<float>> *c, int ldc) {
  return false;
  // return DoBlasInternal(
  //     cublasCgemm, stream, true /* = pointer_mode_host */,
  //     CLBlasTranspose(transa), CLBlasTranspose(transb), m, n, k,
  //     CLComplex(&alpha), CLComplex(CLMemory(a)), lda,
  //     CLComplex(CLMemory(b)), ldb, CLComplex(&beta),
  //     CLComplex(CLMemoryMutable(c)), ldc);
}

bool CLBlas::DoBlasGemm(Stream *stream, blas::Transpose transa,
                          blas::Transpose transb, uint64 m, uint64 n, uint64 k,
                          std::complex<double> alpha,
                          const DeviceMemory<std::complex<double>> &a, int lda,
                          const DeviceMemory<std::complex<double>> &b, int ldb,
                          std::complex<double> beta,
                          DeviceMemory<std::complex<double>> *c, int ldc) {
  return false;
  // return DoBlasInternal(
  //     cublasZgemm, stream, true /* = pointer_mode_host */,
  //     CLBlasTranspose(transa), CLBlasTranspose(transb), m, n, k,
  //     CLComplex(&alpha), CLComplex(CLMemory(a)), lda,
  //     CLComplex(CLMemory(b)), ldb, CLComplex(&beta),
  //     CLComplex(CLMemoryMutable(c)), ldc);
}

template <typename T, typename FuncT>
port::Status CLBlas::DoBlasGemmBatchedInternal(
    FuncT cublas_func, Stream *stream, blas::Transpose transa,
    blas::Transpose transb, uint64 m, uint64 n, uint64 k, T alpha,
    const port::ArraySlice<DeviceMemory<T> *> &a_ptrs_to_wrappers, int lda,
    const port::ArraySlice<DeviceMemory<T> *> &b_ptrs_to_wrappers, int ldb,
    T beta, const port::ArraySlice<DeviceMemory<T> *> &c_ptrs_to_wrappers,
    int ldc, int batch_count, ScratchAllocator *scratch_allocator) {
  // return false;
  // std::vector<T *> a_raw_ptrs, b_raw_ptrs, c_raw_ptrs;
  // for (int i = 0; i < batch_count; ++i) {
  //   a_raw_ptrs.push_back(static_cast<T *>(a_ptrs_to_wrappers[i]->opaque()));
  //   b_raw_ptrs.push_back(static_cast<T *>(b_ptrs_to_wrappers[i]->opaque()));
  //   c_raw_ptrs.push_back(static_cast<T *>(c_ptrs_to_wrappers[i]->opaque()));
  // }

  // typedef typename CLComplexT<T>::type CL_T;

  // const size_t size = batch_count * sizeof(CL_T *);

  // // Device-side copy of pointers to matrices.
  // DeviceMemory<CL_T *> a;
  // DeviceMemory<CL_T *> b;
  // DeviceMemory<CL_T *> c;

  // // If temporary space is allocated for device-side copies of pointers to
  // // matrices, that temporary space should not be freed until this function
  // // returns. Although the values for these unique_ptrs are not set here, they
  // // are declared at this scope so they will be destroyed when the function
  // // returns.
  // //
  // // If a scratch allocator is provided, these pointers will not be used at all.
  // std::unique_ptr<TemporaryDeviceMemory<CL_T *>> a_temporary;
  // std::unique_ptr<TemporaryDeviceMemory<CL_T *>> b_temporary;
  // std::unique_ptr<TemporaryDeviceMemory<CL_T *>> c_temporary;

  // // Decide how to allocate device-side copy of pointers to matrices based on
  // // whether a scratch allocator was passed.
  // if (scratch_allocator != nullptr) {
  //   SE_ASSIGN_OR_RETURN(DeviceMemory<uint8> a_bytes,
  //                       scratch_allocator->AllocateBytes(stream, size));
  //   SE_ASSIGN_OR_RETURN(DeviceMemory<uint8> b_bytes,
  //                       scratch_allocator->AllocateBytes(stream, size));
  //   SE_ASSIGN_OR_RETURN(DeviceMemory<uint8> c_bytes,
  //                       scratch_allocator->AllocateBytes(stream, size));
  //   a = DeviceMemory<CL_T *>(a_bytes);
  //   b = DeviceMemory<CL_T *>(b_bytes);
  //   c = DeviceMemory<CL_T *>(c_bytes);
  // } else {
  //   SE_ASSIGN_OR_RETURN(a_temporary,
  //                       stream->AllocateTemporaryArray<CL_T *>(batch_count));
  //   SE_ASSIGN_OR_RETURN(b_temporary,
  //                       stream->AllocateTemporaryArray<CL_T *>(batch_count));
  //   SE_ASSIGN_OR_RETURN(c_temporary,
  //                       stream->AllocateTemporaryArray<CL_T *>(batch_count));
  //   a = DeviceMemory<CL_T *>(*a_temporary->mutable_device_memory());
  //   b = DeviceMemory<CL_T *>(*b_temporary->mutable_device_memory());
  //   c = DeviceMemory<CL_T *>(*c_temporary->mutable_device_memory());
  // }

  // if (!stream->ThenMemcpy(&a, a_raw_ptrs.data(), size).ok() ||
  //     !stream->ThenMemcpy(&b, b_raw_ptrs.data(), size).ok() ||
  //     !stream->ThenMemcpy(&c, c_raw_ptrs.data(), size).ok()) {
  //   return port::Status(port::error::INTERNAL,
  //                       "failed to copy memory from host to device in "
  //                       "CLBlas::DoBlasGemmBatched");
  // }

  // bool ok = DoBlasInternal(
  //     cublas_func, stream, true /* = pointer_mode_host */,
  //     CLBlasTranspose(transa), CLBlasTranspose(transb), m, n, k,
  //     CLComplex(&alpha), const_cast<const CL_T **>(CLMemory(a)), lda,
  //     const_cast<const CL_T **>(CLMemory(b)), ldb, CLComplex(&beta),
  //     const_cast<CL_T **>(CLMemory(c)), ldc, batch_count);

  // if (ok) {
  //   return port::Status::OK();
  // }
  return port::Status(port::error::INTERNAL,
                      "failed BLAS call, see log for details");
}

bool CLBlas::DoBlasGemmBatched(
    Stream *stream, blas::Transpose transa, blas::Transpose transb, uint64 m,
    uint64 n, uint64 k, float alpha,
    const port::ArraySlice<DeviceMemory<float> *> &a_array, int lda,
    const port::ArraySlice<DeviceMemory<float> *> &b_array, int ldb, float beta,
    const port::ArraySlice<DeviceMemory<float> *> &c_array, int ldc,
    int batch_count, ScratchAllocator *scratch_allocator) {
      std::cout << "CLBlas::DoBlasGemmBatched()" << std::endl;
  return false;
  // SE_RETURN_STATUS_AS_BOOL(DoBlasGemmBatchedInternal(
  //     cublasSgemmBatched, stream, transa, transb, m, n, k, alpha,
  //     a_array, lda, b_array, ldb, beta, c_array, ldc, batch_count,
  //     scratch_allocator));
}

bool CLBlas::DoBlasGemmBatched(
    Stream *stream, blas::Transpose transa, blas::Transpose transb, uint64 m,
    uint64 n, uint64 k, double alpha,
    const port::ArraySlice<DeviceMemory<double> *> &a_array, int lda,
    const port::ArraySlice<DeviceMemory<double> *> &b_array, int ldb,
    double beta, const port::ArraySlice<DeviceMemory<double> *> &c_array,
    int ldc, int batch_count, ScratchAllocator *scratch_allocator) {
  return false;
  // SE_RETURN_STATUS_AS_BOOL(DoBlasGemmBatchedInternal(
  //     cublasDgemmBatched, stream, transa, transb, m, n, k, alpha,
  //     a_array, lda, b_array, ldb, beta, c_array, ldc, batch_count,
  //     scratch_allocator));
}

bool CLBlas::DoBlasGemmBatched(
    Stream *stream, blas::Transpose transa, blas::Transpose transb, uint64 m,
    uint64 n, uint64 k, std::complex<float> alpha,
    const port::ArraySlice<DeviceMemory<std::complex<float>> *> &a_array,
    int lda,
    const port::ArraySlice<DeviceMemory<std::complex<float>> *> &b_array,
    int ldb, std::complex<float> beta,
    const port::ArraySlice<DeviceMemory<std::complex<float>> *> &c_array,
    int ldc, int batch_count, ScratchAllocator *scratch_allocator) {
  return false;
  // SE_RETURN_STATUS_AS_BOOL(DoBlasGemmBatchedInternal(
  //     cublasCgemmBatched, stream, transa, transb, m, n, k, alpha,
  //     a_array, lda, b_array, ldb, beta, c_array, ldc, batch_count,
  //     scratch_allocator));
}

bool CLBlas::DoBlasGemmBatched(
    Stream *stream, blas::Transpose transa, blas::Transpose transb, uint64 m,
    uint64 n, uint64 k, std::complex<double> alpha,
    const port::ArraySlice<DeviceMemory<std::complex<double>> *> &a_array,
    int lda,
    const port::ArraySlice<DeviceMemory<std::complex<double>> *> &b_array,
    int ldb, std::complex<double> beta,
    const port::ArraySlice<DeviceMemory<std::complex<double>> *> &c_array,
    int ldc, int batch_count, ScratchAllocator *scratch_allocator) {
  return false;
  // SE_RETURN_STATUS_AS_BOOL(DoBlasGemmBatchedInternal(
  //     cublasZgemmBatched, stream, transa, transb, m, n, k, alpha,
  //     a_array, lda, b_array, ldb, beta, c_array, ldc, batch_count,
  //     scratch_allocator));
}

bool CLBlas::DoBlasHemm(Stream *stream, blas::Side side,
                          blas::UpperLower uplo, uint64 m, uint64 n,
                          std::complex<float> alpha,
                          const DeviceMemory<std::complex<float>> &a, int lda,
                          const DeviceMemory<std::complex<float>> &b, int ldb,
                          std::complex<float> beta,
                          DeviceMemory<std::complex<float>> *c, int ldc) {
  return false;
  // return DoBlasInternal(
  //     cublasChemm, stream, true /* = pointer_mode_host */,
  //     CLBlasSide(side), CLBlasUpperLower(uplo), m, n, CLComplex(&alpha),
  //     CLComplex(CLMemory(a)), lda, CLComplex(CLMemory(b)), ldb,
  //     CLComplex(&beta), CLComplex(CLMemoryMutable(c)), ldc);
}

bool CLBlas::DoBlasHemm(Stream *stream, blas::Side side,
                          blas::UpperLower uplo, uint64 m, uint64 n,
                          std::complex<double> alpha,
                          const DeviceMemory<std::complex<double>> &a, int lda,
                          const DeviceMemory<std::complex<double>> &b, int ldb,
                          std::complex<double> beta,
                          DeviceMemory<std::complex<double>> *c, int ldc) {
  return false;
  // return DoBlasInternal(
  //     cublasZhemm, stream, true /* = pointer_mode_host */,
  //     CLBlasSide(side), CLBlasUpperLower(uplo), m, n, CLComplex(&alpha),
  //     CLComplex(CLMemory(a)), lda, CLComplex(CLMemory(b)), ldb,
  //     CLComplex(&beta), CLComplex(CLMemoryMutable(c)), ldc);
}

bool CLBlas::DoBlasHerk(Stream *stream, blas::UpperLower uplo,
                          blas::Transpose trans, uint64 n, uint64 k,
                          float alpha,
                          const DeviceMemory<std::complex<float>> &a, int lda,
                          float beta, DeviceMemory<std::complex<float>> *c,
                          int ldc) {
  return false;
  // return DoBlasInternal(cublasCherk, stream,
  //                       true /* = pointer_mode_host */,
  //                       CLBlasUpperLower(uplo), CLBlasTranspose(trans), n,
  //                       k, CLComplex(&alpha), CLComplex(CLMemory(a)), lda,
  //                       &beta, CLComplex(CLMemoryMutable(c)), ldc);
}

bool CLBlas::DoBlasHerk(Stream *stream, blas::UpperLower uplo,
                          blas::Transpose trans, uint64 n, uint64 k,
                          double alpha,
                          const DeviceMemory<std::complex<double>> &a, int lda,
                          double beta, DeviceMemory<std::complex<double>> *c,
                          int ldc) {
  return false;
  // return DoBlasInternal(cublasZherk, stream,
  //                       true /* = pointer_mode_host */,
  //                       CLBlasUpperLower(uplo), CLBlasTranspose(trans), n,
  //                       k, CLComplex(&alpha), CLComplex(CLMemory(a)), lda,
  //                       &beta, CLComplex(CLMemoryMutable(c)), ldc);
}

bool CLBlas::DoBlasHer2k(Stream *stream, blas::UpperLower uplo,
                           blas::Transpose trans, uint64 n, uint64 k,
                           std::complex<float> alpha,
                           const DeviceMemory<std::complex<float>> &a, int lda,
                           const DeviceMemory<std::complex<float>> &b, int ldb,
                           float beta, DeviceMemory<std::complex<float>> *c,
                           int ldc) {
  return false;
  // return DoBlasInternal(cublasCher2k, stream,
  //                       true /* = pointer_mode_host */,
  //                       CLBlasUpperLower(uplo), CLBlasTranspose(trans), n,
  //                       k, CLComplex(&alpha), CLComplex(CLMemory(a)), lda,
  //                       CLComplex(CLMemory(b)), ldb, &beta,
  //                       CLComplex(CLMemoryMutable(c)), ldc);
}

bool CLBlas::DoBlasHer2k(Stream *stream, blas::UpperLower uplo,
                           blas::Transpose trans, uint64 n, uint64 k,
                           std::complex<double> alpha,
                           const DeviceMemory<std::complex<double>> &a, int lda,
                           const DeviceMemory<std::complex<double>> &b, int ldb,
                           double beta, DeviceMemory<std::complex<double>> *c,
                           int ldc) {
  return false;
  // return DoBlasInternal(cublasZher2k, stream,
  //                       true /* = pointer_mode_host */,
  //                       CLBlasUpperLower(uplo), CLBlasTranspose(trans), n,
  //                       k, CLComplex(&alpha), CLComplex(CLMemory(a)), lda,
  //                       CLComplex(CLMemory(b)), ldb, &beta,
  //                       CLComplex(CLMemoryMutable(c)), ldc);
}

bool CLBlas::DoBlasSymm(Stream *stream, blas::Side side,
                          blas::UpperLower uplo, uint64 m, uint64 n,
                          float alpha, const DeviceMemory<float> &a, int lda,
                          const DeviceMemory<float> &b, int ldb, float beta,
                          DeviceMemory<float> *c, int ldc) {
  return false;
  // return DoBlasInternal(
  //     cublasSsymm, stream, true /* = pointer_mode_host */,
  //     CLBlasSide(side), CLBlasUpperLower(uplo), m, n, &alpha, CLMemory(a),
  //     lda, CLMemory(b), ldb, &beta, CLMemoryMutable(c), ldc);
}

bool CLBlas::DoBlasSymm(Stream *stream, blas::Side side,
                          blas::UpperLower uplo, uint64 m, uint64 n,
                          double alpha, const DeviceMemory<double> &a, int lda,
                          const DeviceMemory<double> &b, int ldb, double beta,
                          DeviceMemory<double> *c, int ldc) {
  return false;
  // return DoBlasInternal(
  //     cublasDsymm, stream, true /* = pointer_mode_host */,
  //     CLBlasSide(side), CLBlasUpperLower(uplo), m, n, &alpha, CLMemory(a),
  //     lda, CLMemory(b), ldb, &beta, CLMemoryMutable(c), ldc);
}

bool CLBlas::DoBlasSymm(Stream *stream, blas::Side side,
                          blas::UpperLower uplo, uint64 m, uint64 n,
                          std::complex<float> alpha,
                          const DeviceMemory<std::complex<float>> &a, int lda,
                          const DeviceMemory<std::complex<float>> &b, int ldb,
                          std::complex<float> beta,
                          DeviceMemory<std::complex<float>> *c, int ldc) {
  return false;
  // return DoBlasInternal(
  //     cublasCsymm, stream, true /* = pointer_mode_host */,
  //     CLBlasSide(side), CLBlasUpperLower(uplo), m, n, CLComplex(&alpha),
  //     CLComplex(CLMemory(a)), lda, CLComplex(CLMemory(b)), ldb,
  //     CLComplex(&beta), CLComplex(CLMemoryMutable(c)), ldc);
}

bool CLBlas::DoBlasSymm(Stream *stream, blas::Side side,
                          blas::UpperLower uplo, uint64 m, uint64 n,
                          std::complex<double> alpha,
                          const DeviceMemory<std::complex<double>> &a, int lda,
                          const DeviceMemory<std::complex<double>> &b, int ldb,
                          std::complex<double> beta,
                          DeviceMemory<std::complex<double>> *c, int ldc) {
  return false;
  // return DoBlasInternal(
  //     cublasZsymm, stream, true /* = pointer_mode_host */,
  //     CLBlasSide(side), CLBlasUpperLower(uplo), m, n, CLComplex(&alpha),
  //     CLComplex(CLMemory(a)), lda, CLComplex(CLMemory(b)), ldb,
  //     CLComplex(&beta), CLComplex(CLMemoryMutable(c)), ldc);
}

bool CLBlas::DoBlasSyrk(Stream *stream, blas::UpperLower uplo,
                          blas::Transpose trans, uint64 n, uint64 k,
                          float alpha, const DeviceMemory<float> &a, int lda,
                          float beta, DeviceMemory<float> *c, int ldc) {
  return false;
  // return DoBlasInternal(
  //     cublasSsyrk, stream, true /* = pointer_mode_host */,
  //     CLBlasUpperLower(uplo), CLBlasTranspose(trans), n, k, &alpha,
  //     CLMemory(a), lda, &beta, CLMemoryMutable(c), ldc);
}

bool CLBlas::DoBlasSyrk(Stream *stream, blas::UpperLower uplo,
                          blas::Transpose trans, uint64 n, uint64 k,
                          double alpha, const DeviceMemory<double> &a, int lda,
                          double beta, DeviceMemory<double> *c, int ldc) {
  return false;
  // return DoBlasInternal(
  //     cublasDsyrk, stream, true /* = pointer_mode_host */,
  //     CLBlasUpperLower(uplo), CLBlasTranspose(trans), n, k, &alpha,
  //     CLMemory(a), lda, &beta, CLMemoryMutable(c), ldc);
}

bool CLBlas::DoBlasSyrk(Stream *stream, blas::UpperLower uplo,
                          blas::Transpose trans, uint64 n, uint64 k,
                          std::complex<float> alpha,
                          const DeviceMemory<std::complex<float>> &a, int lda,
                          std::complex<float> beta,
                          DeviceMemory<std::complex<float>> *c, int ldc) {
  return false;
  // return DoBlasInternal(
  //     cublasCsyrk, stream, true /* = pointer_mode_host */,
  //     CLBlasUpperLower(uplo), CLBlasTranspose(trans), n, k,
  //     CLComplex(&alpha), CLComplex(CLMemory(a)), lda, CLComplex(&beta),
  //     CLComplex(CLMemoryMutable(c)), ldc);
}

bool CLBlas::DoBlasSyrk(Stream *stream, blas::UpperLower uplo,
                          blas::Transpose trans, uint64 n, uint64 k,
                          std::complex<double> alpha,
                          const DeviceMemory<std::complex<double>> &a, int lda,
                          std::complex<double> beta,
                          DeviceMemory<std::complex<double>> *c, int ldc) {
  return false;
  // return DoBlasInternal(
  //     cublasZsyrk, stream, true /* = pointer_mode_host */,
  //     CLBlasUpperLower(uplo), CLBlasTranspose(trans), n, k,
  //     CLComplex(&alpha), CLComplex(CLMemory(a)), lda, CLComplex(&beta),
  //     CLComplex(CLMemoryMutable(c)), ldc);
}

bool CLBlas::DoBlasSyr2k(Stream *stream, blas::UpperLower uplo,
                           blas::Transpose trans, uint64 n, uint64 k,
                           float alpha, const DeviceMemory<float> &a, int lda,
                           const DeviceMemory<float> &b, int ldb, float beta,
                           DeviceMemory<float> *c, int ldc) {
  return false;
  // return DoBlasInternal(
  //     cublasSsyr2k, stream, true /* = pointer_mode_host */,
  //     CLBlasUpperLower(uplo), CLBlasTranspose(trans), n, k, &alpha,
  //     CLMemory(a), lda, CLMemory(b), ldb, &beta, CLMemoryMutable(c), ldc);
}

bool CLBlas::DoBlasSyr2k(Stream *stream, blas::UpperLower uplo,
                           blas::Transpose trans, uint64 n, uint64 k,
                           double alpha, const DeviceMemory<double> &a, int lda,
                           const DeviceMemory<double> &b, int ldb, double beta,
                           DeviceMemory<double> *c, int ldc) {
  return false;
  // return DoBlasInternal(
  //     cublasDsyr2k, stream, true /* = pointer_mode_host */,
  //     CLBlasUpperLower(uplo), CLBlasTranspose(trans), n, k, &alpha,
  //     CLMemory(a), lda, CLMemory(b), ldb, &beta, CLMemoryMutable(c), ldc);
}

bool CLBlas::DoBlasSyr2k(Stream *stream, blas::UpperLower uplo,
                           blas::Transpose trans, uint64 n, uint64 k,
                           std::complex<float> alpha,
                           const DeviceMemory<std::complex<float>> &a, int lda,
                           const DeviceMemory<std::complex<float>> &b, int ldb,
                           std::complex<float> beta,
                           DeviceMemory<std::complex<float>> *c, int ldc) {
  return false;
  // return DoBlasInternal(cublasCsyr2k, stream,
  //                       true /* = pointer_mode_host */,
  //                       CLBlasUpperLower(uplo), CLBlasTranspose(trans), n,
  //                       k, CLComplex(&alpha), CLComplex(CLMemory(a)), lda,
  //                       CLComplex(CLMemory(b)), ldb, CLComplex(&beta),
  //                       CLComplex(CLMemoryMutable(c)), ldc);
}

bool CLBlas::DoBlasSyr2k(Stream *stream, blas::UpperLower uplo,
                           blas::Transpose trans, uint64 n, uint64 k,
                           std::complex<double> alpha,
                           const DeviceMemory<std::complex<double>> &a, int lda,
                           const DeviceMemory<std::complex<double>> &b, int ldb,
                           std::complex<double> beta,
                           DeviceMemory<std::complex<double>> *c, int ldc) {
  return false;
  // return DoBlasInternal(cublasZsyr2k, stream,
  //                       true /* = pointer_mode_host */,
  //                       CLBlasUpperLower(uplo), CLBlasTranspose(trans), n,
  //                       k, CLComplex(&alpha), CLComplex(CLMemory(a)), lda,
  //                       CLComplex(CLMemory(b)), ldb, CLComplex(&beta),
  //                       CLComplex(CLMemoryMutable(c)), ldc);
}

bool CLBlas::DoBlasTrmm(Stream *stream, blas::Side side,
                          blas::UpperLower uplo, blas::Transpose transa,
                          blas::Diagonal diag, uint64 m, uint64 n, float alpha,
                          const DeviceMemory<float> &a, int lda,
                          DeviceMemory<float> *b, int ldb) {
  return false;
  // return DoBlasInternal(
  //     cublasStrmm, stream, true /* = pointer_mode_host */,
  //     CLBlasSide(side), CLBlasUpperLower(uplo), CLBlasTranspose(transa),
  //     CLBlasDiagonal(diag), m, n, &alpha, CLMemory(a), lda,
  //     CLMemoryMutable(b), ldb, CLMemoryMutable(b), ldb);
}

bool CLBlas::DoBlasTrmm(Stream *stream, blas::Side side,
                          blas::UpperLower uplo, blas::Transpose transa,
                          blas::Diagonal diag, uint64 m, uint64 n, double alpha,
                          const DeviceMemory<double> &a, int lda,
                          DeviceMemory<double> *b, int ldb) {
  return false;
  // return DoBlasInternal(
  //     cublasDtrmm, stream, true /* = pointer_mode_host */,
  //     CLBlasSide(side), CLBlasUpperLower(uplo), CLBlasTranspose(transa),
  //     CLBlasDiagonal(diag), m, n, &alpha, CLMemory(a), lda,
  //     CLMemoryMutable(b), ldb, CLMemoryMutable(b), ldb);
}

bool CLBlas::DoBlasTrmm(Stream *stream, blas::Side side,
                          blas::UpperLower uplo, blas::Transpose transa,
                          blas::Diagonal diag, uint64 m, uint64 n,
                          std::complex<float> alpha,
                          const DeviceMemory<std::complex<float>> &a, int lda,
                          DeviceMemory<std::complex<float>> *b, int ldb) {
  return false;
  // return DoBlasInternal(
  //     cublasCtrmm, stream, true /* = pointer_mode_host */,
  //     CLBlasSide(side), CLBlasUpperLower(uplo), CLBlasTranspose(transa),
  //     CLBlasDiagonal(diag), m, n, CLComplex(&alpha),
  //     CLComplex(CLMemory(a)), lda, CLComplex(CLMemoryMutable(b)), ldb,
  //     CLComplex(CLMemoryMutable(b)), ldb);
}

bool CLBlas::DoBlasTrmm(Stream *stream, blas::Side side,
                          blas::UpperLower uplo, blas::Transpose transa,
                          blas::Diagonal diag, uint64 m, uint64 n,
                          std::complex<double> alpha,
                          const DeviceMemory<std::complex<double>> &a, int lda,
                          DeviceMemory<std::complex<double>> *b, int ldb) {
  return false;
  // return DoBlasInternal(
  //     cublasZtrmm, stream, true /* = pointer_mode_host */,
  //     CLBlasSide(side), CLBlasUpperLower(uplo), CLBlasTranspose(transa),
  //     CLBlasDiagonal(diag), m, n, CLComplex(&alpha),
  //     CLComplex(CLMemory(a)), lda, CLComplex(CLMemoryMutable(b)), ldb,
  //     CLComplex(CLMemoryMutable(b)), ldb);
}

bool CLBlas::DoBlasTrsm(Stream *stream, blas::Side side,
                          blas::UpperLower uplo, blas::Transpose transa,
                          blas::Diagonal diag, uint64 m, uint64 n, float alpha,
                          const DeviceMemory<float> &a, int lda,
                          DeviceMemory<float> *b, int ldb) {
  return false;
  // return DoBlasInternal(cublasStrsm, stream,
  //                       true /* = pointer_mode_host */, CLBlasSide(side),
  //                       CLBlasUpperLower(uplo), CLBlasTranspose(transa),
  //                       CLBlasDiagonal(diag), m, n, &alpha, CLMemory(a),
  //                       lda, CLMemoryMutable(b), ldb);
}

bool CLBlas::DoBlasTrsm(Stream *stream, blas::Side side,
                          blas::UpperLower uplo, blas::Transpose transa,
                          blas::Diagonal diag, uint64 m, uint64 n, double alpha,
                          const DeviceMemory<double> &a, int lda,
                          DeviceMemory<double> *b, int ldb) {
  return false;
  // return DoBlasInternal(cublasDtrsm, stream,
  //                       true /* = pointer_mode_host */, CLBlasSide(side),
  //                       CLBlasUpperLower(uplo), CLBlasTranspose(transa),
  //                       CLBlasDiagonal(diag), m, n, &alpha, CLMemory(a),
  //                       lda, CLMemoryMutable(b), ldb);
}

bool CLBlas::DoBlasTrsm(Stream *stream, blas::Side side,
                          blas::UpperLower uplo, blas::Transpose transa,
                          blas::Diagonal diag, uint64 m, uint64 n,
                          std::complex<float> alpha,
                          const DeviceMemory<std::complex<float>> &a, int lda,
                          DeviceMemory<std::complex<float>> *b, int ldb) {
  return false;
  // return DoBlasInternal(
  //     cublasCtrsm, stream, true /* = pointer_mode_host */,
  //     CLBlasSide(side), CLBlasUpperLower(uplo), CLBlasTranspose(transa),
  //     CLBlasDiagonal(diag), m, n, CLComplex(&alpha),
  //     CLComplex(CLMemory(a)), lda, CLComplex(CLMemoryMutable(b)), ldb);
}

bool CLBlas::DoBlasTrsm(Stream *stream, blas::Side side,
                          blas::UpperLower uplo, blas::Transpose transa,
                          blas::Diagonal diag, uint64 m, uint64 n,
                          std::complex<double> alpha,
                          const DeviceMemory<std::complex<double>> &a, int lda,
                          DeviceMemory<std::complex<double>> *b, int ldb) {
  return false;
  // return DoBlasInternal(
  //     cublasZtrsm, stream, true /* = pointer_mode_host */,
  //     CLBlasSide(side), CLBlasUpperLower(uplo), CLBlasTranspose(transa),
  //     CLBlasDiagonal(diag), m, n, CLComplex(&alpha),
  //     CLComplex(CLMemory(a)), lda, CLComplex(CLMemoryMutable(b)), ldb);
}

}  // namespace cl

namespace gpu = ::perftools::gputools;

void initialize_clblas() {
      // std::cout << "CLBlas::initialize_clblas()" << std::endl;
  gpu::port::Status status =
      gpu::PluginRegistry::Instance()
          ->RegisterFactory<gpu::PluginRegistry::BlasFactory>(
              gpu::cl::kClPlatformId, gpu::cl::kClBlasPlugin, "CLBlast",
              [](gpu::internal::StreamExecutorInterface
                     *parent) -> gpu::blas::BlasSupport * {
                gpu::cl::CLExecutor *cl_executor =
                    dynamic_cast<gpu::cl::CLExecutor *>(parent);
                if (cl_executor == nullptr) {
                  LOG(ERROR)
                      << "Attempting to initialize an instance of the CLBlast "
                      << "support library with a non-CL StreamExecutor";
                  return nullptr;
                }

                gpu::cl::CLBlas *blas =
                    new gpu::cl::CLBlas(cl_executor);
                if (!blas->Init()) {
                  // Note: Init() will log a more specific error.
                  delete blas;
                  return nullptr;
                }
                return blas;
              });

  if (!status.ok()) {
    LOG(ERROR) << "Unable to register CLBlast factory: "
               << status.error_message();
  }

  // Prime the CLBlast DSO. The loader will log more information.
  // auto statusor = gpu::internal::CachedDsoLoader::GetCublasDsoHandle();
  // if (!statusor.ok()) {
  //   LOG(INFO) << "Unable to load CLBlast DSO.";
  // }

  gpu::PluginRegistry::Instance()->SetDefaultFactory(gpu::cl::kClPlatformId,
                                                     gpu::PluginKind::kBlas,
                                                     gpu::cl::kClBlasPlugin);
}

}  // namespace gputools
}  // namespace perftools

REGISTER_MODULE_INITIALIZER(register_clblas,
                            { perftools::gputools::initialize_clblas(); });
