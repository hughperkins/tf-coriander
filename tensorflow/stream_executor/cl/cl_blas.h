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

// CL-specific support for BLAS functionality -- this wraps the CLBlast library
// capabilities, and is only included into CL implementation code -- it will
// not introduce cl headers into other code.

#ifndef TENSORFLOW_STREAM_EXECUTOR_CL_CLBLAST_H_
#define TENSORFLOW_STREAM_EXECUTOR_CL_CLBLAST_H_

#include "tensorflow/stream_executor/blas.h"
#include "tensorflow/stream_executor/lib/stringpiece.h"
#include "tensorflow/stream_executor/platform/mutex.h"
#include "tensorflow/stream_executor/platform/port.h"
#include "tensorflow/stream_executor/platform/thread_annotations.h"
#include "tensorflow/stream_executor/plugin_registry.h"

typedef struct cublasContext *cublasHandle_t;

namespace perftools {
namespace gputools {

class Stream;

namespace cl {

// Opaque and unique identifier for the CLBlast plugin.
extern const PluginId kClBlasPlugin;

class CLExecutor;

// BLAS plugin for CL platform via CLBlast library.
//
// This satisfies the platform-agnostic BlasSupport interface.
//
// Note that the CLBlast handle that this encapsulates is implicitly tied to the
// context (and, as a result, the device) that the parent CLExecutor is tied
// to. This simply happens as an artifact of creating the CLBlast handle when a
// CL context is active.
//
// Thread-safe post-initialization.
class CLBlas : public blas::BlasSupport {
 public:
  explicit CLBlas(CLExecutor *parent);

  // Allocates a CLBlast handle.
  bool Init();

  // Releases the CLBlast handle, if present.
  ~CLBlas() override;

  TENSORFLOW_STREAM_EXECUTOR_GPU_BLAS_SUPPORT_OVERRIDES

 private:
  // Tells CLBlast to enqueue the BLAS operation onto a particular Stream.
  //
  // CLBlast is stateful, and only be associated with one stream (in order to
  // enqueue dispatch) at a given time. As a result, this generally must be
  // invoked before calling into CLBlast.
  bool SetStream(Stream *stream) EXCLUSIVE_LOCKS_REQUIRED(mu_);

  // A helper function that calls the real CLBlast function together with error
  // handling.
  //
  // cublas_func:        CLBlast function pointer.
  // cublas_name:        CLBlast function name.
  // stream:             Stream to enqueue the BLAS operation onto.
  // pointer_mode_host:  Indicate if the pointer to a scalar value is from host
  //                     (true) or device (false).
  // args:               Arguments of CLBlast function.
  template <typename FuncT, typename... Args>
  bool DoBlasInternal(FuncT cublas_func, Stream *stream, bool pointer_mode_host,
                      Args... args);

  // A helper function to implement DoBlasGemmBatched interfaces for generic
  // types.
  template <typename T, typename FuncT>
  port::Status DoBlasGemmBatchedInternal(
      FuncT cublas_func, Stream *stream, blas::Transpose transa,
      blas::Transpose transb, uint64 m, uint64 n, uint64 k, T alpha,
      const port::ArraySlice<DeviceMemory<T> *> &a_array, int lda,
      const port::ArraySlice<DeviceMemory<T> *> &b_array, int ldb, T beta,
      const port::ArraySlice<DeviceMemory<T> *> &c_array, int ldc,
      int batch_count, ScratchAllocator *scratch_allocator);

  // mutex that guards the CLBlast handle for this device.
  mutex mu_;

  // CLExecutor which instantiated this CLBlas.
  // Immutable post-initialization.
  CLExecutor *parent_;

  // CLBlast library handle on the device.
  cublasHandle_t blas_ GUARDED_BY(mu_);

  SE_DISALLOW_COPY_AND_ASSIGN(CLBlas);
};

}  // namespace cl
}  // namespace gputools
}  // namespace perftools

#endif  // TENSORFLOW_STREAM_EXECUTOR_CL_CLBLAST_H_
