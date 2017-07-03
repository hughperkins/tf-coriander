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

// #include <cl_device_runtime_api.h>
#include "cuda.h"

#include "tensorflow/stream_executor/cl/cl_gpu_executor.h"

#if defined(__APPLE__)
#include <mach-o/dyld.h>
#endif
#include <unistd.h>

#include "tensorflow/stream_executor/cl/cl_platform_id.h"
#include "tensorflow/stream_executor/cl/cl_diagnostics.h"
#include "tensorflow/stream_executor/cl/cl_driver.h"
#include "tensorflow/stream_executor/cl/cl_event.h"
#include "tensorflow/stream_executor/cl/cl_platform.h"
#include "tensorflow/stream_executor/cl/cl_stream.h"
// #include "tensorflow/stream_executor/cl/cl_timer.h"
#include "tensorflow/stream_executor/dso_loader.h"
#include "tensorflow/stream_executor/kernel_cache_config.h"
#include "tensorflow/stream_executor/lib/casts.h"
#include "tensorflow/stream_executor/lib/env.h"
#include "tensorflow/stream_executor/lib/error.h"
#include "tensorflow/stream_executor/lib/initialize.h"
#include "tensorflow/stream_executor/lib/mathutil.h"
#include "tensorflow/stream_executor/lib/path.h"
#include "tensorflow/stream_executor/lib/process_state.h"
#include "tensorflow/stream_executor/lib/ptr_util.h"
#include "tensorflow/stream_executor/lib/statusor.h"
#include "tensorflow/stream_executor/lib/str_util.h"
#include "tensorflow/stream_executor/lib/strcat.h"
#include "tensorflow/stream_executor/lib/stringprintf.h"
#include "tensorflow/stream_executor/platform.h"
#include "tensorflow/stream_executor/platform/logging.h"
#include "tensorflow/stream_executor/platform/port.h"
#include "tensorflow/stream_executor/plugin_registry.h"
#include "tensorflow/stream_executor/stream.h"
#include "tensorflow/stream_executor/stream_executor_internal.h"
#include "tensorflow/stream_executor/stream_executor_pimpl.h"
#include "tensorflow/stream_executor/timer.h"
#include "tensorflow/stream_executor/lib/numbers.h"

#include <iostream>

#ifdef PLATFORMS_GPUS_CL_DYNAMIC_LIBCL_DYNAMIC_LIBCL_H_
#error \
    "No driver calls in this file, wrap driver functionality in cl_driver.cc."
#endif

#ifdef __CL_RUNTIME_H__
#error \
    "CL runtime being included into CL GPU executor; should be driver only."
#endif

extern bool FLAGS_check_gpu_leaks;
tensorflow::int32 FLAGS_register_occupancy_warning_threshold;
bool FLAGS_prefer_cubin_to_ptx = true;

namespace perftools {
namespace gputools {
namespace rng {
// class RngSupport;
}  // namespace rng
}  // namespace gputools
}  // namespace perftools

namespace perftools {
namespace gputools {

using namespace cl;
namespace cl {

class CLEvent;
class CLTimer;

// Hook that can be used to CUBIN-ate PTX before it is loaded into the driver.
// It has been observed that loading both PTX and cubins into the driver library
// can cause it to crash, but loading only CUBINs avoids those crashes;
// therefore, it's useful to have this hook to hack in uniform CUBIN-ation of
// PTX code.
//
// As this is an implementation-detail workaround, the usage is to declare this
// variable with extern linkage and populate it from another translation unit.
std::function<string(const string &)> g_cubinate;

static CLEvent *AsCLEvent(Event *event) {
  // std::cout << "cl_gpu_executor::AsCLEvent()" << std::endl;
  DCHECK(event != nullptr);
  return static_cast<CLEvent *>(event->implementation());
}


// Given a platform-independent timer datatype, returns the internal CL
// platform implementation pointer.
static CLTimer *AsCLTimer(Timer *timer) {
  std::cout << "cl_gpu_executor::AsCLTimer()" << std::endl;
  return 0;
  // DCHECK(timer != nullptr);
  // return static_cast<CLTimer *>(timer->implementation());
}

// Given const GPU memory, returns a libcl device pointer datatype, suitable
// for passing directly to libcl APIs.
//
// N.B. we must lose constness in order to pass a suitable type to the existing
// libcl APIs, so the caller should take care to only pass the result of const
// GPU memory conversions to libcl functions which will honor constness.
static CUdeviceptr AsClDevicePtr(const DeviceMemoryBase &gpu_mem) {
  // std::cout << "cl_gpu_executor::AsClDevicePtr()" << std::endl;
  return reinterpret_cast<CUdeviceptr>(gpu_mem.opaque());
}

// See description on const version above.
static CUdeviceptr AsClDevicePtr(DeviceMemoryBase *gpu_mem) {
  return AsClDevicePtr(*gpu_mem);
}

static ClContext* GetClContext(Stream *stream) {
  // std::cout << "cl_gpu_executor::GetClContext()" << std::endl;
  return static_cast<CLExecutor *>(stream->parent()->implementation())
      ->cl_context();
}

ClContext* ExtractClContext(CLExecutor *cl_exec) {
  // std::cout << "cl_gpu_executor::ExtractClContext()" << std::endl;
  CHECK(cl_exec != nullptr);
  return cl_exec->cl_context();
}

CLExecutor *ExtractClExecutor(StreamExecutor *stream_exec) {
  // std::cout << "cl_gpu_executor::ExtractClExecutor()" << std::endl;
  return static_cast<CLExecutor *>(stream_exec->implementation());
}

CLExecutor::~CLExecutor() {
  // std::cout << "CLExecutor::~CLExecutor()" << std::endl;
  for (auto &it : disk_modules_) {
    CLDriver::UnloadModule(context_, it.second);
  }
  for (auto &it : in_memory_modules_) {
    CLDriver::UnloadModule(context_, it.second);
  }
  if (context_ != nullptr) {
    CLDriver::DestroyContext(context_);
  }
}

port::Status CLExecutor::Init(int device_ordinal,
                                DeviceOptions device_options) {
  // std::cout << "CLExecutor::Init()" << std::endl;
  device_ordinal_ = device_ordinal;

  auto status = CLDriver::Init();
  if (!status.ok()) {
    std::cout << "CLExecutor::Init() CLDriver::Init() failed" << std::endl;
    return status;
  }

  status = CLDriver::GetDevice(device_ordinal_, &device_);
  if (!status.ok()) {
    std::cout << "CLExecutor::Init() CLDriver::getdevice() failed" << std::endl;
    return status;
  }

  status = CLDriver::CreateContext(device_, device_options, &context_);
  if (!status.ok()) {
    std::cout << "CLExecutor::Init() CLDriver::createcontext() failed" << std::endl;
    return status;
  }

  status = CLDriver::GetComputeCapability(&cc_major_, &cc_minor_, device_);
  // std::cout << "compute capability major=" << cc_major_ << " minor=" << cc_minor_ << std::endl;
  return status;
}

bool CLExecutor::FindOnDiskForComputeCapability(
    port::StringPiece filename, port::StringPiece canonical_suffix,
    string *found_filename) const {
  if (cc_major_ == 0 && cc_minor_ == 0) {
    return false;
  }

  // TODO(22689637): Eliminate unnecessary ToString()s when all dependencies
  // have been migrated.
  string cc_specific = port::StrCat(filename.ToString(), ".cc", cc_major_,
                                    cc_minor_, canonical_suffix.ToString());
  if (port::FileExists(cc_specific)) {
    VLOG(2) << "found compute-capability-specific file, using that: "
            << cc_specific;
    *found_filename = cc_specific;
    return true;
  }

  VLOG(2) << "could not find compute-capability specific file at: "
          << cc_specific;
  if (port::FileExists(filename.ToString())) {
    *found_filename = filename.ToString();
    return true;
  }

  return false;
}

// Returns the path to the running executable.
// N.B. Derived from //knowledge/smalltalk/background_kb.cc
// Arg: strip_exe: if true, remove the name of the executable itself from the
//                 returned string. Example: calling this from /usr/bin/foo
//                 would return /usr/bin.
static string GetBinaryDir(bool strip_exe) {
  char exe_path[PATH_MAX] = {0};
#if defined(__APPLE__)
    uint32_t buffer_size = 0U;
    _NSGetExecutablePath(nullptr, &buffer_size);
    char unresolved_path[buffer_size];
    _NSGetExecutablePath(unresolved_path, &buffer_size);
    CHECK_ERR(realpath(unresolved_path, exe_path) ? 1 : -1);
#else
    CHECK_ERR(readlink("/proc/self/exe", exe_path, sizeof(exe_path) - 1));
#endif
  // Make sure it's null-terminated:
  exe_path[sizeof(exe_path) - 1] = 0;

  if (strip_exe) {
    // The exe is the last component of the path, so remove one component.
    string ret = exe_path;
    std::vector<string> components = port::Split(exe_path, '/');
    components.pop_back();
    return port::Join(components, "/");
  }
  return exe_path;
}

bool CLExecutor::GetKernel(const MultiKernelLoaderSpec &spec,
                             KernelBase *kernel) {
  std::cout << "cl_gpu_executor::GetKernel()" << std::endl;
  return false;
  // CLKernel *cl_kernel = AsCLKernel(kernel);
  // CUmodule module = nullptr;
  // const string *kernelname;

  // const OnDiskKernelLoaderSpec *on_disk_spec = nullptr;
  // bool has_ptx = spec.has_cl_ptx_on_disk();
  // bool has_cubin = spec.has_cl_cubin_on_disk();
  // if (has_cubin && (!has_ptx || FLAGS_prefer_cubin_to_ptx)) {
  //   on_disk_spec = &spec.cl_cubin_on_disk();
  // } else if (has_ptx) {
  //   on_disk_spec = &spec.cl_ptx_on_disk();
  // }

  // if (on_disk_spec != nullptr) {
  //   LOG(WARNING) << "loading CL kernel from disk is not supported";
  //   return false;
  // } else if (spec.has_cl_ptx_in_memory()) {
  //   kernelname = &spec.cl_ptx_in_memory().kernelname();

  //   if (cc_major_ == 0 && cc_minor_ == 0) {
  //     return false;
  //   }

  //   // Note that the orignal ptx may be compressed, and the ptx we get below is
  //   // the decompressed result. To cache the module we should use the original
  //   // ptx (compressed one) as the key. This is because for the same compressed
  //   // ptx, we may get different decompressed ptx wrt the pointer value.
  //   const char *ptx = spec.cl_ptx_in_memory().text(cc_major_, cc_minor_);
  //   const char *orig_ptx =
  //       spec.cl_ptx_in_memory().original_text(cc_major_, cc_minor_);
  //   if (ptx == nullptr || orig_ptx == nullptr) {
  //     ptx = spec.cl_ptx_in_memory().default_text();
  //     orig_ptx = spec.cl_ptx_in_memory().original_default_text();
  //   }
  //   if (ptx == nullptr || orig_ptx == nullptr) {
  //     LOG(FATAL) << "could not load ptx for kernel " << kernelname;
  //     return false;
  //   }

  //   mutex_lock lock{in_memory_modules_mu_};
  //   module = in_memory_modules_[orig_ptx];

  //   if (module == nullptr) {
  //     if (g_cubinate == nullptr) {
  //       if (!CLDriver::LoadPtx(context_, ptx, &module)) {
  //         return false;
  //       }
  //     } else {
  //       string cubin = g_cubinate(ptx);
  //       auto load_status =
  //           CLDriver::LoadCubin(context_, cubin.c_str(), &module);
  //       if (!load_status.ok()) {
  //         LOG(ERROR) << "failed to load cubin via hook: " << load_status;
  //         return false;
  //       }
  //     }
  //     in_memory_modules_[orig_ptx] = module;
  //   }
  // } else if (spec.has_cl_cubin_in_memory()) {
  //   kernelname = &spec.cl_cubin_in_memory().kernelname();
  //   const char *cubin = spec.cl_cubin_in_memory().bytes();
  //   mutex_lock lock{in_memory_modules_mu_};
  //   module = in_memory_modules_[cubin];

  //   if (module == nullptr) {
  //     auto load_status = CLDriver::LoadCubin(context_, cubin, &module);
  //     if (!load_status.ok()) {
  //       LOG(ERROR) << "failed to load CUBIN: " << load_status;
  //       return false;
  //     }

  //     in_memory_modules_[cubin] = module;
  //   }
  // } else {
  //   LOG(WARNING) << "no method of loading CL kernel provided";
  //   return false;
  // }

  // VLOG(2) << "getting function " << kernelname << " from module " << module;
  // if (!CLDriver::GetModuleFunction(context_, module, kernelname->c_str(),
  //                                    cl_kernel->cl_function_ptr())) {
  //   return false;
  // }

  // // We have to trust the kernel loader spec arity because there doesn't appear
  // // to be a way to reflect on the number of expected arguments w/the CL API.
  // cl_kernel->set_arity(spec.arity());

  // KernelMetadata kernel_metadata;
  // if (!GetKernelMetadata(cl_kernel, &kernel_metadata)) {
  //   LOG(WARNING) << "Unable to get metadata for kernel " << kernelname;
  // }
  // kernel->set_metadata(kernel_metadata);
  // kernel->set_name(*kernelname);
  // return true;
}

bool CLExecutor::GetKernelMetadata(CLKernel *cl_kernel,
                                     KernelMetadata *kernel_metadata) {
  std::cout << "cl_gpu_executor::GetKernelMetadata()" << std::endl;
  return false;
  // int value;
  // if (!CLDriver::FuncGetAttribute(CU_FUNC_ATTRIBUTE_NUM_REGS,
  //                                   *cl_kernel->cl_function_ptr(),
  //                                   &value)) {
  //   return false;
  // }
  // kernel_metadata->set_registers_per_thread(value);

  // if (!CLDriver::FuncGetAttribute(CU_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES,
  //                                   *cl_kernel->cl_function_ptr(),
  //                                   &value)) {
  //   return false;
  // }
  // kernel_metadata->set_shared_memory_bytes(value);

  // return true;
}

bool CLExecutor::Launch(Stream *stream, const ThreadDim &thread_dims,
                          const BlockDim &block_dims, const KernelBase &kernel,
                          const std::vector<KernelArg> &args) {
  std::cout << "cl_gpu_executor::Launch()" << std::endl;
  return false;
  // CHECK_EQ(kernel.Arity(), args.size());
  // CUstream custream = AsCLStreamValue(stream);
  // const CLKernel *cl_kernel = AsCLKernel(&kernel);
  // CUfunction cufunc = cl_kernel->AsCLFunctionValue();

  // std::vector<void *> addrs;
  // addrs.reserve(args.size());
  // int shmem_bytes = 0;
  // for (size_t i = 0; i < args.size(); i++) {
  //   switch (args[i].type) {
  //     case KernelArg::kNormal:
  //       addrs.push_back(const_cast<void *>(
  //           static_cast<const void *>(args[i].data.begin())));
  //       break;
  //     case KernelArg::kSharedMemory:
  //       shmem_bytes += args[i].bytes;
  //       break;
  //     default:
  //       LOG(ERROR) << "Invalid kernel arg type passed (" << args[i].type
  //                  << ") for arg " << i;
  //       return false;
  //   }
  // }

  // // Only perform/print the occupancy check 1x.
  // launched_kernels_mu_.lock();
  // if (launched_kernels_.find(cufunc) == launched_kernels_.end()) {
  //   OccupancyCheck(kernel, thread_dims, block_dims);
  //   // TODO(rspringer): Remove elements from launched_kernels_...if we ever
  //   // expose a kernel/module deallocation method.
  //   launched_kernels_.insert(cufunc);
  // }
  // launched_kernels_mu_.unlock();

  // if (cl_kernel->GetPreferredCacheConfig() !=
  //     KernelCacheConfig::kNoPreference) {
  //   CLDriver::FuncSetCacheConfig(cufunc, cl_kernel->GetCLCacheConfig());
  // }

  // if (!CLDriver::LaunchKernel(
  //         GetClContext(stream), cufunc, block_dims.x, block_dims.y,
  //         block_dims.z, thread_dims.x, thread_dims.y, thread_dims.z,
  //         shmem_bytes, custream, addrs.data(), nullptr /* = extra */)) {
  //   LOG(ERROR) << "failed to launch CL kernel with args: " << args.size()
  //              << "; thread dim: " << thread_dims.ToString()
  //              << "; block dim: " << block_dims.ToString();
  //   return false;
  // }

  // return true;
}

// This is a non-essential operation; if there's a failure, proceed without
// logging an error. It's nearly certain that in case of failures, we'd never
// get here in the first place; these are very low-impact routines.
void CLExecutor::OccupancyCheck(const KernelBase &kernel,
                                  const ThreadDim &thread_dims,
                                  const BlockDim &block_dims) {
  std::cout << "cl_gpu_executor::OccupancyCheck()" << std::endl;
  // VLOG(2) << "Computing kernel occupancy for kernel "
  //         << kernel.demangled_name();
  // VLOG(2) << "Thread dimensions (" << thread_dims.x << ", " << thread_dims.y
  //         << ", " << thread_dims.z << ")";

  // int regs_per_thread;
  // if (!kernel.metadata().registers_per_thread(&regs_per_thread)) {
  //   return;
  // }

  // int smem_per_block;
  // if (!kernel.metadata().shared_memory_bytes(&smem_per_block)) {
  //   return;
  // }

  // const DeviceDescription &device_description =
  //     kernel.parent()->GetDeviceDescription();

  // uint64 blocks_per_sm = CalculateOccupancy(
  //     device_description, regs_per_thread, smem_per_block, thread_dims);
  // VLOG(2) << "Resident blocks per SM is " << blocks_per_sm;

  // // To increase occupancy, there must be a sufficient number of blocks
  // // available to spread across the sm's at this new improved occupancy level.
  // int multiprocessor_count = device_description.core_count();
  // int block_count = block_dims.x * block_dims.y * block_dims.z;
  // int available_blocks_per_sm =
  //     port::MathUtil::CeilOfRatio(block_count, multiprocessor_count);
  // if (available_blocks_per_sm <= static_cast<int64>(blocks_per_sm)) {
  //   VLOG(2) << "Occupancy is limited by number of blocks available per sm.";
  //   return;
  // }

  // uint64 improved_regs_per_thread = CalculateRegisterLimitForTargetOccupancy(
  //     device_description, smem_per_block, thread_dims, blocks_per_sm + 1);
  // if (improved_regs_per_thread != 0) {
  //   VLOG(2) << "Reducing register usage from " << regs_per_thread
  //           << " to " << improved_regs_per_thread
  //           << " could increase resident blocks per SM by one.";

  //   uint64 reg_reduction = regs_per_thread - improved_regs_per_thread;
  //   if (reg_reduction <=
  //       static_cast<uint64>(FLAGS_register_occupancy_warning_threshold)) {
  //     LOG(INFO) << "Notice: occupancy would increase if register usage was"
  //               << " reduced from " << regs_per_thread
  //               << " to " << improved_regs_per_thread
  //               << " registers per thread for kernel: "
  //               << kernel.demangled_name();
  //   }
  // } else {
  //   VLOG(2) << "Resident blocks per SM cannot be increased by reducing "
  //       "register usage.";
  // }
}

void *CLExecutor::Allocate(uint64 size) {
  // std::cout << "cl_gpu_executor::Allocate()" << std::endl;
  return CLDriver::DeviceAllocate(context_, size);
}

void *CLExecutor::AllocateSubBuffer(DeviceMemoryBase *mem,
                                      uint64 offset_bytes, uint64 size_bytes) {
  std::cout << "cl_gpu_executor::AllocateSubBuffer()" << std::endl;
  // offset and size are in bytes, so char* works as the pointer type.
  // return reinterpret_cast<char *>(mem->opaque()) + offset_bytes;
  return 0;
}

void CLExecutor::Deallocate(DeviceMemoryBase *mem) {
  // std::cout << "cl_gpu_executor::Deallocate()" << std::endl;
  // CL "sub-buffers" are just pointer + offset, so no dealloc is necessary.
  if (!mem->is_sub_buffer()) {
     CLDriver::DeviceDeallocate(context_, mem->opaque());
  }
}

bool CLExecutor::HostMemoryRegister(void *location, uint64 size) {
  std::cout << "cl_gpu_executor::HostMemoryRegister()" << std::endl;
  return false;
  // if (location == nullptr || size == 0) {
  //   LOG(WARNING) << "attempting to register null or zero-sized memory: "
  //                << location << "; size " << size;
  // }
  // VLOG(2) << "registering " << location << " size " << size;
  // return CLDriver::HostRegister(context_, location, size);
}

bool CLExecutor::HostMemoryUnregister(void *location) {
  std::cout << "cl_gpu_executor::HostMemoryUnregister()" << std::endl;
  return false;
  // VLOG(2) << "unregistering " << location;
  // return CLDriver::HostUnregister(context_, location);
}

bool CLExecutor::SynchronizeAllActivity() {
  // std::cout << "cl_gpu_executor::SynchronizeAllActivity()" << std::endl;
  return CLDriver::SynchronizeContext(context_);
}

bool CLExecutor::SynchronousMemZero(DeviceMemoryBase *location, uint64 size) {
  // std::cout << "cl_gpu_executor::SynchronousMemZero()" << std::endl;
  // return false;
  if (reinterpret_cast<uintptr_t>(location->opaque()) % 4 == 0 &&
      size % 4 == 0) {
    return CLDriver::SynchronousMemsetUint32(
        context_, AsClDevicePtr(location), 0x0, size / 4);
  }
  return CLDriver::SynchronousMemsetUint8(context_, AsClDevicePtr(location),
                                            0x0, size);
}

bool CLExecutor::SynchronousMemSet(DeviceMemoryBase *location, int value,
                                     uint64 size) {
  std::cout << "cl_gpu_executor::SynchronousMemSet()" << std::endl;
  return false;
  // if (reinterpret_cast<uintptr_t>(location->opaque()) % 4 == 0 &&
  //     size % 4 == 0) {
  //   // clMemset reinterprets "value" as a uint8.
  //   uint8 byte_value = static_cast<uint8>(value);
  //   uint32 pattern = (byte_value << 24) | (byte_value << 16) |
  //                    (byte_value << 8) | byte_value;
  //   return CLDriver::SynchronousMemsetUint32(
  //       context_, AsClDevicePtr(location), pattern, size / 4);
  // }
  // return CLDriver::SynchronousMemsetUint8(context_, AsClDevicePtr(location),
  //                                           value, size);
}

bool CLExecutor::SynchronousMemcpy(DeviceMemoryBase *gpu_dst,
                                     const void *host_src, uint64 size) {
  // std::cout << "cl_gpu_executor::SynchronousMemcpy()" << std::endl;
  return CLDriver::SynchronousMemcpyH2D(context_, AsClDevicePtr(gpu_dst),
                                        host_src, size);
}

bool CLExecutor::SynchronousMemcpy(void *host_dst,
                                     const DeviceMemoryBase &gpu_src,
                                     uint64 size) {
  // std::cout << "cl_gpu_executor::SynchronousMemcpy()" << std::endl;
  return CLDriver::SynchronousMemcpyD2H(context_, host_dst,
                                        AsClDevicePtr(gpu_src), size);
}

bool CLExecutor::SynchronousMemcpyDeviceToDevice(
    DeviceMemoryBase *gpu_dst, const DeviceMemoryBase &gpu_src, uint64 size) {
  // std::cout << "cl_gpu_executor::SynchronousMemcpyDeviceToDevice()" << std::endl;
  return CLDriver::SynchronousMemcpyD2D(context_, AsClDevicePtr(gpu_dst),
                                        AsClDevicePtr(gpu_src), size);
}

bool CLExecutor::MemZero(Stream *stream, DeviceMemoryBase *location,
                           uint64 size) {
  std::cout << "cl_gpu_executor::MemZero()" << std::endl;
  return false;
  // if (reinterpret_cast<uintptr_t>(location->opaque()) % 4 == 0 &&
  //     size % 4 == 0) {
  //   return Memset32(stream, location, 0x0, size);
  // } else {
  //   return Memset(stream, location, 0x0, size);
  // }
}

bool CLExecutor::Memset(Stream *stream, DeviceMemoryBase *location,
                           uint8 pattern, uint64 size) {
  std::cout << "cl_gpu_executor::Memset()" << std::endl;
  return false;
  // VLOG(2) << "enqueueing memset8 operation onto stream " << stream
  //         << " at location " << location << " with size " << size
  //         << " and pattern " << std::hex << pattern;
  // return CLDriver::AsynchronousMemsetUint8(
  //     context_, AsClDevicePtr(location), pattern, size,
  //     AsCLStreamValue(stream));
}

bool CLExecutor::Memset32(Stream *stream, DeviceMemoryBase *location,
                            uint32 pattern, uint64 size) {
  std::cout << "cl_gpu_executor::Memset32()" << std::endl;
  return false;
  // VLOG(2) << "enqueueing memset32 operation onto stream " << stream
  //         << " at location " << location << " with size " << size
  //         << " and pattern " << std::hex << pattern;
  // CHECK(reinterpret_cast<uintptr_t>(location->opaque()) % 4 == 0 &&
  //       size % 4 == 0);
  // return CLDriver::AsynchronousMemsetUint32(
  //     context_, AsClDevicePtr(location), pattern, size / 4,
  //     AsCLStreamValue(stream));
}

bool CLExecutor::Memcpy(Stream *stream, void *host_dst,
                          const DeviceMemoryBase &gpu_src, uint64 size) {
  // std::cout << "cl_gpu_executor::Memcpy()" << std::endl;
  return CLDriver::AsynchronousMemcpyD2H(context_, host_dst,
                                           AsClDevicePtr(gpu_src), size,
                                           AsCLStreamValue(stream));
}

bool CLExecutor::Memcpy(Stream *stream, DeviceMemoryBase *gpu_dst,
                          const void *host_src, uint64 size) {
  // std::cout << "cl_gpu_executor::Memcpy()" << std::endl;
  return CLDriver::AsynchronousMemcpyH2D(context_, AsClDevicePtr(gpu_dst),
                                           host_src, size,
                                           AsCLStreamValue(stream));
}

bool CLExecutor::MemcpyDeviceToDevice(Stream *stream,
                                        DeviceMemoryBase *gpu_dst,
                                        const DeviceMemoryBase &gpu_src,
                                        uint64 size) {
  std::cout << "cl_gpu_executor::MemcpyDevicetoDevice()" << std::endl;
  return false;
  // return CLDriver::AsynchronousMemcpyD2D(context_, AsClDevicePtr(gpu_dst),
  //                                          AsClDevicePtr(gpu_src), size,
  //                                          AsCLStreamValue(stream));
}

bool CLExecutor::HostCallback(Stream *stream,
                                std::function<void()> callback) {
  std::cout << "cl_gpu_executor::HostCallback()" << std::endl;
  return false;
  // auto callback_ptr = new std::function<void()>(callback);
  // return CLDriver::AddStreamCallback(context_, AsCLStreamValue(stream),
  //                                      InternalHostCallback, callback_ptr);
}

/* static */ void CLExecutor::InternalHostCallback(CUstream stream,
                                                     CUresult status,
                                                     void *data) {
  std::cout << "cl_gpu_executor::InternalHostCallback()" << std::endl;
  // std::function<void()> *callback =
  //     reinterpret_cast<std::function<void()> *>(data);
  // (*callback)();
  // delete callback;
}

port::Status CLExecutor::AllocateEvent(Event *event) {
  // std::cout << "cl_gpu_executor::AllocateEvent()" << std::endl;
  return AsCLEvent(event)->Init();
  }

port::Status CLExecutor::DeallocateEvent(Event *event) {
  // std::cout << "cl_gpu_executor::DeallocateEvent()" << std::endl;
  return AsCLEvent(event)->Destroy();
}

port::Status CLExecutor::RecordEvent(Stream *stream, Event *event) {
  // std::cout << "cl_gpu_executor::RecordEvent()" << std::endl;
 return AsCLEvent(event)->Record(AsCLStream(stream));
}

port::Status CLExecutor::WaitForEvent(Stream *stream, Event *event) {
  // std::cout << "cl_gpu_executor::WaitForEvent()" << std::endl;
  if (CLDriver::WaitStreamOnEvent(context_,
                                    AsCLStream(stream)->cl_stream(),
                                    AsCLEvent(event)->cl_event())) {
    return port::Status::OK();
  } else {
    return port::Status{
        port::error::INTERNAL,
        port::Printf("error recording waiting for CL event on stream %p",
                     stream)};
  }
}

Event::Status CLExecutor::PollForEventStatus(Event *event) {
  // std::cout << "cl_gpu_executor::PollForEventSTatus()" << std::endl;
  return AsCLEvent(event)->PollForStatus();
}

bool CLExecutor::AllocateStream(Stream *stream) {
  // std::cout << "cl_gpu_executor::AllocateStream()" << std::endl;
  return AsCLStream(stream)->Init();
}

void CLExecutor::DeallocateStream(Stream *stream) {
  // std::cout << "cl_gpu_executor::DeallocateStream()" << std::endl;
  CLStream *cl_stream = AsCLStream(stream);
  if (!cl_stream->IsIdle()) {
    LOG(ERROR) << "Deallocating stream with pending work";
  }
  cl_stream->Destroy();
}

bool CLExecutor::AllocateTimer(Timer *timer) {
  std::cout << "cl_gpu_executor::AllocateTimer()" << std::endl;
  return false;
  // return AsCLTimer(timer)->Init();
}

void CLExecutor::DeallocateTimer(Timer *timer) {
  std::cout << "cl_gpu_executor::DeallocateTimer()" << std::endl;
  // AsCLTimer(timer)->Destroy();
}

bool CLExecutor::CreateStreamDependency(Stream *dependent, Stream *other) {
  // std::cout << "cl_gpu_executor::CreateStreamDependency()" << std::endl;
  CUevent other_completed_event = *AsCLStream(other)->completed_event();
  bool ok = CLDriver::RecordEvent(context_, other_completed_event,
                                    AsCLStreamValue(other))
      .ok();
  if (!ok) {
    LOG(ERROR) << "failed to record completion event; "
                  "therefore, failed to create inter-stream dependency";
    return false;
  }

  return CLDriver::WaitStreamOnEvent(context_, AsCLStreamValue(dependent),
                                       other_completed_event);
}

bool CLExecutor::StartTimer(Stream *stream, Timer *timer) {
  std::cout << "cl_gpu_executor::StartTimer()" << std::endl;
  return false;
  // return AsCLTimer(timer)->Start(AsCLStream(stream));
}

bool CLExecutor::StopTimer(Stream *stream, Timer *timer) {
  std::cout << "cl_gpu_executor::StopTimer()" << std::endl;
  return false;
  // return AsCLTimer(timer)->Stop(AsCLStream(stream));
}

bool CLExecutor::BlockHostUntilDone(Stream *stream) {
  std::cout << "cl_gpu_executor::BlockHostUntilDone()" << std::endl;
  return false;
  // return CLDriver::SynchronizeStream(context_, AsCLStreamValue(stream));
}

blas::BlasSupport *CLExecutor::CreateBlas() {
  // std::cout << "cl_gpu_executor::CreateBlas()" << std::endl;
  PluginRegistry *registry = PluginRegistry::Instance();
  port::StatusOr<PluginRegistry::BlasFactory> status =
      registry->GetFactory<PluginRegistry::BlasFactory>(kClPlatformId,
                                                        plugin_config_.blas());
  if (!status.ok()) {
    LOG(ERROR) << "Unable to retrieve BLAS factory: "
               << status.status().error_message();
    return nullptr;
  }
  // std::cout << "cl_gpu_executor::CreateBlas() => created blas ok :-)" << std::endl;

  return status.ValueOrDie()(this);
}

dnn::DnnSupport *CLExecutor::CreateDnn() {
  std::cout << "cl_gpu_executor::Creatednn()" << std::endl;
  // PluginRegistry *registry = PluginRegistry::Instance();
  // port::StatusOr<PluginRegistry::DnnFactory> status =
  //     registry->GetFactory<PluginRegistry::DnnFactory>(kClPlatformId,
  //                                                      plugin_config_.dnn());
  // if (!status.ok()) {
  //   LOG(ERROR) << "Unable to retrieve DNN factory: "
  //              << status.status().error_message();
    return nullptr;
  // }

  // return status.ValueOrDie()(this);
}

fft::FftSupport *CLExecutor::CreateFft() {
  std::cout << "cl_gpu_executor::CreateFft()" << std::endl;
  // PluginRegistry *registry = PluginRegistry::Instance();
  // port::StatusOr<PluginRegistry::FftFactory> status =
  //     registry->GetFactory<PluginRegistry::FftFactory>(kClPlatformId,
  //                                                      plugin_config_.fft());
  // if (!status.ok()) {
  //   LOG(ERROR) << "Unable to retrieve FFT factory: "
  //              << status.status().error_message();
    return nullptr;
  // }

  // return status.ValueOrDie()(this);
}

rng::RngSupport *CLExecutor::CreateRng() {
  std::cout << "cl_gpu_executor::CreateRng()" << std::endl;
  // PluginRegistry *registry = PluginRegistry::Instance();
  // port::StatusOr<PluginRegistry::RngFactory> status =
  //     registry->GetFactory<PluginRegistry::RngFactory>(kClPlatformId,
  //                                                      plugin_config_.rng());
  // if (!status.ok()) {
  //   LOG(ERROR) << "Unable to retrieve RNG factory: "
  //              << status.status().error_message();
    return nullptr;
  // }

  // return status.ValueOrDie()(this);
}

// TODO(rspringer): Remove in b/18544742.
bool CLExecutor::SupportsDnn() const {
  // std::cout << "CLExecutor::SupportsDnn" << std::endl;
  return false;
}

bool CLExecutor::CanEnablePeerAccessTo(StreamExecutorInterface *other) {
  // std::cout << "CLExecutor::CanEnablePeerAccessTo" << std::endl;
  return false;
  // CLExecutor *cl_other = static_cast<CLExecutor *>(other);
  // return CLDriver::CanEnablePeerAccess(context_, cl_other->context_);
}

port::Status CLExecutor::EnablePeerAccessTo(StreamExecutorInterface *other) {
  std::cout << "CLExecutor::EnablePeerAccessTo" << std::endl;
    return port::Status{
        port::error::INTERNAL,
        port::Printf("not implemented")};
  // CLExecutor *cl_other = static_cast<CLExecutor *>(other);
  // return CLDriver::EnablePeerAccess(context_, cl_other->context_);
}

SharedMemoryConfig CLExecutor::GetDeviceSharedMemoryConfig() {
  // std::cout << "CLExecutor::GetDeviceSharedMemoryConfig" << std::endl;
  // port::StatusOr<CUsharedconfig> cl_config =
  //     CLDriver::ContextGetSharedMemConfig(context_);
  // if (!cl_config.ok()) {
    // Don't log; the failed call will log necessary output.
    return SharedMemoryConfig::kDefault;
  // }

  // switch (cl_config.ValueOrDie()) {
  //   case CU_SHARED_MEM_CONFIG_DEFAULT_BANK_SIZE:
  //     return SharedMemoryConfig::kDefault;
  //   case CU_SHARED_MEM_CONFIG_FOUR_BYTE_BANK_SIZE:
  //     return SharedMemoryConfig::kFourByte;
  //   case CU_SHARED_MEM_CONFIG_EIGHT_BYTE_BANK_SIZE:
  //     return SharedMemoryConfig::kEightByte;
  //   default:
  //     LOG(FATAL) << "Invalid shared memory configuration returned: "
  //                << cl_config.ValueOrDie();
  // }
}

port::Status CLExecutor::SetDeviceSharedMemoryConfig(
    SharedMemoryConfig config) {
  std::cout << "CLExecutor::SetDeviceSharedMemoryConfig" << std::endl;
    return port::Status{
        port::error::INTERNAL,
        port::Printf("not implemented")};

  // CUsharedconfig cl_config;
  // switch (config) {
  //   case SharedMemoryConfig::kDefault:
  //     cl_config = CU_SHARED_MEM_CONFIG_DEFAULT_BANK_SIZE;
  //     break;
  //   case SharedMemoryConfig::kFourByte:
  //     cl_config = CU_SHARED_MEM_CONFIG_FOUR_BYTE_BANK_SIZE;
  //     break;
  //   case SharedMemoryConfig::kEightByte:
  //     cl_config = CU_SHARED_MEM_CONFIG_EIGHT_BYTE_BANK_SIZE;
  //     break;
  //   default:
  //     LOG(FATAL) << "Invalid shared memory configuration specified: "
  //                << static_cast<int>(config);
  // // }
  // return CLDriver::ContextSetSharedMemConfig(context_, cl_config);
}

bool CLExecutor::DeviceMemoryUsage(int64 *free, int64 *total) const {
  // std::cout << "CLExecutor::DeviceMemoryUsage" << std::endl;
  //return false;
  return CLDriver::GetDeviceMemoryInfo(context_, free, total);
}

bool CLExecutor::GetSymbol(const string& symbol_name, void **mem,
                             size_t *bytes) {
  std::cout << "CLExecutor::GetSymbol" << std::endl;
  return false;
  // {  // give limited scope to mutex_lock
  //   mutex_lock lock{disk_modules_mu_};
  //   for (auto &it : disk_modules_) {
  //     if (CLDriver::GetModuleSymbol(context_, it.second, symbol_name.c_str(),
  //                                     reinterpret_cast<CUdeviceptr *>(mem),
  //                                     bytes)) {
  //       return true;
  //     }
  //   }
  // }

  // {  // give limited scope to mutex_lock
  //   mutex_lock lock{in_memory_modules_mu_};
  //   for (auto &it : in_memory_modules_) {
  //     if (CLDriver::GetModuleSymbol(context_, it.second, symbol_name.c_str(),
  //                                     reinterpret_cast<CUdeviceptr *>(mem),
  //                                     bytes)) {
  //       return true;
  //     }
  //   }
  // }

  // LOG(INFO) << "Falied to find symbol in any modules: " << symbol_name;
  // return false;
}

bool CLExecutor::FillBlockDimLimit(BlockDim *block_dim_limit) const {
  // std::cout << "CLExecutor::FillBlockDimLimit" << std::endl;
  // The BlockDim name is a mismatch against these GRID_DIM_* queries because
  // we use BlockDims to express the dimensions of blocks within a grid
  // (as opposed to ThreadDim which expresses the dimensions of threads
  // within a block).
  int x, y, z;
  if (!CLDriver::GetGridLimits(&x, &y, &z, device_)) {
    return false;
  }

  block_dim_limit->x = x;
  block_dim_limit->y = y;
  block_dim_limit->z = z;
  return true;
}

KernelArg CLExecutor::DeviceMemoryToKernelArg(
    const DeviceMemoryBase &gpu_mem) const {
  // std::cout << "CLExecutor::DeviceMemoryToKernelArg" << std::endl;
  const void* arg = gpu_mem.opaque();
  const uint8 *arg_ptr = reinterpret_cast<const uint8 *>(&arg);

  KernelArg kernel_arg;
  kernel_arg.type = KernelArg::kNormal;
  kernel_arg.data = port::InlinedVector<uint8, 4>(arg_ptr, arg_ptr + sizeof(arg));
  kernel_arg.bytes = sizeof(arg);
  return kernel_arg;
}

bool CLExecutor::SupportsBlas() const {
  std::cout << "CLExecutor::SupportsBlas" << std::endl;
  return true;
}

bool CLExecutor::SupportsFft() const {
  std::cout << "CLExecutor::SupportsFft" << std::endl;
  return false;
}

bool CLExecutor::SupportsRng() const {
  std::cout << "CLExecutor::SupportsRng" << std::endl;
  return false;
}

std::unique_ptr<internal::EventInterface>
CLExecutor::CreateEventImplementation() {
  // std::cout << "CLExecutor::CreateEventImplementation" << std::endl;
  return std::unique_ptr<internal::EventInterface>(new CLEvent(this));
}

std::unique_ptr<internal::KernelInterface>
CLExecutor::CreateKernelImplementation() {
  std::cout << "CLExecutor::CreateKernelImplementation" << std::endl;
  // return std::unique_ptr<internal::KernelInterface>();
  return std::unique_ptr<internal::KernelInterface>(new CLKernel());
}

std::unique_ptr<internal::StreamInterface>
CLExecutor::GetStreamImplementation() {
  // std::cout << "CLExecutor::GetStreamImplementation" << std::endl;
  return std::unique_ptr<internal::StreamInterface>(new CLStream(this));
}

std::unique_ptr<internal::TimerInterface>
CLExecutor::GetTimerImplementation() {
  // std::cout << "CLExecutor::GetTimerImplementation" << std::endl;
  return std::unique_ptr<internal::TimerInterface>();
  // return std::unique_ptr<internal::TimerInterface>(new CLTimer(this));
}

void *CLExecutor::CudaContextHack() { 
  // std::cout << "CLExecutor::CudaContextHack" << std::endl;
  return context_;
}

ClContext* CLExecutor::cl_context() { 
  // std::cout << "CLExecutor::cl_context" << std::endl;
  return context_; 
}

// Attemps to read the NUMA node corresponding to the GPU device's PCI bus out
// of SysFS. Returns -1 if it cannot.
//
// For anything more complicated/prod-focused than this, you'll likely want to
// turn to gsys' topology modeling.
static int TryToReadNumaNode(const string &pci_bus_id, int device_ordinal) {
  // std::cout << "CLExecutor::TryToReadNumaNode" << std::endl;
// #if defined(__APPLE__)
  //LOG(INFO) << "OS X does not support NUMA - returning NUMA node zero";
  return 0;
// #else
//   VLOG(2) << "trying to read NUMA node for device ordinal: " << device_ordinal;
//   static const int kUnknownNumaNode = -1;

//   if (pci_bus_id.empty()) {
//     LOG(INFO) << "no PCI bus ID for device ordinal: " << device_ordinal;
//     return kUnknownNumaNode;
//   }

//   string filename =
//       port::Printf("/sys/bus/pci/devices/%s/numa_node", pci_bus_id.c_str());

//   // We have to use fopen/fread here so that the device properties can be
//   // populated before InitGoogle procedure has been completed (at which point we
//   // could use the file::* utilities).
//   FILE *file = fopen(filename.c_str(), "r");
//   if (file == nullptr) {
//     LOG(ERROR) << "could not open file to read NUMA node: " << filename
//                << "\nYour kernel may have been built without NUMA support.";
//     return kUnknownNumaNode;
//   }

//   string content;
//   char buf[32];
//   size_t did_read = fread(buf, sizeof(buf[0]), sizeof(buf) - 1, file);
//   buf[did_read] = '\0';
//   content = buf;

//   int32 value;
//   if (port::safe_strto32(content, &value)) {
//     if (value < 0) {  // See http://b/18228951 for details on this path.
//       LOG(INFO) << "successful NUMA node read from SysFS had negative value ("
//                 << value << "), but there must be at least one NUMA node"
//                             ", so returning NUMA node zero";
//       fclose(file);
//       return 0;
//     }
//     fclose(file);
//     return value;
//   }

//   LOG(WARNING)
//       << "could not convert SysFS file contents to integral NUMA node value: "
//       << content;

//   fclose(file);
//   return kUnknownNumaNode;
// #endif
}

// Set of compute capability specific device parameters that cannot be
// queried from the driver API.  These values instead are baked into a
// lookup table indexed by compute capability version.
struct UnqueryableDeviceParams {
  int cc_major;
  int cc_minor;
  uint64 blocks_per_core_limit;
  uint64 registers_per_core_limit;
  uint64 registers_per_thread_limit;
  uint64 warp_alloc_granularity;
  uint64 register_alloc_granularity;
  uint64 shared_memory_alloc_granularity;
};

static const UnqueryableDeviceParams kAllUnqueryableDeviceParams[] = {
  {
    3, 5,       // compute capability (3.5)
    16,         // blocks_per_core_limit
    64 * 1024,  // registers_per_core_limit
    255,        // registers_per_thread_limit
    4,          // warp_alloc_granularity
    256,        // register_alloc_granularity
    256         // shared_memory_alloc_granularity
  }
};

DeviceDescription *CLExecutor::PopulateDeviceDescription() const {
  // std::cout << "CLExecutor::PopulateDeviceDescription" << std::endl;
  internal::DeviceDescriptionBuilder builder;
  {
    int driver_version = 0;
    (void)CLDriver::GetDriverVersion(&driver_version);
    string augmented_driver_version = port::Printf(
        "%d (%s)", driver_version,
        DriverVersionStatusToString(Diagnostician::FindDsoVersion()).c_str());
    builder.set_driver_version(augmented_driver_version);
  }

  {
    string pci_bus_id = CLDriver::GetPCIBusID(device_);

    // Lower the hex characters to match sysfs.
    pci_bus_id = port::Lowercase(pci_bus_id);
    builder.set_pci_bus_id(pci_bus_id);

    // Read the NUMA node corresponding to the PCI bus ID out of sysfs.
    int numa_node = TryToReadNumaNode(pci_bus_id, device_ordinal_);
    builder.set_numa_node(numa_node);
  }

  CUdevprop prop;
  if (CLDriver::GetDeviceProperties(&prop, device_ordinal_)) {
    // std::cout << "getdeviceproperties succeeded" << std::endl;
    builder.set_threads_per_block_limit(prop.maxThreadsPerBlock);

    ThreadDim thread_dim_limit;
    thread_dim_limit.x = prop.maxThreadsDim[0];
    thread_dim_limit.y = prop.maxThreadsDim[1];
    thread_dim_limit.z = prop.maxThreadsDim[2];
    builder.set_thread_dim_limit(thread_dim_limit);

    float clock_rate_ghz = static_cast<float>(prop.clockRate) / 1e6;
    builder.set_clock_rate_ghz(clock_rate_ghz);
  }

  {
    bool ecc_enabled = false;
    (void)CLDriver::IsEccEnabled(device_, &ecc_enabled);
    builder.set_ecc_enabled(ecc_enabled);
  }

  {
    uint64 device_memory_size = -1;
    (void)CLDriver::GetDeviceTotalMemory(device_, &device_memory_size);
    // std::cout << "gl_gpu_executor.cc getdevicetotalemmory " << device_memory_size << std::endl;
    builder.set_device_memory_size(device_memory_size);
  }

  {
    BlockDim block_dim_limit;
    FillBlockDimLimit(&block_dim_limit);
    builder.set_block_dim_limit(block_dim_limit);
  }

  {
    string device_name;
    (void)CLDriver::GetDeviceName(device_, &device_name);
    builder.set_name(device_name);
  }

  for (size_t i = 0; i < ARRAYSIZE(kAllUnqueryableDeviceParams); i++) {
    const auto &params = kAllUnqueryableDeviceParams[i];
    if (params.cc_major == cc_major_ && params.cc_minor == cc_minor_) {
      builder.set_blocks_per_core_limit(params.blocks_per_core_limit);
      builder.set_registers_per_core_limit(params.registers_per_core_limit);
      builder.set_registers_per_thread_limit(params.registers_per_thread_limit);
      builder.set_warp_alloc_granularity(params.warp_alloc_granularity);
      builder.set_register_alloc_granularity(params.register_alloc_granularity);
      builder.set_shared_memory_alloc_granularity(
          params.shared_memory_alloc_granularity);
    }
  }

  builder.set_platform_version(
      port::StrCat("Compute Capability ", cc_major_, ".", cc_minor_));

  // TODO(leary) should be a way to query this from the driver, but this is
  // unlikely to change for us any time soon.
  builder.set_device_address_bits(64);

  // builder.set_device_vendor("NVIDIA Corporation");
  // builder.set_cl_compute_capability(cc_major_, cc_minor_);
  // std::cout << "get max shared memory per core" << std::endl;
  builder.set_shared_memory_per_core(
      CLDriver::GetMaxSharedMemoryPerCore(device_).ValueOrDie());
  // std::cout << "deviceprops 1" << std::endl;
  builder.set_shared_memory_per_block(
      CLDriver::GetMaxSharedMemoryPerBlock(device_).ValueOrDie());
  // std::cout << "deviceprops 2" << std::endl;
  builder.set_core_count(
      CLDriver::GetMultiprocessorCount(device_).ValueOrDie());
  // std::cout << "deviceprops 3" << std::endl;
  builder.set_threads_per_core_limit(
      CLDriver::GetMaxThreadsPerMultiprocessor(device_).ValueOrDie());
  // std::cout << "deviceprops 4" << std::endl;
  builder.set_registers_per_block_limit(
      CLDriver::GetMaxRegistersPerBlock(device_).ValueOrDie());
  // std::cout << "deviceprops 5" << std::endl;
  builder.set_threads_per_warp(
      CLDriver::GetThreadsPerWarp(device_).ValueOrDie());
  // std::cout << "cl_gpu_executor.cc calling builder.Build()" << std::endl;
  auto built = builder.Build();
  // std::cout << "cl_gpu_executor.cc called builder.Build()" << std::endl;
  return built.release();
}

}  // namespace cl

namespace gpu = ::perftools::gputools;

void initialize_cl_gpu_executor() {
  // std::cout << "cl_gpu_executor.cc initialize_cl_gpu_executor()" << std::endl;
  // port::StatusOr<void *> status =
  //     gpu::internal::CachedDsoLoader::GetLibclDsoHandle();
  // if (!status.ok()) {
  //   gpu::cl::Diagnostician::LogDriverVersionInformation();
  //   LOG(INFO) << "LD_LIBRARY_PATH: " << getenv("LD_LIBRARY_PATH");
  //   LOG(INFO) << "failed to find libcl.so on this system: "
  //             << status.status();
  // }

  // // TODO(b/22689637): Temporary until users are migrated off of PlatformKind.
   gpu::PluginRegistry::Instance()->MapPlatformKindToId(
       gpu::PlatformKind::kCl, gpu::cl::kClPlatformId);

   *gpu::internal::MakeCLExecutorImplementation() = [](
       const gpu::PluginConfig &config) {
     return new gpu::cl::CLExecutor{config};
   };
}

}  // namespace gputools
}  // namespace perftools

// REGISTER_MODULE_INITIALIZER(
//     cl_gpu_executor, {perftools::gputools::initialize_cl_gpu_executor();});
