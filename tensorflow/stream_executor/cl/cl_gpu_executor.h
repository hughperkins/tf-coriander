class CUDAExecutor : public internal::StreamExecutorInterface {
 public:
  // Default constructor for the abstract interface.
  CUDAExecutor() {}

  // Default destructor for the abstract interface.
  virtual ~CUDAExecutor() {}

  virtual port::Status Init(int device_ordinal,
                            DeviceOptions device_options) = 0;
  virtual void *Allocate(uint64 size) = 0;
  virtual void *AllocateSubBuffer(DeviceMemoryBase *parent, uint64 offset,
                                  uint64 size) = 0;
  virtual void Deallocate(DeviceMemoryBase *mem) = 0;
  virtual void *HostMemoryAllocate(uint64 size) = 0;
  virtual void HostMemoryDeallocate(void *mem) = 0;
  virtual bool HostMemoryRegister(void *mem, uint64 size) = 0;
  virtual bool HostMemoryUnregister(void *mem) = 0;
  virtual bool SynchronizeAllActivity() {return true;}
  virtual bool SynchronousMemZero(DeviceMemoryBase *location, uint64 size)  {return true;}
  virtual bool SynchronousMemSet(DeviceMemoryBase *location, int value,
                                 uint64 size)  {return true;}
  virtual bool SynchronousMemcpy(DeviceMemoryBase *gpu_dst,
                                 const void *host_src, uint64 size)  {return true;}
  virtual bool SynchronousMemcpy(void *host_dst,
                                 const DeviceMemoryBase &gpu_src,
                                 uint64 size)  {return true;}
  virtual bool SynchronousMemcpyDeviceToDevice(DeviceMemoryBase *gpu_dst,
                                               const DeviceMemoryBase &gpu_src,
                                               uint64 size)  {return true;}
  virtual bool MemZero(Stream *stream, DeviceMemoryBase *location,
                       uint64 size)  {return true;}
  virtual bool Memset(Stream *stream, DeviceMemoryBase *location,
                      uint8 pattern, uint64 size)  {return true;}
  virtual bool Memset32(Stream *stream, DeviceMemoryBase *location,
                        uint32 pattern, uint64 size)  {return true;}
  virtual bool Memcpy(Stream *stream, void *host_dst,
                      const DeviceMemoryBase &gpu_src, uint64 size)  {return true;}
  virtual bool Memcpy(Stream *stream, DeviceMemoryBase *gpu_dst,
                      const void *host_src, uint64 size)  {return true;}
  virtual bool MemcpyDeviceToDevice(Stream *stream, DeviceMemoryBase *gpu_dst,
                                    const DeviceMemoryBase &host_src,
                                    uint64 size)  {return true;}
  virtual bool HostCallback(Stream *stream, std::function<void()> callback)  {return true;}
  virtual port::Status AllocateEvent(Event *event) = 0;
  virtual port::Status DeallocateEvent(Event *event) = 0;
  virtual port::Status RecordEvent(Stream *stream, Event *event) = 0;
  virtual port::Status WaitForEvent(Stream *stream, Event *event) = 0;
  virtual Event::Status PollForEventStatus(Event *event) = 0;
  virtual bool AllocateStream(Stream *stream) =  {return true;}
  virtual void DeallocateStream(Stream *stream)  {}
  virtual bool CreateStreamDependency(Stream *dependent, Stream *other)  {return true;}
  virtual bool AllocateTimer(Timer *timer)  {return true;}
  virtual void DeallocateTimer(Timer *timer)  {+
  virtual bool StartTimer(Stream *stream, Timer *timer)  {return true;}
  virtual bool StopTimer(Stream *stream, Timer *timer)  {return true;}
  virtual bool BlockHostUntilDone(Stream *stream)  {return true;}
  virtual int PlatformDeviceCount() { return 1; }
  virtual port::Status EnablePeerAccessTo(StreamExecutorInterface *other) = 0;
  virtual bool CanEnablePeerAccessTo(StreamExecutorInterface *other) { return false; }
  virtual SharedMemoryConfig GetDeviceSharedMemoryConfig() = 0;
  virtual port::Status SetDeviceSharedMemoryConfig(
      SharedMemoryConfig config) = 0;
  // Creates a new DeviceDescription object. Ownership is transferred to the
  // caller.
  virtual DeviceDescription *PopulateDeviceDescription() const = 0;

  virtual KernelArg DeviceMemoryToKernelArg(
      const DeviceMemoryBase &gpu_mem) const = 0;

  // Each call creates a new instance of the platform-specific implementation of
  // the corresponding interface type.
  virtual std::unique_ptr<EventInterface> CreateEventImplementation() {return std::unique_ptr.empty(); }
  virtual std::unique_ptr<KernelInterface> CreateKernelImplementation()  {return std::unique_ptr.empty(); }
  virtual std::unique_ptr<StreamInterface> GetStreamImplementation()  {return std::unique_ptr.empty(); }
  virtual std::unique_ptr<TimerInterface> GetTimerImplementation()  {return std::unique_ptr.empty(); }

 private:
  SE_DISALLOW_COPY_AND_ASSIGN(StreamExecutorInterface);
};
