# Mutex issue

This doc documents things I find, as I try to solve an issue with mutex initialization on Mac, https://github.com/hughperkins/tensorflow-cl/issues/11

```
$ python -c 'import tensorflow'
libc++abi.dylib: terminating with uncaught exception of type std::__1::system_error: mutex lock failed: Invalid argument
Abort trap: 6
```

Some initial analysis, using `cout`, shows that this appears to happen at the time of the call to `mutex.lock()`, in `plugin_registry.cc`:

```
/* static */ PluginRegistry* PluginRegistry::Instance() {
  std::cout << "plugin_registry.cc PluginRegistry::Instance() v0.2" << std::endl;
  mutex_lock lock{mu_};
  std::cout << "plugin_registry.cc PluginRegistry::Instance() locked mutex" << std::endl;
  if (instance_ == nullptr) {
    instance_ = new PluginRegistry();
  }
  return instance_;
}
```
... gives the following output:
```
plugin_registry.cc PluginRegistry::Instance() v0.2
libc++abi.dylib: terminating with uncaught exception of type std::__1::system_error: mutex lock failed: Invalid argument
./run1: line 5: 93053 Abort trap: 6           python -c 'import tensorflow'
```
=> suggests that it crashes on the `.lock()`.

Adding code to the constructors of the two `mutex` classes, and the two `mutex_lock` classes, I can find, ie in:
- `core/platform/default/mutex.h`, and
- `stream_executor/platform/mutex.h`

... produces no additional output. Therefore it looks like the constructors of these objects are not being called.  So, it looks like some issue with the order of initialization of the various static initializers.  Given that the problem looks complex, going to start documenting the architecture etc here.

## General architecture of `platform` files

Directory layout:

```
tensorflow/
  core/
    platform/
      mutex.h => default/mutex.h
      thread_annotations.h => default/thread_annotations.h
      default/
        mutex.h:
          definitions for:
            class mutex
            class mutex_lock
            enum LinkedInitialized{ LINKER_INITIALIZED };
        thread_annotations.h:
          bunch of attribute macros, that boil down to __attribute__(something)
  stream_executor/
    platform/
      default/
        mutex.h:
          definitions for:
            class mutex
            class mutex_lock
```

Note that `#if (__cplusplus >= 201402L)` appears to evaluate to false, in a separate test program, thus implying that `STREAM_EXECUTOR_USE_SHARED_MUTEX` is false too.

## Module registration

In eg `cl_blas.cc`:
```
REGISTER_MODULE_INITIALIZER(register_clblas,
                            { perftools::gputools::initialize_clblas(); });
```
Where does this come from?  Grep shows this is defined in `tensorflow/stream_executor/lib/initialize.h`:
```

class Initializer {
 public:
  typedef void (*InitializerFunc)();
  explicit Initializer(InitializerFunc func) { func(); }
};

#define REGISTER_INITIALIZER(type, name, body)                               \
  static void google_init_##type##_##name() { body; }                        \
  perftools::gputools::port::Initializer google_initializer_##type##_##name( \
      google_init_##type##_##name)

#define REGISTER_MODULE_INITIALIZER(name, body) \
  REGISTER_INITIALIZER(module, name, body)
  ```
we have:
- `body` is `{ initialize_clblas(); }`
- `name` is `register_clblas`
- the registration methods will all start with `google_init_module`, so we can search for them:
```
nm env/lib/python3.6/site-packages/tensorflow/python/_pywrap_tensorflow.so | grep google_init_module
00000000047d11c0 t __ZL30google_init_module_cl_platformv
00000000047cc090 t __ZL34google_init_module_cl_gpu_executorv
```
=> there are only the ones for the opencl functions (presumably also for cuda, when that is compiled in)

I think I might run the registration manually for now perhaps.

## Interesting references

### Static initializers

- https://meowni.ca/posts/static-initializers/ (very readable, though a bit quirky)
- http://cplusplus.bordoon.com/static_initialization.html
- http://neugierig.org/software/chromium/notes/2011/08/static-initializers.html