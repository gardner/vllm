# Install OpenAI triton_kernels from a pinned Git ref.
set(TRITON_KERNELS_GIT_REPOSITORY
    "https://github.com/gardner/triton.git"
    CACHE STRING "Git repository for bundled triton_kernels.")
set(TRITON_KERNELS_GIT_TAG
    "28c73277042f3140a7c8c448913416d24fb57e61"
    CACHE STRING "Git tag, branch, or commit for bundled triton_kernels.")
if(DEFINED ENV{TRITON_KERNELS_GIT_REPOSITORY})
  set(TRITON_KERNELS_GIT_REPOSITORY "$ENV{TRITON_KERNELS_GIT_REPOSITORY}"
      CACHE STRING "Git repository for bundled triton_kernels." FORCE)
endif()
if(DEFINED ENV{TRITON_KERNELS_GIT_TAG})
  set(TRITON_KERNELS_GIT_TAG "$ENV{TRITON_KERNELS_GIT_TAG}"
      CACHE STRING "Git tag, branch, or commit for bundled triton_kernels." FORCE)
endif()

# Set TRITON_KERNELS_SRC_DIR for use with local development with vLLM. We expect TRITON_KERNELS_SRC_DIR to
# be directly set to the triton_kernels python directory.
if (DEFINED ENV{TRITON_KERNELS_SRC_DIR})
  set(TRITON_KERNELS_SRC_DIR $ENV{TRITON_KERNELS_SRC_DIR})
elseif(VLLM_USE_LOCAL_GB10_DEPS AND NOT TRITON_KERNELS_SRC_DIR)
  get_filename_component(_VLLM_SIBLING_TRITON_KERNELS_SRC_DIR
    "${CMAKE_SOURCE_DIR}/../triton/python/triton_kernels/triton_kernels"
    ABSOLUTE)
  if(EXISTS "${_VLLM_SIBLING_TRITON_KERNELS_SRC_DIR}/__init__.py")
    set(TRITON_KERNELS_SRC_DIR "${_VLLM_SIBLING_TRITON_KERNELS_SRC_DIR}")
  endif()
endif()

if(TRITON_KERNELS_SRC_DIR)
  message(STATUS "[triton_kernels] Fetch from ${TRITON_KERNELS_SRC_DIR}")
  FetchContent_Declare(
          triton_kernels
          SOURCE_DIR ${TRITON_KERNELS_SRC_DIR}
  )

else()
  message (STATUS "[triton_kernels] Fetch from ${TRITON_KERNELS_GIT_REPOSITORY}:${TRITON_KERNELS_GIT_TAG}")
  FetchContent_Declare(
          triton_kernels
          # TODO (varun) : Fetch just the triton_kernels directory from Triton
          GIT_REPOSITORY ${TRITON_KERNELS_GIT_REPOSITORY}
          GIT_TAG ${TRITON_KERNELS_GIT_TAG}
          GIT_PROGRESS TRUE
          SOURCE_SUBDIR python/triton_kernels/triton_kernels
  )
endif()

# Fetch content
FetchContent_MakeAvailable(triton_kernels)

if (NOT triton_kernels_SOURCE_DIR)
  message (FATAL_ERROR "[triton_kernels] Cannot resolve triton_kernels_SOURCE_DIR")
endif()

if (TRITON_KERNELS_SRC_DIR)
  set(TRITON_KERNELS_PYTHON_DIR "${triton_kernels_SOURCE_DIR}/")
else()
  set(TRITON_KERNELS_PYTHON_DIR "${triton_kernels_SOURCE_DIR}/python/triton_kernels/triton_kernels/")
endif()

message (STATUS "[triton_kernels] triton_kernels is available at ${TRITON_KERNELS_PYTHON_DIR}")

add_custom_target(triton_kernels)

# Ensure the vllm/third_party directory exists before installation
install(CODE "file(MAKE_DIRECTORY \"\${CMAKE_INSTALL_PREFIX}/vllm/third_party/triton_kernels\")")

## Copy .py files to install directory.
install(DIRECTORY
        ${TRITON_KERNELS_PYTHON_DIR}
        DESTINATION
        vllm/third_party/triton_kernels/
        COMPONENT triton_kernels
        FILES_MATCHING PATTERN "*.py")
