# docker-bake.hcl - vLLM Docker build configuration
#
# This file lives in vLLM repo at docker/docker-bake.hcl
#
# Usage:
#   cd docker && docker buildx bake        # Build default target (openai)
#   cd docker && docker buildx bake test   # Build test target
#   docker buildx bake --print             # Show resolved config
#
# Reference: https://docs.docker.com/build/bake/reference/

# Build configuration

variable "MAX_JOBS" {
  default = 16
}

variable "NVCC_THREADS" {
  default = 8
}

variable "TORCH_CUDA_ARCH_LIST" {
  default = "12.1a"
}

variable "FLASHINFER_CUDA_ARCH_LIST" {
  default = "12.1a"
}

variable "CMAKE_CUDA_ARCHITECTURES" {
  default = "121a"
}

variable "CUTE_DSL_ARCH" {
  default = "sm_121a"
}

variable "VLLM_FLASH_ATTN_GIT_REPOSITORY" {
  default = "https://github.com/gardner/vllm-flash-attention.git"
}

variable "VLLM_FLASH_ATTN_GIT_TAG" {
  default = "de3849e75d07edd1c00aec02c92ec852ba757adc"
}

variable "GB10_PREBUILT_WHEEL_URLS" {
  default = ""
}

variable "GB10_REQUIRE_FLASHINFER_WHEELS" {
  default = true
}

variable "COMMIT" {
  default = ""
}

variable "VLLM_BUILD_COMMIT" {
  default = "unknown"
}

variable "VLLM_BUILD_PIPELINE" {
  default = "local"
}

variable "VLLM_BUILD_URL" {
  default = ""
}

variable "VLLM_IMAGE_TAG" {
  default = "local/vllm-openai:dev"
}

# Groups

group "default" {
  targets = ["openai"]
}

group "all" {
  targets = ["openai", "openai-ubuntu2404"]
}

# Base targets

target "_common" {
  dockerfile = "docker/Dockerfile"
  context    = "."
  args = {
    max_jobs                    = MAX_JOBS
    nvcc_threads                = NVCC_THREADS
    torch_cuda_arch_list        = TORCH_CUDA_ARCH_LIST
    flashinfer_cuda_arch_list   = FLASHINFER_CUDA_ARCH_LIST
    cmake_cuda_architectures    = CMAKE_CUDA_ARCHITECTURES
    cute_dsl_arch               = CUTE_DSL_ARCH
    vllm_flash_attn_git_repository = VLLM_FLASH_ATTN_GIT_REPOSITORY
    vllm_flash_attn_git_tag        = VLLM_FLASH_ATTN_GIT_TAG
    gb10_prebuilt_wheel_urls       = GB10_PREBUILT_WHEEL_URLS
    gb10_require_flashinfer_wheels = GB10_REQUIRE_FLASHINFER_WHEELS
    VLLM_BUILD_COMMIT           = VLLM_BUILD_COMMIT != "unknown" ? VLLM_BUILD_COMMIT : (COMMIT != "" ? COMMIT : "unknown")
    VLLM_BUILD_PIPELINE         = VLLM_BUILD_PIPELINE
    VLLM_BUILD_URL              = VLLM_BUILD_URL
    VLLM_IMAGE_TAG              = VLLM_IMAGE_TAG
  }
}

target "_labels" {
  labels = {
    "org.opencontainers.image.source"      = "https://github.com/vllm-project/vllm"
    "org.opencontainers.image.vendor"      = "vLLM"
    "org.opencontainers.image.title"       = "vLLM"
    "org.opencontainers.image.description" = "vLLM: A high-throughput and memory-efficient inference and serving engine for LLMs"
    "org.opencontainers.image.licenses"    = "Apache-2.0"
    "org.opencontainers.image.revision"    = VLLM_BUILD_COMMIT != "unknown" ? VLLM_BUILD_COMMIT : (COMMIT != "" ? COMMIT : "unknown")
    "org.opencontainers.image.version"     = VLLM_IMAGE_TAG
    "org.opencontainers.image.url"         = VLLM_BUILD_URL
    "ai.vllm.build.commit"                 = VLLM_BUILD_COMMIT != "unknown" ? VLLM_BUILD_COMMIT : (COMMIT != "" ? COMMIT : "unknown")
    "ai.vllm.build.pipeline"               = VLLM_BUILD_PIPELINE
    "ai.vllm.build.url"                    = VLLM_BUILD_URL
    "ai.vllm.image.tag"                    = VLLM_IMAGE_TAG
  }
  annotations = [
    "index,manifest:org.opencontainers.image.revision=${VLLM_BUILD_COMMIT != "unknown" ? VLLM_BUILD_COMMIT : (COMMIT != "" ? COMMIT : "unknown")}",
  ]
}

# Build targets

target "test" {
  inherits = ["_common", "_labels"]
  target   = "test"
  tags     = ["vllm:test"]
  output   = ["type=docker"]
}

target "openai" {
  inherits = ["_common", "_labels"]
  target   = "vllm-openai"
  tags     = ["vllm:openai"]
  output   = ["type=docker"]
}

# Ubuntu 24.04 targets

target "test-ubuntu2404" {
  inherits = ["_common", "_labels"]
  target   = "test"
  tags     = ["vllm:test-ubuntu24.04"]
  args = {
    UBUNTU_VERSION          = "24.04"
    GDRCOPY_OS_VERSION      = "Ubuntu24_04"
  }
  output = ["type=docker"]
}

target "openai-ubuntu2404" {
  inherits = ["_common", "_labels"]
  target   = "vllm-openai"
  tags     = ["vllm:openai-ubuntu24.04"]
  args = {
    UBUNTU_VERSION          = "24.04"
    GDRCOPY_OS_VERSION      = "Ubuntu24_04"
  }
  output = ["type=docker"]
}
