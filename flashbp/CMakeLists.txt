cmake_minimum_required(VERSION 3.18)
project(flashbp)

find_package(pybind11 REQUIRED)

# ── Optional LibTorch backend ────────────────────────────────────────────────
# Powers the tensor-network ML decoder.  Requires LibTorch on
# CMAKE_PREFIX_PATH (or Torch_DIR pointing at .../libtorch/share/cmake/Torch).
# Disable with:  pip install . -C cmake.args="-DFLASHBP_ENABLE_TORCH=OFF"
option(FLASHBP_ENABLE_TORCH
       "Build LibTorch-backed decoders (needs LibTorch installed)."
       ON)

if(FLASHBP_ENABLE_TORCH)
    find_package(Torch REQUIRED)
    message(STATUS "flashbp: LibTorch enabled (TORCH_INSTALL_PREFIX=${TORCH_INSTALL_PREFIX})")
else()
    message(STATUS "flashbp: LibTorch disabled — TN decoders will report unavailable.")
endif()

pybind11_add_module(_flashbp
    src/_flashbp.cpp
    src/flashbp_core.cpp
    src/torch_backend.cpp
    src/decoders/simple_decoder.cpp
    src/decoders/tensor_decoder.cpp
    src/decoders/gbp_region_policy.cpp
    src/decoders/gbp_decoder.cpp
    src/decoders/degree_decoder.cpp
    src/decoders/ml_decoder.cpp
    src/decoders/surprise_ml_decoder.cpp
)

target_include_directories(_flashbp PRIVATE src)
target_compile_features(_flashbp PRIVATE cxx_std_17)

if(FLASHBP_ENABLE_TORCH)
    target_link_libraries(_flashbp PRIVATE ${TORCH_LIBRARIES})
    if(TORCH_CXX_FLAGS)
        target_compile_options(_flashbp PRIVATE ${TORCH_CXX_FLAGS})
    endif()
    target_compile_definitions(_flashbp PRIVATE FLASHBP_HAS_TORCH=1)
endif()

install(TARGETS _flashbp DESTINATION flashbp)
