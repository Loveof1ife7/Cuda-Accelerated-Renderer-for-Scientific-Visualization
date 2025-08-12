cmake_minimum_required(VERSION 3.8.0)

project(cuda-volume-renderer LANGUAGES C CXX CUDA)

# Use modern CMake target-based design
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 11)

# Compilation flags
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -w")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-relaxed-constexpr -Xcompiler=-O3")

set(CMAKE_CUDA_ARCHITECTURES 86)

# === Dependencies ===
# CUDA
find_package(CUDA REQUIRED)
include_directories(${CUDA_INCLUDE_DIRS})
if(CMAKE_CUDA_COMPILER)
  set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcudafe --diag_suppress=20236")
endif()

# OpenGL
find_package(OpenGL REQUIRED)
if(OpenGL_FOUND)
	message(STATUS "OpenGL found.")
	include_directories(${OPENGL_INCLUDE_DIR})
endif()

# glfw
find_package(glfw3 3.2 REQUIRED)
if(glfw3_FOUND)
	message(STATUS "GLFW3 found.")
endif()

# imGui
file(GLOB_RECURSE IMGUI_SOURCES
    "third-party/imgui/*.cpp"
    "third-party/imgui/*.h"
)
add_library(imgui STATIC ${IMGUI_SOURCES})

target_include_directories(imgui PUBLIC 
    third-party/imgui
    third-party/imgui/backends)

target_compile_definitions(imgui PUBLIC IMGUI_IMPL_OPENGL_LOADER_GLAD)

target_link_libraries(imgui PUBLIC
    glad
    glfw)

# === Include and Link Third-Party Libraries ===
add_library(glad STATIC third-party/glad/src/glad.c)
target_include_directories(glad PUBLIC third-party/glad/include)

# === Add Subdirectories ===
add_subdirectory(test)

# === Executable ===
file(GLOB_RECURSE SOURCE_FILES
    "src/*.cpp"
    "src/*.cu"
)

add_executable(cuda-volume-renderer ${SOURCE_FILES})

# CUDA settings
set_target_properties(cuda-volume-renderer PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

target_include_directories(cuda-volume-renderer PRIVATE include src third-party third-party/imgui third-party/imgui/backends)

target_link_libraries(cuda-volume-renderer 
    PRIVATE    
    glad 
    glfw 
    OpenGL::GL
    imgui
    test_helper
    ${CMAKE_DL_LIBS})

# === Resource Handling ===
file(GLOB_RECURSE RES_IMG CONFIGURE_DEPENDS resources/*.{png,jpg})
file(GLOB_RECURSE RES_BIN CONFIGURE_DEPENDS resources/*.bin)
file(GLOB_RECURSE RES_FONT CONFIGURE_DEPENDS third-party/imgui/misc/fonts/*.ttf)

set(RES_DIR ${CMAKE_BINARY_DIR}/resources)
file(MAKE_DIRECTORY ${RES_DIR}/images ${RES_DIR}/fonts ${RES_DIR}/bin)

foreach(file IN LISTS RES_IMG)
    file(COPY ${file} DESTINATION ${RES_DIR}/images)
endforeach()

foreach(file IN LISTS RES_BIN)
    file(COPY ${file} DESTINATION ${RES_DIR}/bin)
endforeach()

foreach(file IN LISTS RES_FONT)
    file(COPY ${file} DESTINATION ${RES_DIR}/fonts)
endforeach()

target_compile_definitions(cuda-volume-renderer PRIVATE RESOURCES_DIRECTORY="${RES_DIR}")

if(WIN32)
    target_compile_definitions(cuda-volume-renderer PRIVATE _AMD64_)
endif()
