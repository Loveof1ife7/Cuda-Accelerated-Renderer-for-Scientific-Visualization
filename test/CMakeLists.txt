add_library(test_helper STATIC 
    helper.cpp)


target_include_directories(test_helper PUBLIC
    ${CMAKE_SOURCE_DIR}/third-party/glad/include
    ${CMAKE_SOURCE_DIR}/third-party/glfw/include
    ${CMAKE_SOURCE_DIR}/test
)

target_link_libraries(test_helper
    glad
    glfw
    OpenGL::GL
)
