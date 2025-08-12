#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include <imgui.h>
#include <imgui_impl_glfw.h>
#include <imgui_impl_opengl3.h>
#include <iostream>

#include "image_presenter.hpp"
#include "scene.hpp"
#include "volume_renderer.hpp"

#include "helper.hpp"

const int WINDOW_WIDTH = 800;
const int WINDOW_HEIGHT = 600;

GLFWwindow *initWindow()
{
  if (!glfwInit())
  {
    std::cerr << "Failed to initialize GLFW" << std::endl;
    return nullptr;
  }

  glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
  glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
  glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

  GLFWwindow *window = glfwCreateWindow(WINDOW_WIDTH, WINDOW_HEIGHT, "ImGui OpenGL Example", nullptr, nullptr);
  if (!window)
  {
    std::cerr << "Failed to create GLFW window" << std::endl;
    glfwTerminate();
    return nullptr;
  }

  glfwMakeContextCurrent(window);
  if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress))
  {
    std::cerr << "Failed to initialize GLAD" << std::endl;
    return nullptr;
  }

  return window;
}

void initImGui(GLFWwindow *window)
{
  IMGUI_CHECKVERSION();
  ImGui::CreateContext();
  ImGuiIO &io = ImGui::GetIO();
  (void)io;

  ImGui::StyleColorsDark();

  const char *glsl_version = "#version 100";
  ImGui_ImplGlfw_InitForOpenGL(window, true);
  ImGui_ImplOpenGL3_Init("#version 330");

  /* Load Fonts */
  // io.Fonts->AddFontFromFileTTF(
  //     (resource_manager.getResource("fonts/Roboto-Medium.ttf")).c_str(), 16.0f);
  // io.Fonts->AddFontDefault();
}

void shutdownImGui()
{
  ImGui_ImplOpenGL3_Shutdown();
  ImGui_ImplGlfw_Shutdown();
  ImGui::DestroyContext();
}

void renderScene(GLuint texID)
{
  glClearColor(0.45f, 0.55f, 0.60f, 1.00f);
  glClear(GL_COLOR_BUFFER_BIT);

  glGenTextures(1, &texID);
}

void renderImGui()
{
  ImGui_ImplOpenGL3_NewFrame();
  ImGui_ImplGlfw_NewFrame();
  ImGui::NewFrame();

  ImGui::Begin("Hello, SciVis!");
  ImGui::Text("Cuda-SciVis.");
  ImGui::End();

  ImGui::Render();
  ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
}

int main(int, char **)
{
  GLFWwindow *window = initWindow();
  if (!window)
  {
    fprintf(stderr, "Failed to create GLFW window!\n");
    return 1;
  }

  initImGui(window);

  ImagePresenter *presenter = new ImagePresenter(WINDOW_WIDTH, WINDOW_HEIGHT);
  VolumeRenderer::CreateInfo ci{WINDOW_WIDTH, WINDOW_HEIGHT, false};
  VolumeRenderer *renderer = new VolumeRenderer(ci);
  Scene *scene = new Scene();
  // renderer->render(scene);
  GLuint textureID = createGradientTexture(WINDOW_WIDTH, WINDOW_HEIGHT);

  while (!glfwWindowShouldClose(window))
  {
    glfwPollEvents();

    // presenter->display(renderer->getOutputTexture());
    presenter->display(textureID);

    glfwSwapBuffers(window);
  }

  shutdownImGui();
  glfwDestroyWindow(window);
  glfwTerminate();

  return 0;
}
