#ifndef BAR_VISUALIZATION_3D_H
#define BAR_VISUALIZATION_3D_H

#include "..\ShaderUtils.h"
#include "BaseVisualization.h"
#include <vector>
#include <GL/glew.h>
#include <GLFW/glfw3.h>

class BarVisualization3D : public BaseVisualization {
public:
    BarVisualization3D();
    ~BarVisualization3D();

    bool initialize(int windowWidth, int windowHeight) override;
    void render(const std::vector<float>& fftMagnitudes) override;
    bool shouldClose() override;
    void cleanup() override;

private:
    GLFWwindow* window;
    GLuint vbo, vao;
    std::vector<float> smoothedFFT;
    GLuint shaderProgram;
    GLint modelLoc, viewLoc, projLoc;

    // Camera / time tracking
    float timeAccum;
    int generation;
    static constexpr int MAX_HISTORY = 64;
    std::vector<std::vector<float>> fftHistory;

    // Build a 3D bar box (8 vertices, 12 triangles = 36 floats per bar)
    void addBar3D(float x, float yBase, float yHeight, float zPos,
                  float barWidth, float barDepth,
                  const Color& color, std::vector<float>& verts);
};

#endif
