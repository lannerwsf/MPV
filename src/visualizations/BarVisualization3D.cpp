#define _USE_MATH_DEFINES
#include "BarVisualization3D.h"
#include "ColorUtils.h"
#include <cmath>
#include <iostream>
#include <cstring>

// ── Minimal 4×4 matrix helpers (column-major for OpenGL) ──────────────────────

static void mat4Identity(float* m) {
    std::memset(m, 0, 16 * sizeof(float));
    m[0] = m[5] = m[10] = m[15] = 1.0f;
}

static void mat4Perspective(float* m, float fovRad, float aspect, float near, float far) {
    std::memset(m, 0, 16 * sizeof(float));
    float f = 1.0f / std::tan(fovRad * 0.5f);
    m[0]  = f / aspect;
    m[5]  = f;
    m[10] = (far + near) / (near - far);
    m[11] = -1.0f;
    m[14] = (2.0f * far * near) / (near - far);
}

static void mat4LookAt(float* m, float ex, float ey, float ez,
                       float tx, float ty, float tz,
                       float ux, float uy, float uz) {
    float f[3] = { tx - ex, ty - ey, tz - ez };
    float flen = std::sqrt(f[0]*f[0] + f[1]*f[1] + f[2]*f[2]);
    if (flen > 1e-8f) { f[0] /= flen; f[1] /= flen; f[2] /= flen; }
    float s[3] = { f[1]*uz - f[2]*uy, f[2]*ux - f[0]*uz, f[0]*uy - f[1]*ux };
    float slen = std::sqrt(s[0]*s[0] + s[1]*s[1] + s[2]*s[2]);
    if (slen > 1e-8f) { s[0] /= slen; s[1] /= slen; s[2] /= slen; }
    float u[3] = { s[1]*f[2] - s[2]*f[1], s[2]*f[0] - s[0]*f[2], s[0]*f[1] - s[1]*f[0] };
    m[0] = s[0]; m[1] = u[0]; m[2]  = -f[0]; m[3]  = 0.0f;
    m[4] = s[1]; m[5] = u[1]; m[6]  = -f[1]; m[7]  = 0.0f;
    m[8] = s[2]; m[9] = u[2]; m[10] = -f[2]; m[11] = 0.0f;
    m[12] = -(s[0]*ex + s[1]*ey + s[2]*ez);
    m[13] = -(u[0]*ex + u[1]*ey + u[2]*ez);
    m[14] =  (f[0]*ex + f[1]*ey + f[2]*ez);
    m[15] = 1.0f;
}

static void mat4Multiply(float* dst, const float* a, const float* b) {
    float tmp[16];
    for (int r = 0; r < 4; ++r)
        for (int c = 0; c < 4; ++c) {
            tmp[c*4+r] = a[r]*b[c*4] + a[4+r]*b[c*4+1] + a[8+r]*b[c*4+2] + a[12+r]*b[c*4+3];
        }
    std::memcpy(dst, tmp, 16 * sizeof(float));
}

// ── BarVisualization3D implementation ────────────────────────────────────────

BarVisualization3D::BarVisualization3D()
    : window(nullptr), vbo(0), vao(0), smoothedFFT(128, 0.0f),
      shaderProgram(0), modelLoc(-1), viewLoc(-1), projLoc(-1),
      timeAccum(0.0f), generation(0) {}

BarVisualization3D::~BarVisualization3D() {
    cleanup();
}

static void fb_callback_3d(GLFWwindow* window, int width, int height) {
    glViewport(0, 0, width, height);
}

bool BarVisualization3D::initialize(int windowWidth, int windowHeight) {
    if (!glfwInit()) {
        std::cerr << "Failed to initialize GLFW." << std::endl;
        return false;
    }

    window = glfwCreateWindow(windowWidth, windowHeight, "3D Bar Visualization", nullptr, nullptr);
    if (!window) {
        std::cerr << "Failed to create GLFW window." << std::endl;
        glfwTerminate();
        return false;
    }

    glfwMakeContextCurrent(window);
    glewExperimental = GL_TRUE;
    if (glewInit() != GLEW_OK) {
        std::cerr << "Failed to initialize GLEW." << std::endl;
        return false;
    }

    glfwSetFramebufferSizeCallback(window, fb_callback_3d);

    glGenVertexArrays(1, &vao);
    glGenBuffers(1, &vbo);

    glBindVertexArray(vao);
    glBindBuffer(GL_ARRAY_BUFFER, vbo);

    // Vertex format: vec3 aPos (location 0) + vec3 aColor (location 1) = 6 floats
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 6 * sizeof(float), (void*)0);
    glEnableVertexAttribArray(0);

    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 6 * sizeof(float), (void*)(3 * sizeof(float)));
    glEnableVertexAttribArray(1);

    glClearColor(0.05f, 0.05f, 0.1f, 1.0f);
    glEnable(GL_DEPTH_TEST);

    shaderProgram = createShaderProgram("visualizations/vertexShader3D.glsl",
                                        "visualizations/fragmentShader.glsl");
    if (shaderProgram == 0) {
        std::cerr << "ERROR: Failed to create 3D shader program!" << std::endl;
        exit(1);
    }

    glUseProgram(shaderProgram);
    modelLoc = glGetUniformLocation(shaderProgram, "model");
    viewLoc  = glGetUniformLocation(shaderProgram, "view");
    projLoc  = glGetUniformLocation(shaderProgram, "projection");

    return true;
}

void BarVisualization3D::addBar3D(float x, float yBase, float yHeight, float zPos,
                                  float barW, float barD,
                                  const Color& color, std::vector<float>& verts) {
    float left  = x;
    float right = x + barW;
    float bot   = yBase;
    float top   = yBase + yHeight;
    float front = zPos;
    float back  = zPos + barD;

    float r = color.r, g = color.g, b = color.b;

    // Each face: 2 triangles => 6 vertices
    // Front face (z = front)
    verts.insert(verts.end(), {
        left,  bot,  front, r,g,b,  right, bot,  front, r,g,b,  left,  top, front, r,g,b,
        right, bot,  front, r,g,b,  right, top,  front, r,g,b,  left,  top, front, r,g,b,
    });
    // Back face (z = back)
    verts.insert(verts.end(), {
        right, bot,  back, r,g,b,  left,  bot,  back, r,g,b,  left,  top, back, r,g,b,
        right, bot,  back, r,g,b,  left,  top,  back, r,g,b,  right, top, back, r,g,b,
    });
    // Left face (x = left)
    verts.insert(verts.end(), {
        left,  bot,  back, r,g,b,  left,  bot,  front,r,g,b,  left,  top, front,r,g,b,
        left,  bot,  back, r,g,b,  left,  top,  front,r,g,b,  left,  top, back, r,g,b,
    });
    // Right face (x = right)
    verts.insert(verts.end(), {
        right, bot,  front,r,g,b,  right, bot,  back, r,g,b,  right, top, back, r,g,b,
        right, bot,  front,r,g,b,  right, top,  back, r,g,b,  right, top, front,r,g,b,
    });
    // Top face (y = top)
    verts.insert(verts.end(), {
        left,  top,  front,r,g,b,  right, top,  front,r,g,b,  right, top, back, r,g,b,
        left,  top,  front,r,g,b,  right, top,  back, r,g,b,  left,  top, back, r,g,b,
    });
    // Bottom face (y = bot)
    verts.insert(verts.end(), {
        left,  bot,  back, r,g,b,  right, bot,  back, r,g,b,  right, bot, front,r,g,b,
        left,  bot,  back, r,g,b,  right, bot,  front,r,g,b,  left,  bot, front,r,g,b,
    });
}

void BarVisualization3D::render(const std::vector<float>& fftMagnitudes) {
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    glUseProgram(shaderProgram);

    if (fftMagnitudes.empty()) {
        glfwSwapBuffers(window);
        glfwPollEvents();
        return;
    }

    // ── Build projection + view matrices ──
    int fbWidth, fbHeight;
    glfwGetFramebufferSize(window, &fbWidth, &fbHeight);
    float aspect = (float)fbWidth / (float)fbHeight;

    float proj[16];
    mat4Perspective(proj, 45.0f * (float)M_PI / 180.0f, aspect, 0.1f, 100.0f);

    // Camera: orbit around the scene
    float camAngle = timeAccum * 0.15f;
    float camRadius = 3.5f;
    float camX = camRadius * std::sin(camAngle);
    float camZ = camRadius * std::cos(camAngle);
    float camY = 1.2f;

    float view[16];
    mat4LookAt(view, camX, camY, camZ,   0.0f, -0.2f, -2.0f,   0.0f, 1.0f, 0.0f);

    glUniformMatrix4fv(projLoc, 1, GL_FALSE, proj);
    glUniformMatrix4fv(viewLoc, 1, GL_FALSE, view);

    // ── Process FFT data ──
    size_t numBars = fftMagnitudes.size() / 8;
    if (numBars < 1) numBars = 1;

    if (smoothedFFT.size() != numBars) {
        smoothedFFT.resize(numBars, 0.0f);
    }

    float maxMagnitude = 0.0f;
    for (size_t i = 0; i < numBars; ++i) {
        if (fftMagnitudes[i] > maxMagnitude) maxMagnitude = fftMagnitudes[i];
    }
    if (maxMagnitude < 1e-6f) maxMagnitude = 1.0f;

    // Smoothed magnitudes for current frame
    std::vector<float> currentBars(numBars);
    for (size_t i = 0; i < numBars; ++i) {
        float norm = fftMagnitudes[i] / maxMagnitude;
        float logMag = std::log10(1.0f + norm * 10.0f) / std::log10(2.0f);
        smoothedFFT[i] = (smoothedFFT[i] * 0.9f) + (0.1f * logMag);
        float h = smoothedFFT[i] * 0.9f;
        if (h < 0.02f) h = 0.02f;
        currentBars[i] = h;
    }

    // ── Store frame in history (z-axis = time) ──
    fftHistory.push_back(currentBars);
    if (fftHistory.size() > MAX_HISTORY) {
        fftHistory.erase(fftHistory.begin());
    }

    // ── Build 3D geometry ──
    std::vector<float> vertices;
    float barW = 1.2f / numBars;
    float barD = 0.08f;
    float zStep = 0.08f;

    // World-space: bars sit on y=-1.0, z goes negative into the screen
    float zStart = 0.0f;

    for (int g = 0; g < (int)fftHistory.size(); ++g) {
        const auto& frame = fftHistory[g];
        float zPos = zStart - (fftHistory.size() - 1 - g) * zStep;
        // Dimmer / smaller for older frames (depth cue)
        float ageFactor = 0.3f + 0.7f * (float)(g + 1) / (float)fftHistory.size();

        for (size_t i = 0; i < frame.size(); ++i) {
            float h = frame[i] * ageFactor;
            float x = -0.6f + i * barW;
            Color c = getColorFromMagnitude(frame[i] / (ageFactor > 0.01f ? ageFactor : 1.0f), 0.0f, 1.0f);
            // Dim color for older frames
            c.r *= ageFactor;
            c.g *= ageFactor;
            c.b *= ageFactor;
            addBar3D(x, -1.0f, h, zPos, barW * 0.8f, barD, c, vertices);
        }
    }

    // ── Upload and draw ──
    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    glBufferData(GL_ARRAY_BUFFER, vertices.size() * sizeof(float), vertices.data(), GL_DYNAMIC_DRAW);

    glBindVertexArray(vao);

    // Identity model matrix
    float model[16];
    mat4Identity(model);
    glUniformMatrix4fv(modelLoc, 1, GL_FALSE, model);

    glDrawArrays(GL_TRIANGLES, 0, vertices.size() / 6);

    // ── Advance time ──
    timeAccum += 0.016f;  // ~60 fps step

    glfwSwapBuffers(window);
    glfwPollEvents();
}

bool BarVisualization3D::shouldClose() {
    return glfwWindowShouldClose(window);
}

void BarVisualization3D::cleanup() {
    if (vbo != 0) glDeleteBuffers(1, &vbo);
    if (vao != 0) glDeleteVertexArrays(1, &vao);
    if (window) glfwDestroyWindow(window);
    glfwTerminate();
}
