#include "helper.hpp"

GLuint createGradientTexture(int width, int height)
{
    unsigned char *data = new unsigned char[width * height * 4];

    for (int y = 0; y < height; ++y)
    {
        for (int x = 0; x < width; ++x)
        {
            int idx = (y * width + x) * 4;
            data[idx + 0] = x * 255 / width;  // R
            data[idx + 1] = y * 255 / height; // G
            data[idx + 2] = 128;              // B
            data[idx + 3] = 255;              // A
        }
    }

    GLuint texID;
    glGenTextures(1, &texID);
    glBindTexture(GL_TEXTURE_2D, texID);

    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, width, height,
                 0, GL_RGBA, GL_UNSIGNED_BYTE, data);

    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

    delete[] data;
    return texID;
}
