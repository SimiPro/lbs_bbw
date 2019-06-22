

GLuint screenshot_to_texture(igl::opengl::glfw::Viewer &viewer) {

    // save current pose as image
    // Allocate temporary buffers
    Eigen::Matrix<unsigned char,Eigen::Dynamic,Eigen::Dynamic> R(1280,800);
    Eigen::Matrix<unsigned char,Eigen::Dynamic,Eigen::Dynamic> G(1280,800);
    Eigen::Matrix<unsigned char,Eigen::Dynamic,Eigen::Dynamic> B(1280,800);
    Eigen::Matrix<unsigned char,Eigen::Dynamic,Eigen::Dynamic> A(1280,800);

    // Draw the scene in the buffers
    viewer.core.draw_buffer(viewer.data(),false,R,G,B,A);

    // Save it to a PNG
    igl::png::writePNG(R,G,B,A,"out.png");

    const int comp = 4;                                  // 4 Channels Red, Green, Blue, Alpha
    const int stride_in_bytes = R.rows()*comp;           // Length of one row in bytes

    std::vector<unsigned char> data(R.size()*comp,0);     // The image itself;
    for (unsigned i = 0; i < R.rows(); ++i) {
        for (unsigned j = 0; j < R.cols(); ++j) {
            data[(j * R.rows() * comp) + (i * comp) + 0] = R(i, R.cols()-1-j);
            data[(j * R.rows() * comp) + (i * comp) + 1] = G(i, R.cols()-1-j);
            data[(j * R.rows() * comp) + (i * comp) + 2] = B(i, R.cols()-1-j);
            data[(j * R.rows() * comp) + (i * comp) + 3] = A(i, R.cols()-1-j);
        }
    }

    // Transfer data to gpu
    GLuint my_opengl_texture;  
    glGenTextures(1, &my_opengl_texture);
    glBindTexture(GL_TEXTURE_2D, my_opengl_texture);    
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, 1280, 800, 0, GL_RGBA, GL_UNSIGNED_BYTE, data.data());

    return my_opengl_texture;
}