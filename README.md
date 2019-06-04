
# Inverse Kinematics on Stereoids ? 


<a href="https://www.youtube.com/watch?v=xzZjtbtjyEE" target="_blank"><img src="http://img.youtube.com/vi/xzZjtbtjyEE/0.jpg" alt="Example video" width="240" height="180" border="10" /></a>

<a href="https://www.youtube.com/watch?v=ZSypmxIydpA" target="_blank"><img src="http://img.youtube.com/vi/ZSypmxIydpA/0.jpg" alt="Example video" width="240" height="180" border="10" /></a>

## Compile
(See also cmkae for more specific compile instructions) 

Compile this project using the standard cmake routine:

    mkdir build
    cd build
    cmake ..
    make
    
  

This should find and build the dependencies and create a `example_bin` binary.

## Run

From within the `build` directory just issue:

    ./example_bin

A glfw app should launch displaying a 3D cube.

## Dependencies

The only dependencies are stl, eigen, [libigl](http://libigl.github.io/libigl/) and
the dependencies of the `igl::opengl::glfw::Viewer`.

We recommend you to install libigl using git via:

    git clone https://github.com/libigl/libigl.git
    cd libigl/
    git submodule update --init --recursive
    cd ..

If you have installed libigl at `/path/to/libigl/` then a good place to clone
this library is `/path/to/libigl-example-project/`.
