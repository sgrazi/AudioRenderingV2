# AudioRenderingV2

AudioRenderingV2 is an advanced acoustic rendering engine developed as a final career project by [Agustín Bologna](https://github.com/Etzior), [Agustín dos Santos](https://github.com/AgusdosSantos), and [Stefano Graziani](https://github.com/sgrazi) at the Facultad de Ingeniería, Universidad de la República. The project aims to provide real-time auralization in 3D virtual scenes using advanced signal processing and graphical computation techniques. The repository hosts examples of auralized sounds in various scenarios and the source code for the project, which builds upon the work provided by the repository [AudioRendering](https://github.com/cameelo/AudioRendering) (hence the V2 in the name).
 
AudioRenderingV2 takes advantage of NVIDIA GPUs and the OptiX framework to create a realistic sound experience by simulating how sound waves interact with objects in a virtual environment. Using CUDA, we can tap into the powerful parallel processing capabilities of GPUs, making the whole process much faster.

### Main app flow

- Setting the Scene: First, we set up the virtual environment, including the layout of objects, sound sources, and where the listener is located. All of this needs to be encapsulated in an .obj and .mtl file (see assets directory).
- Sound Ray Tracing: We use OptiX to shoot sound rays from the sources. These rays bounce around the scene, reflecting off surfaces and being absorbed just like light would. This simulates propagation pretty realistically to helps us figure out how sound travels and changes as it moves. The rays impact the receiver which generates the _impulse response_ of the scene, this is the goal of the raytracer.
- Using FFT: Once we've generated the IR, we use the Fast Fourier Transform (FFT) to process the sound signal with the IR (this is called a convolution). This is done via CUDA to take advantage of parallel processing for math-heavy operations.
- Real-Time Action: Thanks to the technologies used, everything happens (hopefully, depending on your system's specs) fast enough that its possible to experience the sound changes in real-time. Moving around the scene allows for real time auralization, which is the main objective of this project.

## How to build

### Requirements

- Have CUDA v12.1+ installed
- Have OptiX SDK v7.7.0+
- CMake v3.26.0+

### Steps

1. Go to the root folder
2. `mkdir build`
3. `cd build && cmake.exe ..\prebuild\`
4. Open the proyect on visual studio `optix.sln`
5. In visual studio build solution `obj_raytracer`


## Usage

1. Set a `config.json` file similar to the one in this repo with your preferences.

2. Run the app. Main way we recommend is through Visual Studio's UI, by debugging the application you're able to run it during development. 
    1. If you've built the .exe already and don't want to run Visual Studio, this command will run it. Make sure to have all used assets (.objs and sound sources) in the right places.
    `./obj_raytracer.exe <location of your config.json>`
    
3. Have fun!

### Controls

- W,A,S,D: to move
- E: to place the emitter object at current camera location
- R: force audio re-render
- V: mute/unmute
- P: print the next re-rendered IR