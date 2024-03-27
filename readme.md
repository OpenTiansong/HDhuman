
# HDhuman
The official release of the paper “HDhuman: High-quality Human Novel-view Rendering from Sparse Views”.  
[IEEE access of the paper](https://ieeexplore.ieee.org/abstract/document/10168294)  
[Home page of the project](http://cic.tju.edu.cn/faculty/likun/projects/HDhuman/index.html)

<!-- vscode-markdown-toc -->
* 1. [Dependencies overview](#Dependenciesoverview)
* 2. [Install dependencies.](#Installdependencies.)
    * 2.1. [Pytorch](#Pytorch)
    * 2.2. [Taichi](#Taichi)
    * 2.3. [Other python packages](#Otherpythonpackages)
    * 2.4. [File structure](#Filestructure)
    * 2.5. [StableViewSynthesis](#StableViewSynthesishttps:github.comintel-islStableViewSynthesis.git)
* 3. [Run rendering on example data with pre-trained weights.](#Runrenderingonexampledatawithpre-trainedweights.)
    * 3.1. [Prepare data and checkpoints.](#Preparedataandcheckpoints.)
    * 3.2. [Run rendering](#Runrendering)
* 4. [Tips for editing code to run rendering with custom data.](#Tipsforedittingcodetorunrenderingwithcustomdata.)

<!-- vscode-markdown-toc-config
    numbering=true
    autoSave=true
    /vscode-markdown-toc-config -->
<!-- /vscode-markdown-toc -->



##  1. <a name='Dependenciesoverview'></a>Dependencies overview
Python packages
- numpy
- scikit-image
- pillow
- torch
- torchvision
- torch-scatter
- torch-sparse
- torch-geometric
- torch-sparse
- open3d
- python-opencv
- matplotlib
- pandas

External Code Repository
- [StableViewSynthesis](https://github.com/intel-isl/StableViewSynthesis.git), install in ```hdhuman/StableViewSynthesis```, please refer to **Setup example** for details.


To compile the Python extensions you will also need `Eigen` and `cmake`.


##  2. <a name='Installdependencies.'></a>Install dependencies.
The code is tested in ```Ubuntu 20.04```.

Create environment.
```shell
conda create -n hdhuman python=3.9
#the python version is critial when installing certain version of taichi.
conda activate hdhuman
```
###  2.1. <a name='Pytorch'></a>Pytorch
Install pytorch and torchvision, tested with pytorch 2.0.1.  
The specific installing command will change over time, please refer to the official website <https://pytorch.org/get-started/previous-versions/>
```shell
#use the command from the official website
#the following is just an example
conda install pytorch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 pytorch-cuda=11.8 -c pytorch -c nvidia
```

###  2.2. <a name='Taichi'></a>Taichi
Tested with taichi-0.9.0, taichi_glsl-0.0.12
```
pip install taichi==0.7.26 taichi-glsl==0.0.11
```

###  2.3. <a name='Otherpythonpackages'></a>Other python packages
- numpy
- scikit-image
- pillow
- torch
- torchvision
- torch-scatter
- torch-sparse
- torch-geometric
- open3d
- python-opencv
- matplotlib
- pandas
- trimesh
```shell
#according to the current official torch-scatter torch-scatter torch-geometric website, install them via conda.
conda install pytorch-scatter pytorch-sparse pyg -c pyg

pip install opencv-python matplotlib scikit-image pillow torch-scatter torch-sparse torch-geometric pandas icecream trimesh[all]
```

###  2.4. <a name='Filestructure'></a>File structure
The final file struct needs to be like:
```
├── checkpoints
│   ├── encode_net_iter_351000
│   ├── encode_optim_iter_351000
│   ├── render_net_iter_351000
│   └── render_optim_iter_351000
├── example_data
│   ├── 134411543679642
│   ├── 138411542490258
│   ├── 138711555454990
│   ├── 139011543669720
│   └── 140011549916904
├── HDhuman
│   ├── lib
│   ├── readme.md
│   └── script
└── StableViewSynthesis
```
where ```HDhuman``` is the folder of the repository.



###  2.5. <a name='StableViewSynthesishttps:github.comintel-islStableViewSynthesis.git'></a>[StableViewSynthesis](https://github.com/intel-isl/StableViewSynthesis.git)
Clone the repository
```shell
git clone https://github.com/intel-isl/StableViewSynthesis.git
```
Make sure the relative path is correct referring to [File structure](#Filestructure).

Initialize the submodule
```shell
cd StableViewSynthesis
git submodule update --init --recursive
```

Some regular packages are required.
```shell
#run by sudoers
sudo apt install pkg-config libeigen3-dev
```

Build ```ext/preprocess```
```shell
cd ext/preprocess
cmake -DCMAKE_BUILD_TYPE=Release .
make 
```

Edit ```StableViewSynthesis/ext/mytorch/setup.py``` nvcc_args according to the CUDA devices, e.g. sm_86 is for RTX 3090.
```python
nvcc_args = [
    "-arch=sm_86",
    "-gencode=arch=compute_86,code=sm_86"
]
```
Build ```ext/mytorch```
```
cd ../mytorch
python setup.py build_ext --inplace
```
A log of installing StableViewSynthesis can be find in ```log_install_svs.txt```.

##  3. <a name='Runrenderingonexampledatawithpre-trainedweights.'></a>Run rendering on example data with pre-trained weights.
###  3.1. <a name='Preparedataandcheckpoints.'></a>Prepare data and checkpoints.
* Download the pre-trained weights and place them in the ```checkpoints``` folder.   
* Download the example data and place them in the ```example_data``` folder.  
> The download links are (either of them include full files):   
> [Dropbox](https://www.dropbox.com/scl/fo/rv038639461fr81zncltq/h?rlkey=ibpw8p4enf555tabx847zqin5&dl=0)  
> [BaiduCloudDisk](https://pan.baidu.com/s/1YckBxXYAV09dsJMEAdt8aA?pwd=9fn0) (The password is ```9fn0``` and is included in the link.)

* Download (.tar) files and untar them. The target file structure should be like:
> ```
> ├── checkpoints
> │   ├── encode_net_iter_351000
> │   ├── encode_optim_iter_351000
> │   ├── render_net_iter_351000
> │   └── render_optim_iter_351000
> ├── example_data
> │   ├── 134411543679642
> │   ├── 138411542490258
> │   ├── 138711555454990
> │   ├── 139011543669720
> │   └── 140011549916904
> ├── HDhuman
> └── StableViewSynthesis
> ``` 
Run a script to check these paths.
```shell
#run in the folder of 'HDhuman'
#check the paths of example data and checkpoints
python script/check_path_for_demo.py
#with assertions in this script should have no output.
```
###  3.2. <a name='Runrendering'></a>Run rendering
```shell
python script/run_render_on_example_data.py
```
Some config can be edited directly in ```run_render_on_example_data.py```, e.g.
```python
NUM_VIEW_INPUT = 6
NUM_VIEW_OUTPUT = 180 #can be factors of 360, there are 360 pre-defined views
```
The output of the render can be found in ```example_data/render_output```.

##  4. <a name='Tipsforedittingcodetorunrenderingwithcustomdata.'></a>Tips for editing code to run rendering with custom data.
referring to the projection
prepare the following data
* reconstructed mesh.
* multi-view input with camera parameters
make sure the camera parameters can pass the projection check like ```HDhuman/script/check_projection.py``` and call the function ```render_one_subject_from_path_and_write_to_folder()``` in ```HDhuman/lib/test/test_render_api.py```.
```python
#check the path of example data
python script/check_projection.py
#the output should be in ```example_data/check_camera```, which projects the vertice of the mesh to the image according to the camera parameters.
```
