# win 10 + Theano + Lasagne + GPU配置

谨以此文记录我若干个漫漫长夜在安装过程中踩下的坑。

此文记录于2020.4.2，不保证在今后的某个更新后还可用。

## 前言

为了运行某心理学论文的代码，我不得不安装`Theano`+`Lasagne`的机器学习大礼包。

[`Theano`](http://deeplearning.net/software/theano/index.html)和[`Lasagne`](https://lasagne.readthedocs.io/en/latest/#)的文档中有一些安装步骤，但是会出现很多奇怪的问题，让我熬了多少个夜，掉了多少个头发。

Mac下安装还是很方便的，但CPU跑起来实在是太慢了。奈何想在mac下用GPU跑网络有些耗财耗力，我转向了windows的怀抱。为什么没用Linux呢，因为玩起来不方便，也是为了偷懒，现在想起来就是后悔，十分的后悔。

具体配置：

- windows 10, 64 bit
- Visual Studio 2017
- CUDA Toolkit 9.0
- cuDNN v7.0.5 (Dec 5, 2017), for CUDA 9.0
- Anaconda 3, python 3.7 version, 64 bit
- python 3.5的环境
- theano 1.0.2
- pygpu 0.7.6
- Lasagne 0.2.dev 1



## 安装前准备

所有程序安装时一定要勾选添加环境变量。

### Visual Studio 2017

装`Cuda 9.0`时需要VS。不可以用2019，`Cuda 9.0`不支持`Visual Studio 2019`。2013/2015由于比较老，这里没有考虑。

### CUDA Toolkit 9.0

写此文时`CUDA`已经有`10.2`版本了。为什么不装新版本呢，因为要装老版本的`cuDNN`。为什么要装老版本的`cuDNN`呢，因为新版本的`cuDNN`与`Theano`和`libgpuarray`有兼容问题。

在我这里安装目录为

```
c:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v9.0
```

[click here](https://developer.nvidia.com/cuda-90-download-archive)

### cuDNN

这里选择了`cuDNN v7.0.5 (Dec 5, 2017), for CUDA 9.0`。为什么支持`CUDA 9.0`的`cuDNN`都出到`7.6.4`了，我还要用`7.0.5`呢，还是兼容性问题。我装个`7.6`说我版本过高，让我降级`cnDNN`或者升级`Theano`，`Theano`又没有更新的了，我能怎么办呢。

下载完成直接把`lib`等文件夹解压到`CUDA\v9.0\`文件夹内

[click here](https://developer.nvidia.com/rdp/cudnn-archive)

### Anaconda

为了用着习惯，装了python 3.7 ver的Acaconda。配置环境的时候要新建一个*python 3.5*的的环境，`Theano`对python 3.7兼容有问题，至少在我这里是的。

**python 3.5 ! python 3.5 ! python 3.5 !**



## 开始安装



### 库安装

按照官网教程来就行

```shell
conda install numpy scipy mkl-service libpython m2w64-toolchain nose git

pip install parameterized

conda install theano pygpu
```

此时获得的版本为`theano 1.0.2`，`pygpu 0.7.6`



### 配置文件

在自己的`Users`文件夹下新建`.theanorc`配置文件

```
[global]
device = cuda
floatX = float32
allow_gc = false
[gcc]
cxxflags=-IC:\Users\Dell\Anaconda3\envs\theano\Library\mingew-w64\bin
[cuda]
root = c:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v9.0
[dnn]
library_path = c:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v9.0\lib\x64
include_path = c:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v9.0\include
```

`cxxflags`那里找到刚才安装的`m2w64-toolchain`的位置

`root`就是`CUDA`的目录

`[dnn]`下面的两个配置就是刚才解压的`cnDNN`中的文件位置

`device`用来设置使用`cpu`还是`gpu`运行(`cpu`或`cuda`)



### GPU测试代码

```python
from theano import function, config, shared, tensor
import numpy
import time

vlen = 10 * 30 * 768  # 10 x #cores x # threads per core
iters = 1000

rng = numpy.random.RandomState(22)
x = shared(numpy.asarray(rng.rand(vlen), config.floatX))
f = function([], tensor.exp(x))
print(f.maker.fgraph.toposort())
t0 = time.time()
for i in range(iters):
    r = f()
t1 = time.time()
print("Looping %d times took %f seconds" % (iters, t1 - t0))
print("Result is %s" % (r,))
if numpy.any([isinstance(x.op, tensor.Elemwise) and
              ('Gpu' not in type(x.op).__name__)
              for x in f.maker.fgraph.toposort()]):
    print('Used the cpu')
else:
    print('Used the gpu')
```



### Lasagne

```shell
git clone https://github.com/Lasagne/Lasagne.git
cd Lasagne
pip install -r requirements.txt
pip install --editable .
```



## OK

理论上现在就能用了。

**重点**：

- Python 3.5
- CUDA 9.0
- cnDNN 7.0.5