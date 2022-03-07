# How to use the library
## Requirements

First, create either a virtualenv environment or a conda environment to install the libraries on. We developed and tested the library with python 3.6.13.

After publication you will be able to download the toolkit with:
> pip install rppgtk


**If you have a CUDA compatible GPU:**

Run:

> pip install -r requirements.txt

This will install the required pip packages. After its done, run:

> conda install cudnn=7.6.5 cudatoolkit=10.1

**If you don't have a CUDA compatible GPU**
Run:

> pip install -r requirements_cpu.txt

## Example

If you have previously worked with the Sequential module in keras, or with pytorch torch.nn module, the process will be familiar to you, if you haven't, don't worry as it is very straight forward.

In this example, we are going to define a sequence composed by a face detector, from this face we will select the forehead region, and perform an illumination rectification to reduce the noise produced by the ambient light. Then we will estimate the heart rate using frequency analysis.

``` python

from modules.facedetection import SSD
from modules.rois import RoI
from modules.dimensionality import IR
from modules.estimation import Freq
from modules.visualization import Display_Results

fd = SSD(return_background='true')
roi = RoI(region='forehead')
ir = IR()
hr_est = Freq()

def Algorithm(video): ## 

    imgs = video.frames #Get the frames from the video input

    x = fd(imgs) #Detect and stabilize face

    bg = fd.background #Get the background ('negative' of the face detection)

    x = roi(x) #Cut ROI

    ir.background_frames = bg #setup background for illumination balance block
    x = ir(x) #compensate averages with background illumination 

    x = hr_est(x) #estimate heart rate

    return x

```

Now that we have the function defined, we can apply it over a single video, or a dataset.


To evaluate on a video:

```python
from modules.datasets import rPPG_Video

video = rPPG_Video('path_to_video.mpg')

x = Algorithm(video)

Display_Results(video,x)

```

If we want to evaluate over a dataset (cohface in this example):
```python
from modules.dataset import COHFACE
from modules.estimation import Peak

gt_est = Peak() #GT in COHFACE is a PPG signal, we must obtain the GT in bpms from it

dataset = COHFACE('protocol=clean')

results = dataset.Evaluate(Algorithm, Peak) #Evaluate the videos with the algorithm and the GT with Peak module.

gt = results['GT'] #All ground truths for all windows (array of arrays)
est = results ['ESTIMATIONS'] #Estimations for all windows (array of arrays)

#We can now evaluate the performance of our algorithm:
from modules.metrics import AccWindRate

awr = AccWindRate(gt,est)
print("Percentage of accurate windows:{}%".format(awr))

```

> Percentage of accurate windows: 52.69%



