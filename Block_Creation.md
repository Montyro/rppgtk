# How to add new blocks

For this example, code examples are shown using code from the yolov3 block.


First, decide on which module category your new block fits better. In this case, yolov3 is a detector and we will be using it to detect faces, so we would create the .py file inside modules/facedetection. We could also create a new folder inside modules if we wanted to create a different type of block.

We include a new_block_template.py you can copy if you prefer to, in this case we show how to create it while explaining every line.

First we import the Block abstract class, which will be implemented by our block.

``` python

from modules.block import Block #import the abstract block class

```

The next step would be importing the packages that our custom block will be using:

``` python

import cv2 #to work with images
from port modules.tools.facedetection.yolo import yolo #import the yolo class that we are going to embed in this block.

```

Now, we define our block class, and implement our Init function, with our desired additional parameters:

``` python

class YoloFD (Block):

    def __init__(self):
        super(YoloFD,self).__init__()
        self.detector = Yolo()
        

```

Now, it's time to implement the abstract methods. We will start with the implementation of the process method, which is were the computation part of our block is performed.

```python
 def Process(self):

        self.outputs = [] #prepare the outputs attribute of the block

        for input in self.inputs:
            # Create a 4D blob from a frame.
            blob = cv2.dnn.blobFromImage(input, 1 / 255, (IMG_WIDTH, IMG_HEIGHT),
                                        [0, 0, 0], 1, crop=False)

            # Sets the input to the network
            self.detector.setInput(blob)

            # Runs the forward pass to get output of the output layers
            outs = self.detector.forward(get_outputs_names(self.detector))

            # Remove the bounding boxes with low confidence
            face = self.get_face(input, outs, CONF_THRESHOLD, NMS_THRESHOLD)
            
            self.outputs.append(face)

```

Once we have our process function, we can implement a way to visualize our block output directly after its execution. By default, visualization is disabled.

```python
  def Visualize(self):
        #Just quickly show outputs to check if there was some miss-detection

        rate = len(self.outputs)*100/len(self.inputs)

        for output in self.outputs:
            cv2.imshow("YOLO Output. Detection Rate:  {}%".format(rate),output)
            cv2.waitKey(1)

        return

```

At last, we have to implement the BlockInfo method. This allows the library to obtain a description of the block (for visualization purposes).

```python
def BlockInfo(self):
        return "YoloV3 Face Detection"
```