# Chars74k_CNN

VGG style convolution neural network for the case sensitive character recognition Chars74k dataset. Currently gets 83.4% on holdout validation dataset of 6,220 images. Matching performance of Wang and Wu et al. 2012.  

### Architecture

The input are 64 x 64 greyscale images
4 convolution layers with filter size 3x3 and ReLU activations. Max pooling layers after every other convolution layer.
2 hidden layers with dropout and PReLU. Softmax output.

| Layer Type | Parameters |
| -----------|----------- |
| Input      | size: 64x64, channel: 1 |
| convolution| kernel: 3x3, channel: 128 |
| ReLU |  |
| convolution| kernel: 3x3, channel: 128 |
| ReLU | |
| max pool | kernel: 2x2 |
| convolution| kernel: 3x3, channel: 256 |
| ReLU |  |
| convolution| kernel: 3x3, channel: 256 |
| ReLU |  |
| max pool | kernel: 2x2 |
| fully connected | units: 2048 |
| ReLU |  |
| dropout | 0.5 |
| fully connected | units: 2048 |
| ReLU |  |
| dropout | 0.5 |
| softmax | units: 62 |

### Data augmentation

Images are randomly transformed 'on the fly' while they are being prepared in each batch. The CPU will prepare each batch while the GPU will run the previous batch through the network. 

Random rotations between -10 and 10 degrees.
Random translation between -10 and 10 pixels in any direction. 
Random zoom between factors of 1 and 1.3. 
Random shearing between -25 and 25 degrees.

### To-do

Stream data from SSD instead of holding all images in memory (need to install SSD first).
Try different network archetectures and data pre-processing.
