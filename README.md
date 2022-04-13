# 3D-CNN-PyTorch: PyTorch Implementation for 3dCNNs for Medical Images
<a href="https://opensource.org/licenses/MIT"><img src="https://img.shields.io/badge/License-MIT-yellow.svg"></a>

Keywords: Deep Learning, 3D Convolutional Neural Networks, PyTorch, Medical Images, Gray Scale Images

## Update (2022/4/13)

More 3dCNN models will be added shortly.

## Implemented models

 - Simple CNN
 - ResNet [10, 18, 34, 50, 101, 152, 200]
 - ResNetv2 [10, 18, 34, 50, 101, 152, 200]
 - ResNeXt [50, 101, 152, 200]
 - ResNeXt [50, 101, 152, 200]
 - PreActResNet [50, 101, 152, 200]
 - WideResNet [50, 101, 152, 200]
 - DenseNet [121, 169, 201]
 - SqueezeNet
 - ShuffleNet
 - ShuffleNetV2
 - MobileNet 
 - MobileNetV2
 - EfficientNet (b0-b9)

## Repository Structure

The repository is structured as follows:

All the models to run the deep-learning-based pipeline is found under the models folder.

## Requirements

* Python 3.8.5
* PyTorch 1.11.0

## Set-up

This code was developed and tested using Python 3.8.5.

For the code to run as intended, all the packages under requirements.txt should be installed. In order not to break previous installations and ensure full compatibility, it's highly recommended to create a virtual environment to run the DeepContrast pipeline in. Here follows an example of set-up using python virtualenv:

* install python's virtualenv
```
sudo pip install virtualenv
```
* parse the path to the python3 interpreter
```
export PY2PATH=$(which python3)
```
* create a virtualenv with such python3 interpreter named "venv"
(common name, already found in .gitignore)
```
virtualenv -p $PY2PATH venv 
```
* activate the virtualenv
```
source venv/bin/activate
```
At this point, (venv) should be displayed at the start of each bash line. Furthermore, the command which python3 should return a path similar to /path/to/folder/venv/bin/python3. Once the virtual environment is activated:

* once the virtualenv is activated, install the dependencies
```
pip install -r requirements.txt
```
At this stage, everything should be ready for the data to be processed by the DeepContrast pipeline. Additional details can be found in the markdown file under src.

The virtual environment can be deactivated by running:
```
deactivate
```

## Running the codes

The model generation can be run by executing:

```
python3 generate_model.py --cnn_name resnet --model_depth 101 --n_classes 2 \
--in_channels 1 --sample_size 128
```

## Disclaimer

The code and data of this repository are provided to promote reproducible research. They are not intended for clinical care or commercial use.

The software is provided "as is", without warranty of any kind, express or implied, including but not limited to the warranties of merchantability, fitness for a particular purpose and noninfringement. In no event shall the authors or copyright holders be liable for any claim, damages or other liability, whether in an action of contract, tort or otherwise, arising from, out of or in connection with the software or the use or other dealings in the software.
