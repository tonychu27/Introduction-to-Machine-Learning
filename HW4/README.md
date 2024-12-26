
## Please use conda to build the environment

### Require Python Version is 3.9

For example, `conda create --name HW4 python=3.9`

Or you can simply build the environment from the [envrionment.yml](./environment.yml)

`conda env create -f environment.yml`

Then

`conda activate HW4`

### Install require packages

```bash
pip install numpy
pip install tqdm
pip install pandas
pip install tensorflow
pip install pillow
pip install matplotlib
pip install scikit-learn
```

### Execution

`python inference.py`

Please ensure that you have downloaded the model's weight from my [google drive](./110700045_weight.txt)

### Result

After the script is done, prediction of test images will be stored in `prediction.csv`