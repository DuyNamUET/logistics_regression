# Logistic Regression model with Mobile Price dataset

## Package structure
```
.
+-- mobile_price
|   +-- test.csv
|   +-- train.csv
+-- f1score.py
+-- Machine_Learning.pdf
+-- main.ipynb
+-- main.py
+-- README.md
+-- refer.ipynb
+-- requirement.txt
```
The folder ```mobile_price``` contain datasets, and we work only on ```train.csv``` dataset. <br>
```f1score``` is the python file to test our functions about confusion matrix and computing the precision, recall and f1 score. <br>
```Machine_Learning.pdf``` is our report about Logistic Regression, analysis the dataset, the way which we divide the dataset for trainning and testing and the way to compute accuracy, precision, recall and f1 score. <br>
```main.py``` is the python file that we are trying to code the Binary Logistic Regression model and apply for the dataset. <br>
```main.ipynb``` is the main file running on Jupyter Notebook. We using the Logistic Regression model to predict the mobile range and compute its accuracy, precision, recall and f1 score. <br>
```refer.ipynb``` is the comparison file using another model to predict the mobile range and compare the accuracy between us. <br>
```requirement.txt``` list all required packages to run this model.
## Setting
### Environment
The package can be installed and run on Windows 10, Ubuntu or virtual environment as Anaconda.
### Install the requirements
Before run the model, we need to install all requirements via pip. <br>
* Windows 10 (Recommend Python > 3.5.x) or Anaconda
```
$ pip install requirements.txt
```
* Ubuntu
```
$ sudo apt-get update
$ sudo apt-get install python3 python3-pip
$ pip3 install requirements.txt
```
## Running
Open the Command Prompt (Windows 10) or Terminal (Ubuntu) and navigate to the ```logistics_regression``` folder.
```
$ jupyter notebook
```
When the localhost open, click ```main.ipynb``` file and run all cells until the end. <br>
Ton run another model for comparation, open ```refer.ipynb``` and run it. Model Ridge Classifier is used in this file.
