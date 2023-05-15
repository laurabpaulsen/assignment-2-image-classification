# Baseline classifers for Cifar10
Assignment 2 for visual analytics (S2023), building baseline classifiers on the Cifar10 dataset.

## Data
The data used for this assignment is the Cifar 10 dataset. It contains 60.000 colour images with 32x32 pixels each. The images are divided into 10 different classes, namely airplane, automobile, bird, cat, deer, dog, frog, horse, ship, and truck. 

## Usage
To produce the results for this assignment, run the following commands in the root directory of the repository
1. Create and activate a virtual environment with the required packages
```
bash setup.sh
```

2. Train and test the classifiers. This will produce a classification report for each classifier (logistic and mlp) in the `out` directory
```
bash run.sh
```
The pipeline was developed and tested on [uCloud](https://cloud.sdu.dk/app/dashboard) (Ubuntu v22.10m, Coder python v1.77.3, python v3.10.7).

## Repository overview
```
├── env
├── out
│   ├── logistic_clf_report.txt
│   └── mlp_clf_report.txt
├── src
│   ├── image_classification.py           
│   └── img_clf.py
├── README.md  
├── run.sh                                      <- Run the classification for the assignment
├── requirements.txt
└── setup.sh                                    <- Create venv and install required packages     
```

## Results
A classification report is saved in the `out` directory. The report contains the following metrics:
- Precision
- Recall
- F1-score
- Support

A logistic regression classifier and a neural network classifier were trained and tested on the Cifar10 dataset.

The logistic regression classifier was trained specifying the following parameters:
* No regularization
* Tolerance of 0.01
* The solver used was `saga`
* The multi_class parameter was set to `multinomial`

In the table below the classification report for the logistic regression classifier is shown.

|     Class    | Precision |  Recall  | F1-Score | Support |
|:------------:|:---------:|:--------:|:--------:|:-------:|
|   airplane   |    0.34   |   0.38   |   0.36   |  1000   |
| automobile  |    0.36   |   0.40   |   0.38   |  1000   |
|     bird     |    0.26   |   0.20   |   0.23   |  1000   |
|     cat      |    0.20   |   0.17   |   0.19   |  1000   |
|     deer     |    0.26   |   0.17   |   0.21   |  1000   |
|     dog      |    0.31   |   0.30   |   0.30   |  1000   |
|     frog     |    0.27   |   0.32   |   0.30   |  1000   |
|    horse     |    0.32   |   0.32   |   0.32   |  1000   |
|     ship     |    0.32   |   0.43   |   0.37   |  1000   |
|    truck     |    0.40   |   0.44   |   0.42   |  1000   |



The neural network classifier was trained specifying the following parameters:
* Two hidden layers with 64 and 10 nodes respectively
* Learning_rate was set to be adaptive
* A maximum of 50 iterations, but also allowing for early stopping
See the `sci-kit learn` documentation for more information on the parameters used.

In the table below the classification report for the neural network classifier is shown.

|     Class    | Precision |  Recall  | F1-Score | Support |
|:------------:|:---------:|:--------:|:--------:|:-------:|
|   airplane   |    0.41   |   0.43   |   0.42   |  1000   |
| automobile  |    0.49   |   0.43   |   0.46   |  1000   |
|     bird     |    0.29   |   0.32   |   0.31   |  1000   |
|     cat      |    0.25   |   0.20   |   0.22   |  1000   |
|     deer     |    0.32   |   0.27   |   0.29   |  1000   |
|     dog      |    0.44   |   0.30   |   0.36   |  1000   |
|     frog     |    0.34   |   0.57   |   0.42   |  1000   |
|    horse     |    0.44   |   0.48   |   0.46   |  1000   |
|     ship     |    0.50   |   0.43   |   0.46   |  1000   |
|    truck     |    0.47   |   0.47   |   0.47   |  1000   |



As the data included 10 different classes, the chance level accuracy for each class is 10%. As seen in the classification reports, it holds for all classes that the precision, recall and f1 score is higher for the neural network classifier as compared to the logistic classifier.
