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

2. Run the classification for the assignment
```
bash run.sh
```

Alternatively, if you want to run the scripts manually, you can run the following commands in the root directory of the repository.
1. Create and activate a virtual environment
```
bash setup.sh
```

2. Run the `image_classification.py` script with the desired arguments
```
python src/image_classification.py --clf <model>
```

### Arguments
The following arguments can be passed to the `image_classification.py` script:
- `--model`: The model to use for classification. For now `logistic` and `mlp` are supported.

## Output
A classification report is saved in the `out` directory. The report contains the following metrics:
- Precision
- Recall
- F1-score
- Support

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
A logistic reggression classifier and a neural network classifier were trained and tested on the Cifar10 dataset.

The logistic regression classifier was trained specifying the following parameters:
* No regularization
* Tolerance of 0.01
* The solver used was `saga`
* The multi_class parameter was set to `multinomial`


The neural network classifier was trained specifying the following parameters:
* Two hidden layers with 64 and 10 nodes respectively
* Learning_rate was set to be adaptive
* A maximum of 50 iterations, but also allowing for early stopping

See the `sci-kit learn` documentation for more information on the parameters used.

As the data included 10 different classes, the chance level accuracy for each class is 10%. As seen in the classification reports in the `out` directory, it holds for all classes that the precision, recall and f1 score is higher for the neural network classifier as compared to the logistic classifier.
