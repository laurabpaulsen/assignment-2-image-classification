# running image classifications
source env/bin/activate
python3 src/image_classification.py --clf logistic
python3 src/image_classification.py --clf mlp

deactive

