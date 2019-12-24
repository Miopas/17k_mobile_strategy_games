# 17k_mobile_strategy_games
CS534 Final project.

## Intruduction
This dataset is from [Kaggle](https://www.kaggle.com/tristan581/17k-apple-app-store-strategy-games). 

Our task is to:
* Figure out what factors contribute to the success of strategy games;
* Predict the number of ratings of testing data.
       
## Dataset
The dataset consists of 17007 games wiht 16 features. The target is to predict the output of average user rating from 0.5 to 5.0.


## Requirement
* Python 3
* See `requirements.txt`.

## Models
- [x] Logistic Regression (baseline)
- [x] Boosting tree
- [x] SVM
- [ ] CNN
- [ ] fastText

#### Metric
- [x] auroc

#### Examples
```shell
$ sh run.sh lr
$ sh run.sh svm
$ sh run.sh bt
```
*use `classification_all.py` to plot roc curves of three models in one image.*

## Results

See `cs534_Final_17k_revised.pdf`.
