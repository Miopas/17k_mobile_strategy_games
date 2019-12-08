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
- [ ] auroc
- [x] accuracy

#### Examples
```shell
$ sh run.sh lr
0.3747521480502313

$ sh run.sh svm
0.38995373430270985

$ sh run.sh bt
0.39193654990085924
```

## Experiments and Analysis

By counting the frequency of average user rating, we can find that the data is imbalanced.
```
2861 4.5
990 5.0
14 1.0
60 1.5
158 2.0
317 2.5
514 3.0
925 3.5
1722 4.0
```
#### feature selection
```
$ python feature_selection.py --train_file ../data/train.csv
['Intercept', 'In-app Purchases', 'Board', 'Simulation', 'Subtitle', 'Size', 'Puzzle', 'Price', 'User Rating Count', 'Name', 'Update_Gap', 'Languages', 'age_rating', 'Action']
```

