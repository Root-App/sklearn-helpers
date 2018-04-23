# sklearn-helpers

Some helper classes for using sklearn. 

This `README` is a WIP.

### How to Install

You can install this package by running these commands:

```bash
git clone https://github.com/Root-App/sklearn-helpers
cd sklearn-helpers
pip install .
```

OR

```bash
pip install git+https://github.com/Root-App/sklearn-helpers
```

(I haven't tried this one yet, so use it at your own risk.)

### Class Descriptions

* `QuantileCalibrator` inherets from `RegressorMixIn` and 
`TransformerMixIn`. It will allow you to calibrate the output of
a regression model to training target quantiles. You can choose to
use smoothed isotonic regression, or not.
* `RandomForestTransformer` wraps `RandomForestRegressor`.
Using this allows you to put steps after the random forest in a sklearn
random forest.
* `RandomizedSearchOOB` is a modified `RandomSearchCV` which
uses out-of-bag predictions to calculate r<sup>2</sup> instead of
a hold out set.


**TODO:** Write more.

