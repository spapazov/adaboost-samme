# adaboost-samme
adaboost-samme is a "from scratch" implementation of the [Adaboost-SAMME](http://ww.web.stanford.edu/~hastie/Papers/SII-2-3-A8-Zhu.pdf) ensemble learning classifier developed by Zhu et al. which supports multi-class classification.

## Background

The Adaboost-SAMME algorithm creates a Random Forest model ![formula](https://render.githubusercontent.com/render/math?math=R_f(X)) which is composed of a series of weak classifiers ![formula](https://render.githubusercontent.com/render/math?math=c_1,c_2,c_3,...,c_n). Each classifier is assigned a weight and the ![formula](https://render.githubusercontent.com/render/math?math=\beta_1,\beta_2,\beta_3,...,\beta_n) and the final output of the model is defined to be:

![formula](https://render.githubusercontent.com/render/math?math=\Large%20R_f(X)=argmax(k)\sum_{m=1}^M\beta_m\cdot\mathbb{I}[c_m(X)=k])

Where ![formula](https://render.githubusercontent.com/render/math?math=k) is the and element of the set of possible labels. We can visualze the model formed by using the diagram below, also noting that each weak classifer carries its associated weight ![formula](https://render.githubusercontent.com/render/math?math=\beta_i):


![image](https://www.researchgate.net/profile/Antti-Jussi_Tahvanainen/publication/305501731/figure/fig42/AS:614222060855299@1523453359099/Underlying-structure-of-AdaBoost-and-Random-Forest-Models.png)

The algorithm applied to generate this Random Forest assigns initial weights to each data point in the training set. Using these weights and error term is defined and the respective weight of each newly learned weak classifier is assigned. The pseudocode is provided below:

![image](https://i.stack.imgur.com/5b2VM.png)


## Installation
Clone the repo and add the class to your project as follows:

``` python
from adaboost-samme.adaboost_samme import BoostedDT
```

## Usage

``` python
#fit classifier to the data, specify boosting iterations and depth
adaboost_samme = BoostedDT(numBoostingIters=100, maxTreeDepth = 3)

#fit to pandas data frame
adaboost_samme.fit(X, y)

#predict, outputs pandas data frame
pred = adaboost_samme.predict(X)
```
