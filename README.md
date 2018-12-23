<h1 align="center"> Kaggle Competition: Quora Insincere Questions Classification </h1> <br>
<p align="center">
  <a href="https://www.kaggle.com/c/quora-insincere-questions-classification">
    <img alt="Kaggle: Quora Competition" title="Kaggle: Quora Competition" src="https://raw.githubusercontent.com/rafapetter/udacity-machine-learning-capstone/master/eda/word_cloud.png">
  </a>
</p>

<!-- START doctoc generated TOC please keep comment here to allow auto update -->
<!-- DON'T EDIT THIS SECTION, INSTEAD RE-RUN doctoc TO UPDATE -->

## Table of Contents
- [Introduction](#introduction)
- [Model Development](#model-development)
- [Kaggle Public LB Ranking](#kaggle-public-lb-ranking)
- [Reference](#reference)

<!-- END doctoc generated TOC please keep comment here to allow auto update -->

## Introduction
[![Awesome](https://cdn.rawgit.com/sindresorhus/awesome/d7305f38d29fed78fa85652e3a63e154dd8e8829/media/badge.svg)](https://github.com/KevinLiao159/Quora)
![Python](https://img.shields.io/badge/python-v3.6+-blue.svg)
![Dependencies](https://img.shields.io/badge/dependencies-up%20to%20date-brightgreen.svg)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](https://opensource.org/licenses/MIT)

[This competition](https://www.kaggle.com/c/quora-insincere-questions-classification) is sponsored by Quora. The objective is to predict whether a question asked on Quora is sincere or not. This is a kernels only comeptition with contraint of two-hour runtime.

An insincere question is defined as a question intended to make a statement rather than look for helpful answers. Some characteristics that can signify that a question is insincere:
* has a non-neutral tone
* is disparaging or inflammatory
* isn't grounded in reality
* uses sexual content

Submissions are evaluated on F1 score between the predicted and the observed targets


## Model Development
<p align="center">
  <a href="http://s8.picofile.com/file/8342707700/workflow2.png">
    <img alt="Data Science Workflow" title="Data Science Workflow" src="http://s8.picofile.com/file/8342707700/workflow2.png">
  </a>
</p>

I have a standard workflow for model development. First starts with simple linear-based model, then add complexities if needed. Eventually, I will deploy neural network models with ensemble technique for final submission. Following is each step during my model development:

1. Establish a strong baseline with the hybrid **"NB-SVM"** model [(link to model V0)](https://github.com/KevinLiao159/Quora/blob/master/src/model_v0.py)

2. Try tree-based model **LightGBM** [(link to model V1)](https://github.com/KevinLiao159/Quora/blob/master/src/model_v1.py)

3. Try a blending model: **"NB-SVM"** + **LightGBM** [(link to the blending model V11)](https://github.com/KevinLiao159/Quora/blob/master/src/model_v11.py)

4. Establish baseline for neural network model [(link to model V2)](https://github.com/KevinLiao159/Quora/blob/master/src/model_v2.py)

  - 1st layer: embedding layer without pretrained
  - 2nd layer: spatial dropout
  - 3rd layer: bidirectional with LSTM
  - 4th layer: global max pooling 1D
  - 5th layer: output dense layer

5. Try neural network model with pretrained embedding weights
I used a very similar neural network architecture like above. The only changes are 1) adding text cleaning 2). using pretrained word embedding weights

  - Neural Networks with **Glove** word embedding [(link to model V30)](https://github.com/KevinLiao159/Quora/blob/master/src/model_v30.py)
  - Neural Networks with **Paragram** word embedding [(link to model V31)](https://github.com/KevinLiao159/Quora/blob/master/src/model_v31.py)
  - Neural Networks with **FastText** word embedding [(link to model V32)](https://github.com/KevinLiao159/Quora/blob/master/src/model_v32.py)


6. Try to use Attension layer with **Glove** word embedding [(link to model V40)](https://github.com/KevinLiao159/Quora/blob/v5/src/model_v40.py)

7. Use both LSTM Attention and CapsNet [(link to model V5)](https://github.com/KevinLiao159/Quora/blob/v4/kernels/submission_v50.py) 


## Kaggle Public LeaderBoard Ranking

| model | public score | public leaderboard | 
|---|---|---|
| model V0 | 0.641 | 1600th (*top66%*)|
| model V30 | 0.683 | 1075th (*top40%*)|
| model V40 | 0.690 | 700th (*top28%*)|
| model V50 | 0.697 | 91th (*top4%*)|


## Reference
https://www.kaggle.com/fizzbuzz/beginner-s-guide-to-capsule-networks

https://www.kaggle.com/ashishpatel26/nlp-text-analytics-solution-quora

https://www.kaggle.com/gmhost/gru-capsule

https://www.kaggle.com/larryfreeman/toxic-comments-code-for-alexander-s-9872-model

https://www.kaggle.com/shujian/single-rnn-with-5-folds-snapshot-ensemble

https://www.kaggle.com/thebrownviking20/analyzing-quora-for-the-insinceres

https://www.kaggle.com/mjbahmani/a-data-science-framework-for-quora

https://www.kaggle.com/christofhenkel/how-to-preprocessing-when-using-embeddings

https://www.kaggle.com/sudalairajkumar/a-look-at-different-embeddings
