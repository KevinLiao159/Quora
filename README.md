<h1 align="center"> Kaggle Competition: Quora Insincere Questions Classification </h1> <br>
<p align="center">
  <a href="https://www.kaggle.com/c/quora-insincere-questions-classification">
    <img alt="Kaggle: Quora Competition" title="Kaggle: Quora Competition" src="http://www.chiranjeevivegi.com/Toxic-Comment-Challenge/img_gh/word_cloud.png">
  </a>
</p>

<!-- START doctoc generated TOC please keep comment here to allow auto update -->
<!-- DON'T EDIT THIS SECTION, INSTEAD RE-RUN doctoc TO UPDATE -->

## Table of Contents
- [Introduction](#introduction)

<!-- END doctoc generated TOC please keep comment here to allow auto update -->

## Introduction
[![Awesome](https://cdn.rawgit.com/sindresorhus/awesome/d7305f38d29fed78fa85652e3a63e154dd8e8829/media/badge.svg)](https://github.com/KevinLiao159/Quora)
![Python](https://img.shields.io/badge/python-v3.6+-blue.svg)
![Dependencies](https://img.shields.io/badge/dependencies-up%20to%20date-brightgreen.svg)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](https://opensource.org/licenses/MIT)


NOTE: add keras logo, tensorflow logo as well


## Points of attack (impacts order from big to small)
1. Embeddeding layer is the key
>90% of a model’s complexity resides in the embedding layer, focus on embeddeding layers rather than post-embedded layers. 

Usually two bi-directional LSTM/GRU + two dense is good enough. Additional dense layers, gaussian vs. spatial dropout, additional dropout layers at the dense level, attention instead of max pooling, time distributed dense layers, and more barely changed the overall score of the model.

For choosing pre-trained weights or non-pretrained, more coverage is better.
wikinews FastText > Glove > Paragram > Google News


NOTE: preprocessing was also not particularly impactful, although leaving punctuation in the embeddings for some models (with fasttext, so they could be accomodated) was helpful in stacking.
NOTE: preprocessing needs to clean up things like f.u.c.k -> fuck)

2. Single best model
First layer: concatenated fasttext and glove twitter embeddings. Fasttext vector is used by itself if there is no glove vector but not the other way around. Words without word vectors are replaced with a word vector for a word "something". Also, I added additional value that was set to 1 if a word was written in all capital letters and 0 otherwise.

Second layer: SpatialDropout1D(0.5)

Third layer: Bidirectional CuDNNLSTM with a kernel size 40. I found out that LSTM as a first layer works better than GRU.

Fourth layer: Bidirectional CuDNNGRU with a kernel size 40.

Fifth layer: A concatenation of the last state, maximum pool, average pool and two features: "Unique words rate" and "Rate of all-caps words"

Sixth layer: output dense layer.

Batch size: 512. I found that bigger batch size makes results more stable.
Epochs: 15
Sequence length: 900.
Optimizer: Adam with clipped gradient.

Fixed some misspellings with TextBlob dictionary.
Fixed misspellings by finding word vector neighborhoods.

3. Rough-bore pseudo-labelling (PL)
Use the best ensemble model to label the test samples, adding them to the train set (with shuffle?) and continue training to convergence

4. Blending, or CV stacking
For CV stacking, could use logit or LightGBM (small trees with low depth and strong l1 regularization) to stack. Tracked accuracy, log loss and AUC to decide wether or not to stack

potential models: RNNs (LSTM, GRU), CNNs (DPCNN), (word + char-ngram) + One BiGru + Four CNNs, and GBM, NbSvm, (word + char-ngram) + logistic

5. Others
  i. Many comments were toxic only in the last sentences -- adding some models trained with the ending 25-50 characters in addition to the starting 200-300 assisted our stack.
  
  ii. Some approaches struggled to deal with the “ordering” problem of words. This meant that CNN approaches were difficult to work with, as they rely on max-pooling as a crutch, whereas RNN is better

  iii. Attention Layer takes much longer to train



## Plan of attack

model_v0: strong baseline with word (char) + NbSvm

model_v1: strong baseline with word (char) + LGBM

model_v2: strong NN baseline with non-pretrained weights, word (char) + above structure w/wo attention

add kernals:
  1). EDA kernals for word clouds in toxic comments
  2). Mispelled and missing vectors from pre-trained weights

model_v31: NN with pretrained weight wikinews FastText
model_v311: NN with pretrained weight wikinews FastText and cleaning and mispell

model_v32: NN with pretrained weight wikinews Glove
model_v321: NN with pretrained weight wikinews Glove and cleaning and mispell

model_v33: NN with pretrained weight wikinews Paragram

model_v34: NN with pretrained weight wikinews Google News

model_4: concat embeddeding weights (weights from v2, FastTest, Glove) + NN

model_5: ensemble

model_6: ensemble and PL



## Kernals
https://www.kaggle.com/christofhenkel/how-to-preprocessing-when-using-embeddings

https://www.kaggle.com/sudalairajkumar/a-look-at-different-embeddings

https://www.kaggle.com/shujian/single-rnn-with-5-folds-snapshot-ensemble

https://www.kaggle.com/thebrownviking20/analyzing-quora-for-the-insinceres

https://www.kaggle.com/mjbahmani/a-data-science-framework-for-quora