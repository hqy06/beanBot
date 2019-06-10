This is a simple demo of sentiment analysis using ~~PyTorch~~ `sci-kit learn`.

Currently WIP (work in progress)

Dataset used: US airline twitter from [kaggle](https://www.kaggle.com/crowdflower/twitter-airline-sentiment)

**Since log regression, kNN and SVM are performing poorly, switch to ensemble models**

Also, a simple SVM can only do binary classification. And I want to avoid the one-vs-all' approach.

## A designed workflow:

1. data preprocessing
2. tokenize
3. ~~padding + trucating~~
4. creat train/test dataset
5. ~~create dataloader(?)~~
6. ~~define RNN~~ not necessary
7. train
8. evaluate

## New concepts and terminologies encounter (may or maynot be used)

- Term Frequency & TF-IDF

- Sentiment Classification

- [Emotion Space Model [paper](https://ieeexplore.ieee.org/document/5313815)

- Feature selection: chi-squre
