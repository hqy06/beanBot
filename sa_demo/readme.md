This is a simple demo of sentiment analysis using ~~PyTorch~~ `sci-kit learn`.

What's Next: use deep learning to do the analysis. Ref [1](https://blog.usejournal.com/sentiment-classification-with-natural-language-processing-on-lstm-4dc0497c1f19) and [2](https://towardsdatascience.com/sentiment-analysis-using-lstm-step-by-step-50d074f09948)

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

- Confidence Level: [Q & A](https://datascience.stackexchange.com/questions/44215/confidence-score-for-trained-sentiment-analyser-model)
