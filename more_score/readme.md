This implements the 2nd step.

```
1. Given user profile P={f_1, f_2, f_3, …, f_n, d_1, d_2, …, d_m}
2. Calculate scores Filtering for each service in service table S={s_1, s_2, …, s_x} by matching user feature to service features.
3. For each feature f∈P:
  a. Obtain a table Rf  from the review table R so that f is included in the user features in R.
  b. Calculate scores for each service in Rf  using review score and matching score
4. Combine all the scores above (with normalization and weighting) obtain ratings for all services in the case of user profile P. Find top k and return.
```

### Data

Single user data: cooked up

Service features: preprocessed using keyword extracter, a bunch of keyword with "importancy rate"

### Cosine similarity:

> We can use the Cosine Similarity algorithm to work out the similarity between two things. We might then use the computed similarity as part of a recommendation query. For example, to get movie recommendations based on the preferences of users who have given similar ratings to other movies that you’ve seen.
> Checkout this [tutorial](https://www.machinelearningplus.com/nlp/cosine-similarity/)
