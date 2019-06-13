This implements the 3rd step with a dummy dataset.

```
1. Given user profile P={f_1, f_2, f_3, …, f_n, d_1, d_2, …, d_m}
2. Calculate scores Filtering for each service in service table S={s_1, s_2, …, s_x} by matching user feature to service features.
3. For each feature f∈P:
  a. Obtain a table Rf  from the review table R so that f is included in the user features in R.
  b. Calculate scores for each service in Rf  using review score and matching score
4. Combine all the scores above (with normalization and weighting) obtain ratings for all services in the case of user profile P. Find top k and return.
```

Dummy Dataset:

sID | ufeature1 | ufeature2 | ufeature3 | ufeature4 | review_text| service_score | match_score

Online excel-to-csv converter: [link](https://www.beautifyconverter.com/excel-to-csv-converter.php)
