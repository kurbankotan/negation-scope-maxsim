| model | condition | runs | Perm SS_bal | seen | unseen | flip_O_N | flip_O_S | Fresh SS | seen | unseen | NevIR all | nested | complement | equal | SciFact nDCG@10 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| answerai-colbert-small-v1 | baseline | 1 | -0.5 | 0.0 | -1.2 | 4.4 | 4.7 | -2.5 | -2.5 | -2.5 | 21.7 | 23.9 | 21.4 | 22.2 | 74.2 |
| answerai-colbert-small-v1 | finetuned/negpairs | 3 | 11.3 | 12.5 | 9.9 | 47.6 | 37.8 | 57.5 | 59.4 | 55.6 | 28.6 | 27.2 | 28.9 | 25.9 | 67.8 |
| answerai-colbert-small-v1 | finetuned/nevir | 3 | 2.2 | 1.5 | 3.1 | 5.4 | 3.2 | -2.4 | -2.2 | -2.5 | 33.6 | 31.0 | 33.9 | 25.9 | 72.7 |
| colbertv2 | baseline | 1 | -1.1 | 0.0 | -2.6 | 2.6 | 3.6 | -4.6 | -4.2 | -5.0 | 16.4 | 15.8 | 16.5 | 11.1 | 64.6 |
| colbertv2 | finetuned/negpairs | 3 | 11.8 | 14.5 | 7.5 | 36.3 | 25.7 | 30.3 | 43.3 | 17.2 | 24.8 | 25.2 | 24.7 | 24.1 | 53.5 |
| colbertv2 | finetuned/nevir | 3 | 2.2 | 2.6 | 1.7 | 4.3 | 2.1 | -1.9 | -3.9 | -0.0 | 20.3 | 15.8 | 21.0 | 19.4 | 63.6 |