DMC2015
=============
Outputs the probability of customer apply on the i^th product given
they applied some product.

## To Train
```
python train.py {id} -o . -m kwprd.yml -c train-prd-nov.yml -d train-prd.npy
```

## To Test
```
python test-out.py {id-timestamp} -test test-valid.npy
```

The output file can be found in {id-timestamp}/{id-timestamp}.test.o.txt

## Architecture
1. 492 words -> 50 dimension vectors
2. Sum all the keywords in a bag (still 50 dimension vectors).
3. Logistic regression for each product (whether buys a product or not,
given they applied to at least one product).
4. 5x penalty on false negative than false positive.

## Training
* Hyper-parameters of training can be found in kwprd.yml
* Trained for 300 epochs

## Results
* 79% accuracy
* 77% recall