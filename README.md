Output the probability of customer apply on the i^th product given
they applied some product.

To train the neural net:
python train.py {id} -o . -m kwprd.yml -c train-prd-nov.yml -d train-prd.npy

To test on the validation:
python test-out.py {id-timestamp} -test test-valid.npy

The output file can be found in {id-timestamp}/{id-timestamp}.test.o.txt

Architecture:
492 words -> 50 dimension vectors
Sum all the keywords in a bag (still 50 dimension vectors).
Logistic regression for each product (whether buys a product or not,
given they applied to at least one product).
5x penalty on false negative than false positive.
Hyper-parameters of training can be viewed in kwprd.yml
Trained for 300 epochs

79% accuracy
77% recall