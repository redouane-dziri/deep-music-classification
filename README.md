# Music Genre Classification using CNNs

*Authors: Redouane Dziri, Arnaud Stiegler*

Base paper: 	 [Local-feature-map Integration Using CNNs for Music Genre Classification](http://www.me.cs.scitec.kobe-u.ac.jp/~takigu/pdf/2012/1037_Paper.pdf)

Data: [GTZAN dataset](http://marsyas.info/downloads/datasets.html)


**This is an ongoing project**


## Tips

- Requires Python >= 3.6
- Before connecting to the Cloud Storage Bucket, make sure you set the `GOOGLE_APPLICATION_CREDENTIALS` environment variable to point to your Google Cloud credentials
e.g.
```
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "/Users/red/Documents/ADL/Project/My First Project-9a16875b5624.json"
```
## Reproducing the experiments

- Collect the GTZAN dataset and store it somewhere you can access it

In our case, we had the dataset on a Google Cloud Bucket. For all subsequent scripts, we fetch the data from the bucket and process it locally. 
- Generate train and test set using `python early_preprocessing/train_test_split.py`
- Run the preprocessing using `python feature_engineering/preprocess_full_data.py`
- Proceed to EDA on the generated feature-maps using the notebook: `feature_engineering/exploration.ipynb`

To reproduce the paper results:
- Use `training/training_GLCM.ipynb`
To train our models:
- Use `improvement/training_stacked.ipynb` or `improvement/training_stacked.ipynb`
