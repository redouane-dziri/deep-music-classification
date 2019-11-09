# Music Genre Classification using CNNs

*Authors: Redouane Dziri, Arnaud Stiegler*

Base paper: 	 [Local-feature-map Integration Using CNNs for Music Genre Classification](http://www.me.cs.scitec.kobe-u.ac.jp/~takigu/pdf/2012/1037_Paper.pdf)

Data: [GTZAN dataset](http://marsyas.info/downloads/datasets.html)


**This is an ongoing project**

## Timeline

- by 11/14: write the model and launch distributed training and monitor (using TensorBoard?), tune the model
- by 11/21: evaluate like in paper, gather new data
- by 11/28: preprocess new data and run experiments with the new data and the old model,
- by 12/05: short presentation of what was accomplished and what’s left, start writing report
- by 12/12: write text-part of the model, train it and combine both
- by 12/20: finish report, demo, presentation

## Done

- separated data in train/test
- hosted the GTZAN train/test data in Cloud Storage
- pre-processed the data according to the methods set forward in the paper




## Tips

- Requires Python >= 3.6
- Before connecting to the Cloud Storage Bucket, make sure you set the `GOOGLE_APPLICATION_CREDENTIALS` environment variable to point to your Google Cloud credentials
e.g.
```
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "/Users/red/Documents/ADL/Project/My First Project-9a16875b5624.json"
```
