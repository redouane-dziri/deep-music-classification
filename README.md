# Music Genre Classification using CNNs

*Authors: Redouane Dziri, Arnaud Stiegler*

Base paper: 	 [Local-feature-map Integration Using CNNs for Music Genre Classification](http://www.me.cs.scitec.kobe-u.ac.jp/~takigu/pdf/2012/1037_Paper.pdf)

Data: [GTZAN dataset](http://marsyas.info/downloads/datasets.html)


**This is an ongoing project**

## Timeline

- by 12/05: short presentation of what was accomplished and what’s left, start writing report, evaluate in detail like in paper, plot graph like theirs
- by 12/12: finish report, demo, presentation

## Done

- separated data in train/test
- hosted the GTZAN train/test data in Cloud Storage
- pre-processed the data according to the methods set forward in the paper
- explored the audio data
- explored the pre-processed data
- created the different maps (spectrograms, mel-maps, time-MFCC, quantized versions of the two first, GLCMs with different angles)
- explored the features
- wrote the models and debugged training on a batch of data (fit until overfit)
- trained the models on whole data for 400 epochs, monitored and saved them




## Tips

- Requires Python >= 3.6
- Before connecting to the Cloud Storage Bucket, make sure you set the `GOOGLE_APPLICATION_CREDENTIALS` environment variable to point to your Google Cloud credentials
e.g.
```
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "/Users/red/Documents/ADL/Project/My First Project-9a16875b5624.json"
```
