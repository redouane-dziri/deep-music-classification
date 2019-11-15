# Music Genre Classification using CNNs

*Authors: Redouane Dziri, Arnaud Stiegler*

Base paper: 	 [Local-feature-map Integration Using CNNs for Music Genre Classification](http://www.me.cs.scitec.kobe-u.ac.jp/~takigu/pdf/2012/1037_Paper.pdf)

Data: [GTZAN dataset](http://marsyas.info/downloads/datasets.html)


**This is an ongoing project**

## Timeline

- by 11/21: write the model and debug training on the sample data (fit until overfit, and try to extract as much info as possible from the data)
- by 11/28: train models on whole data, monitor, tune (also using keras-tuner?), save the models
- by 12/05: short presentation of what was accomplished and what’s left, start writing report
- by 12/12: evaluate in detail like in paper, plot graph like theirs
- by 12/20: finish report, demo, presentation

## Done

- separated data in train/test
- hosted the GTZAN train/test data in Cloud Storage
- pre-processed the data according to the methods set forward in the paper
- explored the audio data
- explored the pre-processed data
- created the different maps (spectrograms, mel-maps, time-MFCC, quantized versions of the two first, GLCMs with different angles)
- explored the features




## Tips

- Requires Python >= 3.6
- Before connecting to the Cloud Storage Bucket, make sure you set the `GOOGLE_APPLICATION_CREDENTIALS` environment variable to point to your Google Cloud credentials
e.g.
```
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "/Users/red/Documents/ADL/Project/My First Project-9a16875b5624.json"
```
