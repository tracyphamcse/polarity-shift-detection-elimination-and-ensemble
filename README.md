# SENTIMENT ANALYSIS & POLARITY SHIFTING USING 3-STAGES MODEL

The project is inspired by [Rui Xia, et al. 2015](http://www.sentic.net/polarity-shift-detection.pdf), which provided the framework for Sentiment Analysis with Polarity Shifting using Rule-based + Statistic method.  
Dataset collected by [Blitzer et al. 2007](http://www.seas.upenn.edu/~mdredze/datasets/sentiment/). 

### How to run

* Put dataset in resoureces/unprocessed/{domain}

* Preprocess raw data:   
	>>python handle_unprocessed_data.py [domain] 

* Calculate WLLR (for first time):
	>>python wllr.py [domain]

* Detect Polarity Shifting:
	>>python polarity_shift_detector.py [domain] -train
	>>python polarity_shift_detector.py [domain] -test

* Prepare word feature(Change type of NGRAM in code):
	>>python extract_word_feature.py [domain] -train
	>>python extract_word_feature.py [domain] -test

* Training language model (Change algorithm SVM/LogisticRegression in code):
	>>python language_model_learning.py [domain] - train
	>>python language_model_learning.py [domain] - test

* Combine language model by ensemble learning:
	>>python ensemble_learning.py [domain] 

### Result

Result is presented in our thesis. 

