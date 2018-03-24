# Comparing automatic and human evaluation of local explanations for text classification

Nguyen, NAACL 2018

## Data

### Sources

The paper uses two datasets:

1. [Movie reviews](http://www.cs.jhu.edu/~ozaidan/rationales/):
Zaidan et al. (NAACL 2007), using the data with rationales (but rationales were not used in this paper).

2. [Twenty news groups](http://qwone.com/~jason/20Newsgroups/):
The 20news-bydate version is used with the following two categories: alt.atheism and soc.religion.christian

### Processed data
The processed data, including vocabulary files and trained machine learning models, can be found in the *output_data* folder.

## Output of experiments
These files can be download [here](http://dongnguyen.nl/data/dataset-naacl2018-nguyen.zip) (75.1MB). The zip file contains two directories: experiments & output_data

## Explanations
The explanations can be generated as follows for two classifiers (LR and MLP):

### Movie

experiments.explanations_rationales() generates:

- ../experiments/rationales/rationales\_MLP\_explanations.csv
- ../experiments/rationales/rationales\_lr\_explanations.csv

### 20News
experiments.explanations_news() generates:

- ../experiments/news/news\_MLP\_explanations.csv
- ../experiments/news/news\_lr\_explanations.csv


## Crowdsourcing

### Data
- ../experiments/rationales/rationales\_cf\_responses.csv (movie reviews)
- ../experiments/news/news\_cf\_responses.csv (20news)
- ../experiments/rationales/rationales\_cf\_responses\_with\_noise.csv (movie reviews, with noise)

### Analysis
- analysis.rmd 

## Code

- src/analysis.rmd R analysis code
- src/classifierwrapper.py wrapper around keras and scikit-learn models
- src/dataset.py reading, processing and saving the datasets
- src/experiments.py main file to generate the explanations
- src/explanation_evaluation.py computes the automatic evaluation metrics
- src/explanation_test.py some tests
- src/explanation_util.py some utility methods
- src/keras_networks.py code to train the models using keras
- src/process\_crowdflower\_annotations.py computes correlations between automatic and human evaluations and prints out the results
- src/rationales_dataset.py to process the movie data
- src/twentynewsgroups_dataset.py to process the 20news data

### Libraries

- keras
- tensorflow
- numpy
- scikit learn
- LIME http://github.com/marcotcr/lime