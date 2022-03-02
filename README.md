# DataDex
A framework for feature exploration in Data Science

## Background
What do we do when we finish one episode of feature exploration in a jupyter notebook?  
We overwrite the notebook or clone the notebook for the next episode of feature exploration.  
  
I say this is inefficient, thus this project.  
  
## Purpose
To help documentation for grouping all episode of feature exploration in a single notebook.  

## Project Elaboration
### terms
episode: one set of feature trained on a model.  
e.g.  
episode_1 = (sepal_length, sepal_width) | (target_class)  
episode_2 = (sepal_length, sepal_width, petal_length, petal_width) | (target_class)  
episode_3 = (sepal_length_normalized, sepal_width_normalized, petal_length_normalized, petal_width_normalized) | (target_class)  

### get to know the classes
#### Model
One Model should represent one model and its features includes: (metrics, confusion matrix, feature importances)  
(sklearn-based)  

##### ModelKeras
One Model  should represent one model and its features includes: (metrics, confusion matrix)  
(keras-based)  
needed because Keras has different features and functions compared to sklearn  

#### DataLoaded 
dataloader functions to load file-based data convert to necessary pandas dataframe vice versa.
necessary pandas dataframe:  
 original df  
 numerical cols  
 categorical cols  
 
#### Eevee
eevee is a string parser that converts string to json format

#### Experiment
One Experiment should represent one episode.   
Experiment memorizes the tranformation of data.    
Experiment governs all the Models.  
Can be used to pick which model is best in an Experiment.

#### Journal
Journal governs all the Experiments.  
BeeJournalhive is used to print all the metrics across Experiments (or episodes)  
Can also be used to pick new features from the last or best Experiments for the next episode.   

## How to Use


#### uploading to pypi
python setup.py sdist
twine upload dist/*
