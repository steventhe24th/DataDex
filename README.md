# Beehive
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
#### Combee
One Combee should represent one model and its features includes: (metrics, confusion matrix, feature importances)  
(sklearn-based)  

##### CombeKeras
One Combee should represent one model and its features includes: (metrics, confusion matrix)  
(keras-based)  
needed because Keras has different features and functions compared to sklearn  

#### Vespiqueen
One Vespiqueen should represent one episode.   
Vespiqueen memorizes the tranformation of data.    
Vespiqueen governs all the Combees.  
Can be used to pick which model is best in a Vespiqueen.

#### Beehive
Beehive governs the whole Vespiqueen.  
Beehive is used to print all the metrics across Vespiqueen (or episodes)  
Can also be used to pick new features from the last or best vespiqueen for the next episode.   

## How to Use
