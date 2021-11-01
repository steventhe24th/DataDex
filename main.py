import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score,precision_score, recall_score, f1_score
from sklearn.metrics import mean_squared_error
from IPython.display import display, Markdown, HTML

import spacy 
import numpy as np
import pickle
import copy
import os
import json

import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_text as text
from official.nlp import optimization
from tensorflow import keras
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier


def unpack(df,tag_list):
"""todo: remove tag list, get from column label"""

    #validate if folder exist, else create
  try:
      os.listdir('dataset')
  except:
    os.mkdir('dataset')
    os.mkdir('dataset/train')
    os.mkdir('dataset/test')

    for tag in tag_list:
      os.mkdir(f'dataset/train/{tag}')
      os.mkdir(f'dataset/test/{tag}')

  for tag in tag_list:
    temp = df[df['label'] == tag]['text']#self.df
    for i in range(len(temp)):
      random_probability = random.random()
      if random_probability > 0.3:
        f = open(f"dataset/train/{tag}/{i}.txt", "w")
        f.write(temp.iloc[i])
        f.close()
      else:
        f = open(f"dataset/test/{tag}/{i}.txt", "w")
        f.write(temp.iloc[i])
        f.close()

def load_file(path, label_list):
  """
  expected file structure: 
    folder:
      label_a/
      label_b/
      label_z/
  """
  df = pd.DataFrame()
  for label in label_list:
      source = path + '/' + label
      files = os.listdir(source)
      
      temp_df = pd.DataFrame()
      string_list = []
      for file in files:
        try:
          f = open(source+'/'+file, "r")
          string = f.read()
          string = nlp(string)
          for sent in string.sents:
            string_list.append(sent.lemma_)
          f.close()
        except:
          print(file)
      print(string_list)
      print(len(string_list))
      temp_df['text'] = string_list
      temp_df['label'] = i

      df= df.append(temp_df)
    return df #self.df = df



### initiate model
## nnlm tensorhub 128 normalized
def build_nnlm_classifier():
    model = keras.Sequential()
    model.add(hub.KerasLayer('https://tfhub.dev/google/tf2-preview/nnlm-en-dim128-with-normalization/1', output_shape=[128], input_shape=[], dtype=tf.string))
    model.add(keras.layers.Dense(16, activation='relu'))
    model.add(keras.layers.Dense(5, activation='sigmoid'))

    model.compile(optimizer= 'rmsprop', loss= 'binary_crossentropy', metrics=[tf.keras.metrics.Precision(),tf.keras.metrics.Recall()])
    return model

def build_classifier_model():
    text_input = tf.keras.layers.Input(shape=(), dtype=tf.string, name='text')
    preprocessing_layer = hub.KerasLayer("https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3", name='preprocessing')
    encoder_inputs = preprocessing_layer(text_input)
    encoder = hub.KerasLayer("https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/4", trainable=True, name='BERT_encoder')
    outputs = encoder(encoder_inputs)
    net = outputs['pooled_output']
    net = tf.keras.layers.Dense(5, activation='sigmoid', name='classifier')(net)
    model = tf.keras.Model(text_input, net)
    model.compile(optimizer= 'rmsprop', loss= 'binary_crossentropy', metrics=[tf.keras.metrics.Precision(),tf.keras.metrics.Recall()])
    return model

def build_basic_neural_net_model():
    model = keras.Sequential()
    model.add(keras.layers.Dense(16, activation='relu'))
    model.add(keras.layers.Dense(4, activation='sigmoid'))
    
    model.compile(optimizer= 'adam', loss= 'binary_crossentropy')
    return model

###

model_dict = {'random_forest':RandomForestClassifier(random_state=173),
              'support_vector_machine':SVC(random_state=173, probability=True),
              'k_nearest_neighbor':KNeighborsClassifier(),
              'nnlm_128_hub': build_nnlm_classifier(),
              'bert': build_classifier_model(),
              'neural_net': build_basic_neural_net_model()
             }

    
class Combee:
    """class that represents one Model"""
    def __init__(self, model_name= None, model_self= None, X_train= None, X_test= None, y_train= None,y_test= None):
        try:
            self.model = model_dict[model_name]
        except:
            if model_self is not None:
                self.model = model_self
                print('user-defined model is loaded!')
            else:
                self.model = model_dict['random_forest']
                print('model request does not exist! defaulting to random_forest.')
                
        self.model_name = model_name
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        self.prediction = None
        self.prediction_probability = None
        self.labels = None
        
        self.confusion_matrix = None
        self.feature_importance = None
        self.all_info = None
        self.metrics_used = [accuracy_score, precision_score, recall_score, f1_score, mean_squared_error]
        self.metrics = {'Accuracy': None ,'Precision': None, 'Recall': None, 'F1 Score': None, 'Mean Squared Error': None}
    

    def execute(self):
        """execute the whole process"""
        print('training model')
        self.train()
        print('computing all info')
        self.compute_general_info()
        print('computing feature importance')
        try:
            self.compute_feature_importance()
        except:
            print(self.model_name + 'has no feature importance')
        print('compute metrics')
        self.compute_metrics()
    
    def train(self):
        """train the model"""
        print('training' + self.model_name)
        self.model.fit(self.X_train, self.y_train)
        self.prediction = self.model.predict(self.X_test)

    def set_confusion_matrix(self):
        """compute & set confusion matrix"""
        self.confusion_matrix = confusion_matrix(self.y_test, self.prediction, labels = self.labels)
        self.confusion_matrix = pd.DataFrame(self.confusion_matrix, columns=self.labels, index=self.labels)
        
    def prepare_confusion_matrix(self):
        """check if prediction output is equal to 1 or greater. if greater convert columns to 1 then compute CM"""
        if len(self.prediction.shape) == 1:
            self.set_confusion_matrix()
        elif len(self.prediction.shape) > 1: #karena bentuk y_label multilabel, perlu di convert jadi 1 kolom dulu buat compare hasilnya di function sklearn
            self.prediction = pd.DataFrame(self.prediction).apply(lambda x:x.idxmax(), axis=1)
            self.y_test = pd.DataFrame(self.y_test).apply(lambda x:x.idxmax(), axis=1)
            self.set_confusion_matrix()
        
    def compute_general_info(self):
        """compute prediction probabilites and labels for confusion matrix"""
        self.all_info = copy.deepcopy(self.X_test)
        self.all_info['ground_truth'] = self.y_test
        self.all_info['prediction'] = self.prediction  
        self.prediction_probability = self.model.predict_proba(self.X_test)
        self.prediction_probability = pd.DataFrame(self.prediction_probability)
        self.labels = self.model.classes_
        for i in range(len(self.labels)):
            self.all_info[str(self.labels[i])+'_probability'] = np.array(self.prediction_probability[i])   
        
        self.prepare_confusion_matrix()
        
    def compute_feature_importance(self):
        """get feature importances from the model created"""
        self.feature_importance = pd.DataFrame(self.model.feature_importances_)
        self.feature_importance.columns = ['importance']
        self.feature_importance['column_name'] = self.X_train.columns
        self.feature_importance = self.feature_importance.sort_values(by='importance', ascending=False)

    def compute_metrics(self):
        """calculate metrics"""
        for i in range(len(self.metrics_used)):
            try:
                metric_key = list(self.metrics.keys())[i]
                if metric_key == 'Precision' or metric_key == 'Recall' or metric_key == 'F1 Score':
                    self.metrics[metric_key] = self.metrics_used[i](self.y_test, self.prediction, average='weighted')
                else:
                    self.metrics[metric_key] = self.metrics_used[i](self.y_test, self.prediction)
            except:
                print(metric_key + " failed to be computed")
                self.metrics[metric_key] = 'metric failure.' 

                
        self.metrics= pd.DataFrame(self.metrics, index=[1])
        
class CombeeKeras(Combee):
    def execute(self):
        print('training model')
        self.train()
        print('computing all info')
        self.compute_general_info()
        print('compute metrics')
        self.compute_metrics()
        
    def train(self):
        self.model.fit(self.X_train, self.y_train, epochs=3, batch_size=100)
        self.prediction = self.model.predict(self.X_test)
        
    def compute_general_info(self):
        self.prepare_confusion_matrix() # sebab Keras tidak punya predict_proba atau method classes_
        
beehive = []

class Vespiqueen:
    """class for holding data, its transformation/features, and clustering model"""
    # create method to initialize load, and easier process if add attribute in save
    def __init__(self, dataset_name, dataframe= None, selected_columns=None, folder_name=None):
        """
        dataframe: dataframe object
        dataset_name: dataset_name.csv
        selected_columns: all columns used in this experiment
        folder_name: folder name for output
        """
        
        self.initiate_constructor(dataframe= dataframe, dataset_name=dataset_name, selected_columns=selected_columns, folder_name=folder_name)
        beehive.append(self)
        print("Vespiqueen object created and appended to beehive")
    
    def initiate_constructor(self, dataset_name, dataframe= None, selected_columns=None, folder_name=None, models=[],missing_dropped=False,normalized=False,standardized=False, target_col=None):
        try:
            self.original_df = pd.read_csv(dataset_name)
        except:
            self.original_df = dataframe
        self.df = copy.deepcopy(self.original_df)
        self.folder_name = folder_name
        self.dataset_name = dataset_name
        self.selected_columns = selected_columns
        if self.selected_columns is not None:
            self.df = self.df[self.selected_columns] # filter based on selected_columns
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.target_col = None

        #models
        self.models = []

        #flags
        self.missing_dropped = False
        self.normalized = False
        self.standardized = False

        
    def remove_column_with_full_na(self):
        """remove column that has no value"""
        col_missing_sum = self.df.isna().sum()
        self.df = self.df.drop(columns = col_missing_sum[col_missing_sum == df.shape[0]].index)
        print("all columns with zero values has been deleted!")
        
    def remove_any_missing_row(self):
        """remove rows of any missing values"""
        print('before missing values removal')
        print(self.df.shape)
        self.df = self.df.dropna(axis=0,how='any')
        print('after missing values removal')
        print(self.df.shape)
        
        #flag missing_dropped 
        self.missing_dropped = True
    
    def train_test_split(self, target_col:list):
        """ train test split"""
        self.target_col = target_col
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.df.drop(columns=target_col), self.df[target_col], test_size=0.33, random_state=173)
        print('shape of:')
        print('================')
        print('X_train | X_test')
        print(self.X_train.shape, self.X_test.shape)
        print('y_train | y_test')
        print(self.y_train.shape, self.y_test.shape)
        print('label of:')
        print('================')
        print('y_train | y_test')
        try:
            print(pd.DataFrame([self.y_train.value_counts(), self.y_test.value_counts()], index=['train','test']))
        except:
            print("todo: print sum of 1 in each column")
        
    def transform_x(self,target_col):
        """vectorize data X"""
        vectorizer = CountVectorizer()
        df = self.df.drop(columns=[target_col])
        self.df = pd.DataFrame(vectorizer.fit_transform(df[df.columns[0]]).toarray())
        self.df['label'] = self.original_df['label'].values # karena direplace, ambil kolom label dari original df
        
    def transform_y(self, target_col, transform_type='binary'):
        """
        encode ytrain and ytest
        transform type: binary, one_hot_encoding
        """
        if transform_type == 'binary':
            encoder = LabelEncoder()
            self.df['label'] = pd.DataFrame(encoder.fit_transform(self.df[target_col]))
                
        elif transform_type == 'one_hot_encoding':
            encoder =  OneHotEncoder()
            new_label = pd.DataFrame(encoder.fit_transform(pd.DataFrame(self.df[target_col])).toarray())
            self.df = self.df.drop(columns=[target_col])
            self.df = pd.concat([self.df.reset_index(),new_label.reset_index()], axis=1)
            self.df = self.df.drop(columns=['index'])
        else:
            print('wrong transform type')
            
    def train_models(self, model_wanted):
        for model in model_wanted:
            #hard coded untuk conditional model mappingnya
            if 'neural' in model:
                model_instance = CombeeKeras(model_name=model, X_train=self.X_train, X_test=self.X_test, y_train=self.y_train, y_test=self.y_test)
            else:
                model_instance = Combee(model_name=model, X_train=self.X_train, X_test=self.X_test, y_train=self.y_train, y_test=self.y_test)
            model_instance.execute() 
            self.models.append(model_instance)
        print('all training completed successfully')
        print('check result in models[i].[all_info,feature_importance,confusion_matrix,metrics]') # todo: show attribute yang not none aja disini

    def save(self):  
        """save selected columns, metrics, and model"""
        folder_name = self.folder_name
        #validate if folder exist, else create
        try:
            os.listdir(folder_name)
        except:
            os.mkdir(folder_name)
            
        vespiqueen_dict = {
            'dataset_name': self.dataset_name,
            'selected_columns':self.selected_columns,
            'target_col': self.target_col,
            'missing_dropped': self.missing_dropped,
            'normalized':self.normalized,
            'standardized':self.standardized,
            'folder_name': self.folder_name
        }
        
        json_text = json.dumps(vespiqueen_dict, indent=4, sort_keys=True, default=str)
        with open(folder_name+'/profile.txt', 'w') as f:
            f.write(json_text)
            
        if len(self.models) == 0:
            print('no model found.')
            
        for i in range(len(self.models)):
            filename = folder_name + '/model_'+str(i+1)
            try: # for keras model
                pickle.dump(self.models[i], open(filename+'.sav', 'wb'))
            except: # if failed using sklearn
                self.models[i].model = None
                pickle.dump(self.models[i], open(filename+'.sav', 'wb'))
        print('vespiqueen object has been saved!')
            
    def load(self, folder_name):
        """load a vespiqueen object
        if model is keras, model object is not saved, only the result
        """
        try:
            os.listdir(folder_name)
        except:
            print('folder does not exist! ending process.')
        
        with open(folder_name + '/profile.txt') as json_file:
            data = json.load(json_file)
            
            self.initiate_constructor(dataset_name=data['dataset_name'], selected_columns=data['selected_columns'], folder_name= data['folder_name'], target_col= data['target_col'], standardized=data['standardized'], normalized=data['normalized'], missing_dropped=data['missing_dropped'])
            
        files = os.listdir(folder_name)
    
        for file in files:
            if file.startswith('model_'):
                loaded_model = pickle.load(open(folder_name+'/'+file, 'rb'))
                self.models.append(loaded_model) 
                
        print('Vespiqueen Object Successfully Loaded!')
        
        
class Beehive:
    def __init__(self, beehive):
        self.df_list = []
        self.name_list = []
        self.metric_list = []
        self.feature_list = []
        for i in beehive:
            self.name_list.append(i.name)
            self.df_list.append(i.models[0].confusion_matrix)
            self.metric_list.append(i.models[0].metrics.values[0])
            self.feature_list.append(i.models[0].feature_importance)
            
        display(pd.DataFrame(self.metric_list, columns=['Accuracy','Precision','Recall','F1 Score','Mean Squared Error'] ,index=self.name_list))

        self.display_side_by_side(self.df_list, self.name_list)
        self.display_side_by_side(self.feature_list, self.name_list)
        
    def display_side_by_side(self,dfs:list, captions:list, tablespacing=5):
        """Display tables side by side to save vertical space
        Input:
            dfs: list of pandas.DataFrame
            captions: list of table captions
        source: https://stackoverflow.com/questions/38783027/jupyter-notebook-display-two-pandas-tables-side-by-side
        """
        output = ""
        combined = dict(zip(captions, dfs))
        for caption, df in combined.items():
            output += df.style.set_table_attributes("style='display:inline'").set_caption(caption)._repr_html_()
            output += tablespacing * "\xa0"
        display(HTML(output))

# global function
def display_full_df(df):
    """show all rows from one df"""
    display(df.style.set_table_attributes("style='display:inline'"))
        
