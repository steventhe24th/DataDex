import pandas as pd
import numpy as np
import pickle
import copy
import os
import json
import random

from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score,precision_score, recall_score, f1_score
from sklearn.metrics import mean_squared_error
from IPython.display import display, Markdown, HTML, clear_output
from sklearn.feature_extraction.text import CountVectorizer
from zipfile import ZipFile

#NLP
import spacy
from spacy.lang.en.stop_words import STOP_WORDS
import tensorflow_text as text
#NLP

#tensorflow
import tensorflow as tf
import tensorflow_hub as hub
from tensorflow import keras
import keras_tuner as kt
from tensorflow.keras import optimizers
#tensorflow

MODEL_DICT = {'random_forest':RandomForestClassifier(random_state=173),
              'support_vector_machine':SVC(random_state=173, probability=True),
              'k_nearest_neighbor':KNeighborsClassifier(),
             }

class Initiator:
  def __init__(self):
    # add the model to model dict
    print("[Datadex - Info] setting up adventure. please wait approx. 2 minutes")
    try:
        MODEL_DICT['nnlm_128_hub'] = self.build_nnlm_classifier()
    except Exception as e:
        print(f'model [nnlm] failed to load {e}. check network connection')
    try:
        MODEL_DICT['bert'] = self.build_classifier_model()
    except Exception as e:
        print(f'model [bert] failed to load {e}. check network connection')
    try:
        MODEL_DICT['neural_net'] = self.build_basic_neural_net_model() 
    except Exception as e:
        print(f'model [neural_net] failed to load {e}. check network connection')
    ###

  ### initiate model
  ## nnlm tensorhub 128 normalized
  def build_nnlm_classifier(self):
      model = keras.Sequential()
      model.add(hub.KerasLayer('https://tfhub.dev/google/tf2-preview/nnlm-en-dim128-with-normalization/1', output_shape=[128], input_shape=[], dtype=tf.string))
      model.add(keras.layers.Dense(16, activation='relu'))
      model.add(keras.layers.Dense(5, activation='sigmoid'))

      model.compile(optimizer= 'rmsprop', loss= 'binary_crossentropy', metrics=[tf.keras.metrics.Precision(),tf.keras.metrics.Recall()])
      return model

  ## bert model tensorhub
  def build_classifier_model(self):
      text_input = tf.keras.layers.Input(shape=(), dtype=tf.string, name='text')
      preprocessing_layer = hub.KerasLayer("https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3", name='preprocessing')
      encoder_inputs = preprocessing_layer(text_input)
      encoder = hub.KerasLayer("https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-2_H-128_A-2/2", trainable=True, name='BERT_encoder')
      outputs = encoder(encoder_inputs)
      net = outputs['pooled_output']
      net = tf.keras.layers.Dense(5, activation='sigmoid', name='classifier')(net)
      model = tf.keras.Model(text_input, net)
      model.compile(optimizer= 'rmsprop', loss= 'binary_crossentropy', metrics=[tf.keras.metrics.Precision(),tf.keras.metrics.Recall()])
      return model

  #basic neural net
  def build_basic_neural_net_model(self):
      model = keras.Sequential()
      model.add(keras.layers.Dense(16, activation='relu'))
      model.add(keras.layers.Dense(4, activation='sigmoid'))

      model.compile(optimizer= 'adam', loss= 'binary_crossentropy')
      return model

class DataLoader:
    def __init__(self, file_path= None, target_col= None, file_folder = None):
        """
        file path: path menuju ke file csv / excel
        target_col: kolom label atau target
        file folder: path menuju file berbentuk folder
        """
        
        #set base dataframe
        if '.csv' in file_path:
            try:
                dataframe = pd.read_csv(file_path)
            except Exception as e:
                raise f'file not found or corrupted: {e}'
            else:
                self.df = dataframe
        elif '.xlsx' in file_path:
            try:
                dataframe = pd.read_excel(file_path, engine='openpyxl')
            except Exception as e:
                raise f'file not found or corrupted: {e}'
            else:
                self.df = dataframe 
        #set target col
        self.target_col = target_col
        
        if dataframe is not None:
            self.columns = self.df.columns
            self.categorical_col = self.df.select_dtypes(include=['object']).columns
            self.numerical_col = self.df.select_dtypes(include=['float64','int64']).columns
            self.data_col = self.df.drop(columns=[self.target_col])
            self.target_values = self.df[target_col].drop_duplicates().values
            self.datetime_col = None
        elif dataframe is None and file_folder is None:
            print('[DataLoaded - FileError] dataset is found empty. if data is folder-based, call load_file()')
        elif file_folder is not None:
            self.load_file(path=file_folder, label_list=os.listdir(file_folder))

    def write_to_text(self, path, tag,file_name, text):
        """given text name and label create file and write"""
        f = open(f"{path}/{tag}/{file_name}.txt", "w")
        f.write(text)
        f.close()
        
    def unpack_train_test(self, random_threshold = 0.3):
        """given a dataframe, create file-based train test"""

        #validate if folder exist, else create
        base_folder = 'dataset'
        train_path = f'{base_folder}/train'
        test_path = f'{base_folder}/test'
        try:
            os.listdir(base_folder)
        except:
            print("[DataLoaded - info] base folder has not yet exist. Creating the folder...  completed")
            os.mkdir(base_folder)
            os.mkdir(train_path)
            os.mkdir(test_path)

        for tag in self.target_values:
            os.mkdir(f'{train_path}/{tag}')
            os.mkdir(f'{test_path}/{tag}')

            X_values = self.df[self.df[self.target_col] == tag][self.data_col]
            for i in range(len(X_values)):
                random_probability = random.random()
                if random_probability > random_threshold:
                    self.write(train_path,tag,i,X_values.iloc[i])
                else:
                    self.write(test_path,tag,i,X_values.iloc[i])
                    
    def load_file(self, path, label_list):
        """
        load folder transform to dataframe
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

            temp_df = pd.DataFrame() #temporary dataframe to hold each folder files, to be appended to df
            string_list = []
            for file in files:
                try:
                    f = open(source+'/'+file, "r")
                    string = f.read()
                    string_list.append(string)
                    f.close()
                except Exception as e:
                    print(f"[DataLoader - FileError] file {source}-{file} is corrupted or missing. exception: {e} ")
            temp_df['text'] = string_list
            temp_df['label'] = label

            df= df.append(temp_df)
        self.df = df

class Model:
    """class that represents one Model"""
    def __init__(self, model_name= None, model_self= None, X_train= None, X_test= None, y_train= None,y_test= None):
        try:
            self.model = MODEL_DICT[model_name]
        except:
            if model_self is not None:
                self.model = model_self
                print('[Model - Info] user-defined model is loaded!')
            else:
                self.model = MODEL_DICT['random_forest']
                print('[Model - Info] model request does not exist! defaulting to random_forest.')
                
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
            print(f"[Model - Info] {self.model_name} has no feature importance")
        print('[Model - Info] compute metrics')
        self.compute_metrics()
        print('[Model - Info] cross validating')
        self.compute_cross_val()
    
    def train(self):
        """train the model"""
        print(f'[Model - Info] training {self.model_name}')
        self.model.fit(self.X_train, self.y_train)
        self.prediction = self.model.predict(self.X_test)

    def set_confusion_matrix(self):
        """compute & set confusion matrix"""
        try: # make sure type is the same
            self.confusion_matrix = confusion_matrix(self.y_test, self.prediction, labels = self.labels)
        except:
            self.y_test = self.y_test.apply(lambda x: int(x))
            self.prediction = self.prediction.apply(lambda x: int(x))
            self.confusion_matrix = confusion_matrix(self.y_test, self.prediction)
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
        self.feature_importance = self.feature_importance.sort_values(by='importance', ascending=False).reset_index() # add reset index to get ordered list when shown

    def compute_cross_val(self):
        """cross val the model and add result """
        X_combined = pd.concat([self.X_train,self.X_test])
        y_combined = pd.concat([self.y_train,self.y_test])
        cv_score = cross_val_score(self.model, X_combined, y_combined, cv=10)
        
        self.metrics['min_cvscore'] = cv_score.min()
        self.metrics['max_cvscore'] = cv_score.max()
        self.metrics['mean_cvscore'] = cv_score.mean()
    
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
                print(f"[Model - Info] {metric_key} failed to be computed. skipped")
                self.metrics[metric_key] = 'metric failure.' 
                
        self.metrics= pd.DataFrame(self.metrics, index=[1])
        
        
class ModelKeras(Model):
    def execute(self):
        print('[ModelKeras - info] training model')
        self.train()
        print('[ModelKeras - info] computing all info')
        self.compute_general_info()
        print('[ModelKeras - info] compute metrics')
        self.compute_metrics()
        
    def train(self):
        self.model.fit(self.X_train, self.y_train, epochs=1, batch_size=32)
        self.prediction = self.model.predict(self.X_test)
        
    def compute_general_info(self):
        self.prepare_confusion_matrix() # sebab Keras tidak punya predict_proba atau method classes_
        
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
                print(f"[Model - Info] {metric_key} failed to be computed. skipped")
                self.metrics[metric_key] = 'metric failure.' 
                
        self.metrics= pd.DataFrame(self.metrics, index=[1])


class Journal:
    """class to contain all experiments, and show each metrics"""
    def __init__(self):
        self.experiment_list = []
        self.df_list = []
        self.name_list = []
        self.metric_list = []
        self.feature_list = []
        
    def reset(self):
        self.experiment_list = []
        self.df_list = []
        self.name_list = []
        self.metric_list = []
        self.feature_list = []
        
    def show(self):  
        """execute iteration to compute metrics, confusion matrix, and feature importances"""
        for i in self.experiment_list:
            self.name_list.append(i.folder_name + '| shape: ' + str(self.sum_all(i.models[0].confusion_matrix)))
            self.df_list.append(i.models[0].confusion_matrix)
            self.metric_list.append(i.models[0].metrics.values[0])
            self.feature_list.append(i.models[0].feature_importance)
            
        display(pd.DataFrame(self.metric_list, columns=['Accuracy','Precision','Recall','F1 Score','Mean Squared Error','cvMin', 'cvMax', 'cvMean'] ,index=self.name_list))

        self.display_side_by_side(self.df_list, self.name_list)
        self.display_side_by_side(self.feature_list, self.name_list)

    def sum_all(self, df):
        """given a dataframe, count all values inside that dataframe"""
        return df.apply(lambda x: sum(x), axis=1).sum()
    
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
        
journal = Journal() 


class ExperimentTools:
    def __init__(self,df):
        self.df = df

    def clean(self, text):
        not_empty = []

        for item in x:
            if item != '':
                not_empty.append(item)
        text = " ".join(not_empty)

        nlp = spacy.load("en_core_web_sm")
        string = nlp(text)
        filtered_sentence = []
        for token in string:
            word = token.text
            lexeme = nlp.vocab[word]
            if lexeme.is_stop == False:
                filtered_sentence.append(word) 
        
        string = nlp(" ".join(filtered_sentence))

        clean_sentence = []
        for sent in string.sents:
            if sent.lemma_ != '':
                clean_sentence.append(sent.lemma_)

        return " ".join(clean_sentence)

    def clean_text(self, col):
        """given df, clean column specified, return series"""
        self.df[col] = self.df[col].apply(lambda x: self.clean(x))
        full_list = []
        for i in range(len(self.df)):
            row = self.df.iloc[i]
            label = row['label']
            for sentence in row['text']:
                item = [sentence,label]
                full_list.append(item)
        self.df = pd.DataFrame(full_list, columns=self.df.columns)

    def remove_column_with_full_na(self):
        """remove column that has no value"""
        col_missing_sum = self.df.isna().sum()
        self.df = self.df.drop(columns = col_missing_sum[col_missing_sum == self.df.shape[0]].index)
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

    def transform_x(self,target_col):
        """
        vectorize data X
        target_col = label / predicting column
        """
        print('[ExperimentTools - Info] transforming x')
        vectorizer = CountVectorizer()
        df = self.df.drop(columns=[target_col])
        self.df = pd.DataFrame(vectorizer.fit_transform(df[df.columns[0]]).toarray())
        self.df['label'] = self.original_df['label'].values # karena direplace, ambil kolom label dari original df
        
    def transform_y(self, target_col, transform_type='binary'):
        """
        encode ytrain and ytest
        transform type: binary, one_hot_encoding
        """
        print('[ExperimentTools - Info] transforming y')
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
            print('[Experiment - Error] wrong transform type')
            
    def push_mlflow(self, project_name):
        for i in range(len(self.models)):
            base_model = self.models[i].model
            with mlflow.start_run(run_name=self.folder_name+"_"+str(base_model).split('(')[0]):
                mlflow.log_params(base_model.get_params())
                mlflow.log_metrics(self.models[i].metrics.iloc[0].to_dict())
                mlflow.set_tag('user', 'IJE')
                signature = infer_signature(self.models[i].X_train, base_model.predict(self.models[i].X_train))
                mlflow.sklearn.log_model(base_model, project_name, signature=signature)

class Experiment(ExperimentTools):
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
        journal.experiment_list.append(self)
        print("Experiment object created and appended to Journal")
    
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
    
    def train_test_split(self, target_col:list):
        """ train test split"""
        print('[Experiment - Info] train test splitting')
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
        if len(self.y_train.shape) == 1:
            print(pd.DataFrame([self.y_train.value_counts(), self.y_test.value_counts()], index=['train','test']))
        elif len(self.y_train.shape) == 2 and self.y_train.shape[1] > 1:
            print(pd.DataFrame([self.y_train.sum(), self.y_test.sum()], index=['train','test']))
            
    def train_models(self, model_wanted):
        """model_wanted: 
            [0] model must be selected from MODEL_DICT
            [1] True/False -> True - uses Keras | False - uses Sklearn
        """
        for model,is_keras in model_wanted:
            if is_keras == True:
                model_instance = ModelKeras(model_name=model, X_train=self.X_train, X_test=self.X_test, y_train=self.y_train, y_test=self.y_test)
            else:
                model_instance = Model(model_name=model, X_train=self.X_train, X_test=self.X_test, y_train=self.y_train, y_test=self.y_test)
            model_instance.execute() 
            self.models.append(model_instance)
        print('[Experiment - Info] all training completed successfully')
        print('[Experiment - Info] check result in models[i].[all_info,feature_importance,confusion_matrix,metrics]')

    def save(self):  
        """save selected columns, metrics, and model"""
        folder_name = self.folder_name
        #validate if folder exist, else create
        try:
            os.listdir(folder_name)
        except:
            os.mkdir(folder_name)
            
        experiment_dict = {
            'dataset_name': self.dataset_name,
            'selected_columns':self.selected_columns,
            'target_col': self.target_col,
            'missing_dropped': self.missing_dropped,
            'normalized':self.normalized,
            'standardized':self.standardized,
            'folder_name': self.folder_name
        }
        
        json_text = json.dumps(experiment_dict, indent=4, sort_keys=True, default=str)
        with open(folder_name+'/profile.txt', 'w') as f:
            f.write(json_text)
            
        if len(self.models) == 0:
            print('no model found.')
            
        for i in range(len(self.models)):
            filename = folder_name + '/model_'+str(i+1)
            try:
                pickle.dump(self.models[i], open(filename+'.sav', 'wb'))
            except: # if failed using sklearn
                self.models[i].model = None
                pickle.dump(self.models[i], open(filename+'.sav', 'wb'))
        print('Experiment object has been saved!')
            
    def load(self, folder_name):
        """load a Experiment object
        if model is keras, model object is not saved, only the result
        """
        try:
            os.listdir(folder_name)
        except:
            print(f'[Experiment - FileError] folder {folder_name} does not exist! ending process.')
        
        with open(folder_name + '/profile.txt') as json_file:
            data = json.load(json_file)
            
            self.initiate_constructor(dataset_name=data['dataset_name'], selected_columns=data['selected_columns'], folder_name= data['folder_name'], target_col= data['target_col'], standardized=data['standardized'], normalized=data['normalized'], missing_dropped=data['missing_dropped'])
            
        files = os.listdir(folder_name)
    
        for file in files:
            if file.startswith('model_'):
                loaded_model = pickle.load(open(folder_name+'/'+file, 'rb'))
                self.models.append(loaded_model) 
                
        print('Experiment Object Successfully Loaded!')

class Eevee:

    def split_brackets(self, string):
        splitted_item = []
        
        open_count = 0
        item = ""
        for char in string:
            if char == "{":
                open_count += 1
                item += char

            elif char == "}":
                open_count -=1
                item+=char
                if open_count == 0:
                    splitted_item.append(item)
                    item = ""
            else:
                item += char
        return splitted_item


    def clean_text(self, dirty_text):
        """given text, return stripped whitspace and curly brackets"""
        string = dirty_text.strip()
        string = string.replace('{','')
        string = string.replace('}','')
        string = string.replace("'",'')
        return string

    def build(self, string, is_value):
        splitted_brackets = self.split_brackets(string)

        if len(splitted_brackets) > 1:
            child_list = []
            for bracket in splitted_brackets:
                item_splitted = bracket.split(":")
                key = self.build(item_splitted[0], False)
                value = self.build(":".join(item_splitted[1:]), True)
                child_list.append({key:value})
            return child_list

        if string.count(":") == 0:
            string = self.clean_text(string)
            return string
        if string.count(":") == 1:
            string_split = string.split(":")
            key=string_split[0]
            value = self.clean_text(string_split[1])
            string_dict = {key: value}
            return string_dict
        if string.count(":") > 1 and is_value == True:
            string_split = string.split(":")
            string_split = string_split
            
            #untuk handle string value yang memiliki banyak ':'
            if isinstance(string_split[0], int) == False:
                return " ".join(string_split)

            is_head = True
            key = ""
            cumulative_string = ""
            sub_list = []
            for sub in string_split:
                if is_head == True:
                    try:
                        int(sub)
                    except:
                        continue
                    key = sub
                    is_head = False
                    continue
                if '\\n' in sub:
                    cumulative_string += sub + ' ' 
                    sub_list.append({key: cumulative_string.strip()})
                    is_head = True
                    key = ""
                    cumulative_string = ""
                else:
                    cumulative_string += sub + ' '
            return sub_list



        item_splitted = string.split(":")
        key = self.build(item_splitted[0], False)
        value = self.build(":".join(item_splitted[1:]), True)
        return {key:value}


# global function
class Utility:
    def __init__(self):
        pass
      
    def is_different(val_1, val_2):
        if val_1 != val_2: return True
        return False

    def get_all_mismatch_dtype(df_1_cols,df_2_cols):
        result = []
        for i in range(len(df_1_cols)):
            if is_different(df_1_cols[i], df_2_cols[i]):
                result.append(i)
        return result

    def df_is_exact_match(df_1, df_2):
        df_first = df_1.copy()
        df_second = df_2.copy()

        if df_first.shape != df_second.shape: return False

        mismatch_list = get_all_mismatch_dtype(df_1.dtypes, df_2.dtypes)
        mismatch_list = [df_1.columns[x] for x in mismatch_list]
        if len(mismatch_list) > 0:
            print('Warning!', 'different data types found', mismatch_list)

        if list(df_first.columns) != list(df_second.columns): return (False, 'different column names found')

        filter_first_df = df_1.drop(columns=mismatch_list).dropna()
        filter_second_df = df_2.drop(columns=mismatch_list).dropna()
        comparison_df = pd.DataFrame(filter_first_df == filter_second_df)
        true_count_in_comparison_df = comparison_df.sum().value_counts().index[0]

        first_df_shape = filter_first_df.shape[0] #first or second does not matter
        if true_count_in_comparison_df == first_df_shape: return True
          

    def display_full_df(self, df):
        """show all rows from one df"""
        display(df.style.set_table_attributes("style='display:inline'"))

    def swap(self, target_list, index_one, index_two):
        """swap element of two lists"""
        temp = target_list[index_one]
        target_list[index_one] = target_list[index_two]
        target_list[index_two] = temp

    def insert(self, target_list, target_index, index_wanted):
        """take one element and insert it to another index while removing the element"""
        temp_list = []
        last_i = 0
        for i in range(len(target_list)):
            if i < target_index:
                temp_list.append(target_list[i])
                continue
            temp_list.append(target_list[index_wanted])
            last_i = i
            break
        
        for i in range(last_i, len(target_list)):
            if i == index_wanted:
                continue
            temp_list.append(target_list[i])
        return temp_list
  
    def get_all_file_paths(self, directory):
        # initializing empty file paths list
        file_paths = []
    
        # crawling through directory and subdirectories
        for root, directories, files in os.walk(directory):
            for filename in files:
                # join the two strings in order to form the full filepath.
                filepath = os.path.join(root, filename)
                file_paths.append(filepath)
    
        # returning all file paths
        return file_paths        
    
    def zip_path(self, path, output_name):
        # path to folder which needs to be zipped
        directory = f'./{path}'
    
        # calling function to get all file paths in the directory
        file_paths = self.get_all_file_paths(directory)
    
        # printing the list of all files to be zipped
        print('Following files will be zipped:')
        for file_name in file_paths:
            print(file_name)
    
        # writing files to a zipfile
        with ZipFile(f'{output_name}.zip','w') as zip:
            # writing each file one by one
            for file in file_paths:
                zip.write(file)
    
        print('All files zipped successfully!')
      
  def check_nan(self, value):
      if isinstance(value, str):
          return False
      return np.isnan(value)
  
  def concat_df_list(self,data_dir):
    '''load all preprocessed data and concat'''
    df = pd.concat([pd.read_csv(data_dir+'/'+x) for x in os.listdir(data_dir)])
    return df
  
  def display_full_df(self, df):
    """show all rows from one df"""
    display(df.style.set_table_attributes("style='display:inline'"))

  def contains_words(self, value, word_list):
    for word in word_list:
      if word in value:
        return True
    return False
    
  def convert_value_to_target_value(self, value, target_cols, target_value):
      if value in target_cols:
          return target_value
      return value
  
  def create_zero_one_df(self, row_value):
    max_index = row_value.idxmax()
    result_list = [0 for x in range(len(row_value))]
    result_list[max_index] = 1

    return result_list      
