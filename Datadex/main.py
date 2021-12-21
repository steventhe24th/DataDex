try:
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
  
except Exception as e:
    print(f'[DataDex - LibError] core library not found {e}.\n please run: pip3 install scikit-learn pandas')


try:
    #NLP
    import spacy
    from spacy.lang.en.stop_words import STOP_WORDS
    import tensorflow_text as text
    from sklearn.feature_extraction.text import CountVectorizer
    #NLP
except Exception as e:
    print(f'[DataDex - LibError] NLP library is not found {e}.\n please run: pip3 install spacy tensorflow_text | python3 -m spacy download en_core_web_sm \n. Otherwise, no NLP preprocessing.')

MODEL_DICT = {'random_forest':RandomForestClassifier(random_state=173),
              'support_vector_machine':SVC(random_state=173, probability=True),
              'k_nearest_neighbor':KNeighborsClassifier(),
             }

try:
    #tensorflow
    import tensorflow as tf
    import tensorflow_hub as hub
    from tensorflow import keras
    import keras_tuner as kt
    from tensorflow.keras import optimizers
    #tensorflow


    ### initiate model
    ## nnlm tensorhub 128 normalized
    def build_nnlm_classifier():
        model = keras.Sequential()
        model.add(hub.KerasLayer('https://tfhub.dev/google/tf2-preview/nnlm-en-dim128-with-normalization/1', output_shape=[128], input_shape=[], dtype=tf.string))
        model.add(keras.layers.Dense(16, activation='relu'))
        model.add(keras.layers.Dense(5, activation='sigmoid'))

        model.compile(optimizer= 'rmsprop', loss= 'binary_crossentropy', metrics=[tf.keras.metrics.Precision(),tf.keras.metrics.Recall()])
        return model

    ## bert model tensorhub
    def build_classifier_model():
        text_input = tf.keras.layers.Input(shape=(), dtype=tf.string, name='text')
        preprocessing_layer = hub.KerasLayer("https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3", name='preprocessing')
        encoder_inputs = preprocessing_layer(text_input)
        encoder = hub.KerasLayer("https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-2_H-128_A-2/2", trainable=True, name='BERT_encoder')
#         encoder = hub.KerasLayer("https://tfhub.dev/google/experts/bert/wiki_books/2", trainable=True, name='BERT_encoder')
        outputs = encoder(encoder_inputs)
        net = outputs['pooled_output']
        net = tf.keras.layers.Dense(5, activation='sigmoid', name='classifier')(net)
        model = tf.keras.Model(text_input, net)
        model.compile(optimizer= 'rmsprop', loss= 'binary_crossentropy', metrics=[tf.keras.metrics.Precision(),tf.keras.metrics.Recall()])
        return model

    #basic neural net
    def build_basic_neural_net_model():
        model = keras.Sequential()
        model.add(keras.layers.Dense(16, activation='relu'))
        model.add(keras.layers.Dense(4, activation='sigmoid'))
        
        model.compile(optimizer= 'adam', loss= 'binary_crossentropy')
        return model

    # add the model to model dict
    print("[Datadex - Info] setting up adventure. please wait approx. 2 minutes")
#     MODEL_DICT['nnlm_128_hub'] = build_nnlm_classifier()
    MODEL_DICT['bert'] = build_classifier_model()
#     MODEL_DICT['neural_net'] = build_basic_neural_net_model() 
    ###

except Exception as e:
    print(f'[DataDex - LibError] library does not exist:{e}.please run: pip3 install tensorflow keras_tuner.\n Otherwise, Tensorflow is not loaded and CombeeKeras is not functional.')
finally:
    print("[Datadex - Info] adventure is ready.")

class Diglett:
    def __init__(self, dataframe= None, target_col= None, file_folder = None):
        self.df = dataframe
        self.target_col = target_col
        if dataframe is not None:
            self.columns = self.df.columns
            self.categorical_col = self.df.select_dtypes(include=['object']).columns
            self.numerical_col = self.df.select_dtypes(include=['float64','int64']).columns
            self.data_col = self.df.drop(columns=[self.target_col])
            self.target_values = self.df[target_col].drop_duplicates().values
            self.datetime_col = None
        elif dataframe is None and file_folder is None:
            print('[Diglett - FileError] dataset is found empty. if data is folder-based, call load_file()')
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
            print("[Diglett - info] base folder has not yet exist. Creating the folder...  completed")
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
                    print(f"[Diglett - FileError] file {source}-{file} is corrupted or missing. exception: {e} ")
            temp_df['text'] = string_list
            temp_df['label'] = label

            df= df.append(temp_df)
        self.df = df
   
class Combee:
    """class that represents one Model"""
    def __init__(self, model_name= None, model_self= None, X_train= None, X_test= None, y_train= None,y_test= None):
        try:
            self.model = MODEL_DICT[model_name]
        except:
            if model_self is not None:
                self.model = model_self
                print('[Combee - Info] user-defined model is loaded!')
            else:
                self.model = MODEL_DICT['random_forest']
                print('[Combee - Info] model request does not exist! defaulting to random_forest.')
                
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
            print(f"[Combee - Info] {self.model_name} has no feature importance")
        print('[Combee - Info] compute metrics')
        self.compute_metrics()
        print('[Combee - Info] cross validating')
        self.compute_cross_val()
    
    def train(self):
        """train the model"""
        print(f'[Combee - Info] training {self.model_name}')
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
                print(f"[Combee - Info] {metric_key} failed to be computed. skipped")
                self.metrics[metric_key] = 'metric failure.' 
                
        self.metrics= pd.DataFrame(self.metrics, index=[1])
        
        

        
class CombeeKeras(Combee):
    def execute(self):
        print('[CombeeKeras - info] training model')
        self.train()
        print('[CombeeKeras - info] computing all info')
        self.compute_general_info()
        print('[CombeeKeras - info] compute metrics')
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
                print(f"[Combee - Info] {metric_key} failed to be computed. skipped")
                self.metrics[metric_key] = 'metric failure.' 
                
        self.metrics= pd.DataFrame(self.metrics, index=[1])


class Beehive:
    """class to contain all vespiqueens, and show each metrics"""
    def __init__(self):
        self.vespiqueen_list = []
        self.df_list = []
        self.name_list = []
        self.metric_list = []
        self.feature_list = []
        
    def reset(self):
        self.vespiqueen_list = []
        self.df_list = []
        self.name_list = []
        self.metric_list = []
        self.feature_list = []
        
    def show(self):  
        """execute iteration to compute metrics, confusion matrix, and feature importances"""
        for i in self.vespiqueen_list:
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
        
beehive = Beehive() 


class VespiqueenTools:
    def __init__(self):
        pass

    def clean(self, text):
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

        return clean_sentence

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
        print('[VespiqueenTools - Info] transforming x')
        vectorizer = CountVectorizer()
        df = self.df.drop(columns=[target_col])
        self.df = pd.DataFrame(vectorizer.fit_transform(df[df.columns[0]]).toarray())
        self.df['label'] = self.original_df['label'].values # karena direplace, ambil kolom label dari original df
        
    def transform_y(self, target_col, transform_type='binary'):
        """
        encode ytrain and ytest
        transform type: binary, one_hot_encoding
        """
        print('[VespiqueenTools - Info] transforming y')
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
            print('[Vespiqueen - Error] wrong transform type')

class Vespiqueen(VespiqueenTools):
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
        beehive.vespiqueen_list.append(self)
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
    
    def train_test_split(self, target_col:list):
        """ train test split"""
        print('[Vespiqueen - Info] train test splitting')
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
                model_instance = CombeeKeras(model_name=model, X_train=self.X_train, X_test=self.X_test, y_train=self.y_train, y_test=self.y_test)
            else:
                model_instance = Combee(model_name=model, X_train=self.X_train, X_test=self.X_test, y_train=self.y_train, y_test=self.y_test)
            model_instance.execute() 
            self.models.append(model_instance)
        print('[Vespiqueen - Info] all training completed successfully')
        print('[Vespiqueen - Info] check result in models[i].[all_info,feature_importance,confusion_matrix,metrics]')

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
            try:
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
            print(f'[Vespiqueen - FileError] folder {folder_name} does not exist! ending process.')
        
        with open(folder_name + '/profile.txt') as json_file:
            data = json.load(json_file)
            
            self.initiate_constructor(dataset_name=data['dataset_name'], selected_columns=data['selected_columns'], folder_name= data['folder_name'], target_col= data['target_col'], standardized=data['standardized'], normalized=data['normalized'], missing_dropped=data['missing_dropped'])
            
        files = os.listdir(folder_name)
    
        for file in files:
            if file.startswith('model_'):
                loaded_model = pickle.load(open(folder_name+'/'+file, 'rb'))
                self.models.append(loaded_model) 
                
        print('Vespiqueen Object Successfully Loaded!')

class Eevee:
    def __init__(self):
        pass


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