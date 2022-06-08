class ModellingUtility:
    def __init__(self):
        pass
    
    def train(self):
        """train the model"""
        if self.existing_model == None:
          model = self.model
        else:
          model = self.existing_model
        model.fit(self.X_train, self.y_train)
        
    def save_model_to_mlflow(self):
      #needs to be converted because mlflow cant accept adataframe
      metrics_dict = {}
      for metric in self.metrics.columns:
        metrics_dict[metric] = self.metrics[metric].iloc[0]
      
      MLFlowModule().save_model_to_mlflow(model=self.model, metrics=metrics_dict, prediction=self.prediction, X_test= self.X_test)

        
    def set_confusion_matrix(self):
        """compute & set confusion matrix"""
        labels = self.model.classes_
        self.confusion_matrix = pd.DataFrame(confusion_matrix(self.y_test, self.prediction), columns=labels, index=labels)
                
        
    def set_prediction(self):
        """compute prediction probabilites and labels for confusion matrix"""
        self.prediction = self.model.predict(self.X_test)
        self.all_info = self.X_test.copy()
        self.all_info['ground_truth'] = self.y_test
        self.all_info['prediction'] = self.prediction
        
        
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
        try:
            cv_score = cross_val_score(self.model, X_combined, y_combined, cv=10)
        except Exception as e:
            print('cross validation is not valid for this model.')
            print('error:', e)
            self.metrics['min_cvscore'] = None
            self.metrics['max_cvscore'] = None
            self.metrics['mean_cvscore'] = None
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

        
class Model(ModellingUtility, Logger):
    '''excecuter or driver of training module'''
    def __init__(self, ori_df, df, process_type='undefined', existing_model =None):
        self.df = df
        
        self.ori_df = ori_df # dataframe yang memuat kolom2 categorical |dibutuhkan untuk filterout rulebase
        
        self.X_train = None
        self.X_test = None
        self.y_test = None
        self.y_train = None
        
        self.model = RandomForestClassifier()
        self.prediction = None
        self.prediction_probability = None
        self.labels = None
        
        self.confusion_matrix = None
        self.feature_importance = None
        self.all_info = None
        self.metrics_used = [accuracy_score, precision_score, recall_score, f1_score, mean_squared_error]
        self.metrics = {'Accuracy': None ,'Precision': None, 'Recall': None, 'F1 Score': None, 'Mean Squared Error': None}
    
        self.existing_model = existing_model
        self.process_type = process_type

        
    def prepare_data(self):
        '''train test split'''   
        self.rule_based_df = self.utility.apply_rule_base(self.ori_df, self.df,TARGET_COL_NAME) #from crrs_utility
        
        
        #buang smua index yang ada di dlm rule_based_df
        #mode-1: remove rule-based
        self.processed_df = self.df[~self.df.index.isin(self.rule_based_df.index)]  
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.processed_df.drop(columns=[TARGET_COL_NAME]), self.processed_df[TARGET_COL_NAME], test_size=0.33, random_state=42)          
        #mode-2: all data
#        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.df.drop(columns=[TARGET_COL_NAME]), self.df[TARGET_COL_NAME], test_size=0.33, random_state=42) # run yan ini jika ingin model tanpa rulebased dibuang atau data all| jangan lupa untuk comment self.preprocesss_df diatas dan line daiatas ini
          
        
        

      
class MLFlowModule(Logger):
    def __init__(self):
      os.environ['HTTPS_PROXY'] = ''
      os.environ['HTTP_PROXY'] = ''
      os.environ['https_proxy'] = ''
      os.environ['http_proxy'] = ''
      self.client = MlflowClient()

      
    def save_model_to_mlflow(self, model, metrics, prediction,X_test):
        with mlflow.start_run(run_name=str(model).split('()')[0]+'_'+f'{TARGET_COL_NAME}'+'_'+str(date.today())) as run:
            print('<<<', metrics)
            mlflow.log_metrics(metrics)
            mlflow.set_tag('user', 'IJE')
            prediction = model.predict(X_test)
            signature = infer_signature(X_test, prediction)
            mlflow.sklearn.log_model(model, MODEL_NAME, signature=signature)
            
        self.register_model(run= run, model_name= MODEL_NAME)
        self.do_transition(model_name=MODEL_NAME)
        
   def deploy_encoder_to_mlflow(self, model):
       with mlflow.start_run(run_name='encoder_'+str(date.today())+'_bert')  as run:
           mlflow.set_tag('user', 'IJE')
           mlflow.sklearn.log_model(model, 'crrs_encoder')
           
       self.register_model(run= run, model_name= ENCODER_NAME)
       self.do_transition(model_name=ENCODER_NAME)    
      
      
    def register_model(self, run, model_name):
      """do register model given run and model_name"""
      desc = f"model {PROJECT_NAME} on {TARGET_COL_NAME}"
      runs_uri = f"runs:/{run.info.run_id}/{model_name}"
      model_src = RunsArtifactRepository.get_underlying_uri(runs_uri)
      mv = self.client.create_model_version(model_name, model_src, run.info.run_id, description=desc)   


    def do_transition(self, model_name):
      """get latest version on staging and registered, model then do transition"""
      #check if there is any existing model in staging
      curr_staging_version = None
      curr_staging = self.client.get_latest_versions(model_name, stages=['Staging'])
      if curr_staging != []:
        curr_staging_version = curr_staging[0].version

      curr_model_version = self.client.get_latest_versions(model_name, stages=['None'])[0].version

      #demote staging to archived if exists
      if curr_staging_version != None:
        self.client.transition_model_version_stage(
            name=model_name,
            version=curr_staging_version,
            stage='Archived'
        ) 

      #promote curr model to staging      
      self.client.transition_model_version_stage(
          name=model_name,
          version=curr_model_version,
          stage='Staging'
      )       
      
      
