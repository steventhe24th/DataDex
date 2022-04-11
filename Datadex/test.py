import unittest
import numpy as np
from main import *


class TestMainUtility(unittest.TestCase):
  
    def test_check_nan_string(self):
        utility = Utility()  
        self.assertEqual(utility.check_nan('11:58:49'), False)
        self.assertEqual(utility.check_nan("11:58:49"), False)
        self.assertEqual(utility.check_nan('test_text'), False)
        
    def test_check_nan_numeric_empty(self):
        utility = Utility()  
        self.assertEqual(utility.check_nan(np.nan), True) 
        
    def test_check_nan_numeric_integer(self):
        utility = Utility()  
        self.assertEqual(utility.check_nan(10), False)
        self.assertEqual(utility.check_nan(-1), False)
        self.assertEqual(utility.check_nan(0), False)
        self.assertEqual(utility.check_nan(-100), False)

    def test_check_concat_df_list_(self):
        utility = Utility()
        df_1 = pd.read_csv('temp_dataset/df_1.csv')
        df_2 = pd.read_csv('temp_dataset/df_2.csv')
        df_1_shape = df_1.shape
        df_2_shape = df_2.shape
        truth_concated_shape = list(df_1.shape)
        truth_concated_shape[0] += df_2_shape[0]
        truth_concated_shape = tuple(truth_concated_shape)
        
        utility.concat_df_list('temp_dataset')
        self.assertEqual(utility.concat_df_list('temp_dataset').shape, truth_concated_shape)

        
    def test_contains_words_typo(self):
        utility = Utility()
        word_list = ['apple','orange','Orange','Potato','Tomato','tomato','watermelon']
        
        self.assertEqual(utility.contains_words('apply', word_list), False)
        self.assertEqual(utility.contains_words('Apply', word_list), False)
        
        
    def test_contains_words_case_sensitive(self):
        utility = Utility()
        word_list = ['apple','orange','Orange','Potato','Tomato','tomato','watermelon']
        
        self.assertEqual(utility.contains_words('Apple', word_list), False)
        self.assertEqual(utility.contains_words('potato', word_list), False)
        
    def test_create_zero_df(self):
        utility = Utility()
        df = pd.DataFrame([[0.5,0.75,0.35],[0,0.1,0.99],[1,0,0]])
                           
        self.assertEqual(utility.create_zero_one_df(df.iloc[0]), [0,1,0])
        self.assertEqual(utility.create_zero_one_df(df.iloc[1]), [0,0,1])
        self.assertEqual(utility.create_zero_one_df(df.iloc[2]), [1,0,0])        
