#!/usr/bin/env python
# coding: utf-8

# ## 1. Execute imports

# In[1]:


import warnings
warnings.filterwarnings('ignore')
import pandas as pd
import numpy as np
from patsy import ModelDesc, dmatrices, dmatrix, demo_data
import re
import pprint
import json
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB


# ## 2. Create complex operations dict

# In[2]:


# TODO: add more complex operations from numpy
COMPLEX_OPERATIONS = {
    'cos': 'np.cos',
    'tan': 'np.tan',
    'log': 'np.log',
    'log10': 'np.log10',
    'log2': 'np.log2',
    'min': 'np.min',
    'max': 'np.max',
    'pi': 'np.pi'
}

class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


# ## 3. Execute functions

# In[3]:


def add_blank_spaces_to_formula(formula: str) -> str:
    new = ''
    for index, element in enumerate(formula):
        next_idx = index + 1
        if next_idx < len(formula):
            if not re.match('\w', formula[index+1]):
                new += element + ' '
            else:
                new += element
        else:
            new += element + ' '
    return new


# In[4]:


def clean_formula(formula: str) -> str:
    result = formula
    for operation in COMPLEX_OPERATIONS:
        if(operation in formula):
            result = result.replace(operation, "")
    return result

def get_formula_variables(formula: str):
  '''
  Returns a list of every variable (non repeated) from the formula
  '''
  cleaned_formula = clean_formula(formula)
  return sorted(list(set("".join(re.findall("[a-zA-Z]+", cleaned_formula)))))

def group_columns(formula: str, data: pd.DataFrame):
  # get number of variables inside formula
  # convert string to set that only holds unique elements
  characters = get_formula_variables(formula=formula)

  # get dataset number of columns
  columns = len(data.columns)
  columns_lst = list(data.columns)
  characters_len = len(characters)

  result = []
  
  # column by column
  for i in range(0, columns):  
    # current column + 1 and substract 1 from characters so we don't count current character
    for j in range(i+1, columns, characters_len-1):
      column_variables = [columns_lst[i]]
      column_variables.extend(columns_lst[j:j+(characters_len-1)])
      # compare numbers and group columns by number of variables inside the formula
      if(len(column_variables) == characters_len):
        result.append(column_variables)
  return result # grouped columns


# In[5]:


def get_formula_by_columns(formula: str, columns: list) -> dict:
  '''
  Mapping every single formula's variable to a column.
  '''
  to_replace = {}

  # formula variables
  variables = get_formula_variables(formula=formula)
  # iterate over grouped columns
  for cidx, column_group in enumerate(columns):
    formula_grouped = {}
    # iterate over variables
    for idx, variable in enumerate(variables):
      # variable paired to column name
      formula_grouped[variable] = column_group[idx]
    # every column group represents a key
    to_replace[cidx] = formula_grouped
  return to_replace


# In[6]:


def parse_formula(formula: str, formula_columns: dict) -> list:
  '''
  Parses, effectively, every grouped column to a real formula. 
  In simple words, replaces every formula variable for its paired column.
  '''
  result = []
  formula_variables = re.findall(r'\w+', formula)

  for variables_paired in formula_columns.values():
        new_formula = formula
        for variable in formula_variables:
            if variable in variables_paired:
                # we need to put a blank space after a single character, 
                # so we can identify it then with the regex
                replace_regex = f'{variable}(?:[^\w\*\\\+\(\)\-])'
                new_formula = re.sub(replace_regex, variables_paired[variable], new_formula)
#             elif variable in COMPLEX_OPERATIONS:
#                 print(f'Going to replace [{variable} for [{COMPLEX_OPERATIONS[variable]}]')
#                 new_formula = new_formula.replace(variable, COMPLEX_OPERATIONS[variable])
#                 print(f'GOING TO APPEND => [{new_formula}]')
        new_formula = new_formula.replace(" ", "")
        for key, value in COMPLEX_OPERATIONS.items():
            if key in new_formula:
                new_formula = new_formula.replace(key, value)
        
        result.append(new_formula)
  
  return result


# In[7]:


def execute_formula(formula_by_columns: list, data: pd.DataFrame) -> pd.DataFrame:
  '''
  Take every real formula and executes it via patsy dmatrix.
  Saves every formula result inside a new dataframe's column.
  '''
  new_df = data.copy()
     
  for formula_columns in formula_by_columns:
    result_items = []
    add_data = True
#     try:
    formula = "I("+formula_columns+")-1"
    result = dmatrix(formula, data, NA_action='raise')
    for item in result:
        result_items.append(item.item())
#     except:
#         # Ignore Patsy error.
#         add_data = False
        
    if add_data:
        if "np." in formula_columns:
            new_df[formula_columns.replace('np.', '')] = result_items
        else:
            new_df[formula_columns] = result_items
    else:
        print(f"{bcolors.WARNING}Your data has some invalid values. Script will ignore them and their possible result.{bcolors.ENDC}")
        
  return new_df


# In[8]:


def execute(formula_input: str, data: pd.DataFrame) -> pd.DataFrame:
    
    formula = add_blank_spaces_to_formula(formula_input.lower())
    grouped_columns = group_columns(formula, data)
    replaceable_result = get_formula_by_columns(formula, grouped_columns)
    print(f'Got formula => {formula}')
    executable_formulas = parse_formula(formula, replaceable_result)
    new_data = execute_formula(executable_formulas, data)
    
    return new_data

