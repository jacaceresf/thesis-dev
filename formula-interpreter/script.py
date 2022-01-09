def get_formula_variables(formula: str):
  '''
  Returns a list of every variable (non repeated) from the formula
  '''
  return list(set("".join(re.findall("[a-zA-Z]+", formula))))

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
    # current column + 1
    for j in range(i+1, columns, characters_len):
      lst = [columns_lst[i]]
      lst.extend(columns_lst[j:j+(characters_len-1)])
      # compare numbers and group columns by number of variables inside the formula
      if(len(lst) == characters_len):
        result.append(lst)
  return result # grouped columns

def parse_formula(formula: str, formula_columns: dict) -> list:
  '''
  Parses, effectively, every grouped column to a real formula. 
  In simple words, replaces every formula variable for its paired column.
  '''
  result = []

  for key, columns in formula_columns.items():
    new_formula = ""
    for element in formula:
      if element in columns:
        new_formula += columns[element]
      else:
        new_formula += element
    result.append(new_formula)
  
  return result

def execute_formula(formula_by_columns: list, data: pd.DataFrame) -> pd.DataFrame:
  '''
  Take every real formula and executes it via patsy dmatrix.
  Saves every formula result inside a new dataframe's column.
  '''
  new_df = data.copy()
  for formula in formula_by_columns:
    f = "I("+formula+")-1"
    print(f'Formula: [{f}]')
    result = dmatrix(f, data)
    result_items = []
    for item in result:
      result_items.append(item.item())
    new_df[formula] = result_items

  return new_df
  
formula = '(a * a) / c'


grouped_columns = group_columns(formula, x)

replaceable_result = get_formula_by_columns(formula, grouped_columns)

executable_formulas = parse_formula(formula, replaceable_result)

new_data = execute_formula(executable_formulas, x)
new_data.head(3)
