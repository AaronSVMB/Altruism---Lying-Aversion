"""
Clean Chinese Data set and translate responses to english
"""

#=============================================================
# Imports
#=============================================================

import pandas as pd 

#=============================================================
# Chinese Data set Cleaning
#=============================================================

def clean_consent_chinese(data: pd.DataFrame):
    """
    Removes rows where the Consent column value is 'No' and returns 
    the cleaned dataset along with the count of dropped observations.
    
    Parameters:
    - data: pandas DataFrame containing the survey data.
    
    Returns:
    - cleaned_data: pandas DataFrame with rows removed based on the Consent column criteria.
    - dropped_count: int, number of observations dropped.
    """
    # Removing the first two rows which are metadata and not actual data
    data = data.iloc[2:]
    
    # Filtering out rows where Consent is 'No' (否)
    initial_count = data.shape[0]
    cleaned_data = data[data['Consent'] != '否']
    final_count = cleaned_data.shape[0]
    dropped_count = initial_count - final_count
    
    return cleaned_data, dropped_count

def combine_identical_columns_chinese(data: pd.DataFrame):
    """
    Combines pairs of identical columns into one for each variable respectively. 
    Each participant has at most one observation in the pairs of columns.
    
    Parameters:
    - data: pandas DataFrame containing the cleaned survey data.
    
    Returns:
    - combined_data: pandas DataFrame with combined columns.
    """
    # Columns to combine
    column_pairs = [
        ('Min Tokens', 'Min Tokens.1'),
        ('Max Tokens', 'Max Tokens.1'),
        ('SM', 'SM.1'),
        ('University', 'University.1'),
        ('decision', 'decision.1'),
        ('TimerInstrucs_First Click', 'TimerInstrucs_First Click.1'),
        ('TimerInstrucs_Last Click', 'TimerInstrucs_Last Click.1'),
        ('TimerInstrucs_Page Submit', 'TimerInstrucs_Page Submit.1'),
        ('TimerInstrucs_Click Count', 'TimerInstrucs_Click Count.1'),
        ('TimerInstrucsCont_First Click', 'TimerInstrucsCont_First Click.1'),
        ('TimerInstrucsCont_Last Click', 'TimerInstrucsCont_Last Click.1'),
        ('TimerInstrucsCont_Page Submit', 'TimerInstrucsCont_Page Submit.1'),
        ('TimerInstrucsCont_Click Count', 'TimerInstrucsCont_Click Count.1')
    ]
    
    for col1, col2 in column_pairs:
        # For each pair of columns, fill NA values in the first column with values from the second, then drop the second
        data[col1] = data[col1].fillna(data[col2])
        data.drop(columns=[col2], inplace=True)
    
    return data

def deduplicate_by_phone_number(data: pd.DataFrame):
    """
    Deduplicates the dataset based on 'phone#', keeping only the first entry for each recruiter
    based on the earliest 'StartDate'.
    
    Parameters:
    - data: pandas DataFrame, the cleaned dataset with combined columns.
    
    Returns:
    - deduplicated_data: pandas DataFrame, the dataset after deduplication.
    """
    # Ensure 'StartDate' is in datetime format for correct sorting
    data['StartDate'] = pd.to_datetime(data['StartDate'])

    # First, filter out rows where 'decision' is NaN
    data = data.dropna(subset=['decision'])

    # Filter our rows where 'Phone#' is NaN
    data = data.dropna(subset=['Phone#'])
    
    # Sort by 'StartDate' to ensure that the first entry per 'recruiter_id' is the earliest
    sorted_data = data.sort_values(by='StartDate')
    
    # Drop duplicates based on 'recruiter_id', keeping the first (earliest) entry
    deduplicated_data = sorted_data.drop_duplicates(subset='Phone#', keep='first')
    
    return deduplicated_data


def data_cleaning_one_click_chinese(data_path: str, csv_name: str):
    """
    Runs all of the data cleaning and reading functions and produces a 
    *.csv file 

    :param data_path: name of the data set to be cleaned
    :param csv_name: desired name for the output cleaned dataset
    """
    data = pd.read_csv(data_path)
    cleaned_data, dropped_count = clean_consent_chinese(data)
    print(dropped_count)

    cleaned_columns_data = combine_identical_columns_chinese(cleaned_data)

    cleaned_columns_no_dups_data = deduplicate_by_phone_number(cleaned_columns_data)

    cleaned_columns_no_dups_data.to_csv(csv_name)

    return cleaned_columns_no_dups_data

#=============================================================
# Translate answers to English (Us Qualtrics Form)
#=============================================================

def translate_chinese_data(chinese_data: pd.DataFrame, file_name: str):
    """
    Take the chinese Qualtrics data and translate participant responses
    to english to analyze with the US data and match participants

    :param chinese_data: Chinese Qualtrics data
    """
    # Recode 是 to Yes 
    chinese_data['Consent'] = chinese_data['Consent'].replace('是', 'Yes')
    # Recode 查普曼 to Chapman
    chinese_data['Partner'] = chinese_data['Partner'].replace('查普曼', 'Chapman')
    # Recode 武汉 to Wuhan
    chinese_data['Partner'] = chinese_data['Partner'].replace('武汉', 'Wuhan')
    # Recode 对 to True
    chinese_data['SM'] = chinese_data['SM'].replace('对', 'True')
    # Recode 错 to False
    chinese_data['University'] = chinese_data['University'].replace('错', 'False')
    # Recode 美国 to American
    chinese_data['nationality'] = chinese_data['nationality'].replace('美国', 'American')
    # Recode 中国 to Chinese
    chinese_data['nationality'] = chinese_data['nationality'].replace('中国', 'Chinese')
    # Recode 其他 to Other
    chinese_data['nationality'] = chinese_data['nationality'].replace('其他', 'Other')
    # Recode 不想告知 to Prefer not to say
    chinese_data['nationality'] = chinese_data['nationality'].replace('不想告知', 'Prefer not to say')
    # Recode 男 to Male 
    chinese_data['gender'] = chinese_data['gender'].replace('男', 'Male')
    # Recode 女 to Female
    chinese_data['gender'] = chinese_data['gender'].replace('女', 'Female')
    # Recode 其他 to Other
    chinese_data['gender'] = chinese_data['gender'].replace('其他', 'Other')
    # Recode 不想告知 to Prefer not to say
    chinese_data['gender'] = chinese_data['gender'].replace('不想告知', 'Prefer not to say')
    # Recode 我了解了。 to I understand
    chinese_data['understand'] = chinese_data['understand'].replace('我了解了。', 'I understand')

    # 7-point survey question
    response_mapping = {
    '非常不同意': 'Strongly disagree',
    '比较不同意': 'Disagree',
    '有点不同意': 'Somewhat disagree',
    '不同意也不反对': 'Neither agree nor disagree',
    '有点同意': 'Somewhat agree',
    '比较同意': 'Agree',
    '非常同意': 'Strongly agree'
    }
    
    chinese_data['Big5 Gosling 2003_1'] = chinese_data['Big5 Gosling 2003_1'].replace(response_mapping)
    chinese_data['Big5 Gosling 2003_2'] = chinese_data['Big5 Gosling 2003_2'].replace(response_mapping)
    chinese_data['Big5 Gosling 2003_3'] = chinese_data['Big5 Gosling 2003_3'].replace(response_mapping)
    chinese_data['Big5 Gosling 2003_4'] = chinese_data['Big5 Gosling 2003_4'].replace(response_mapping)
    chinese_data['Big5 Gosling 2003_5'] = chinese_data['Big5 Gosling 2003_5'].replace(response_mapping)
    chinese_data['Big5 Gosling 2003_6'] = chinese_data['Big5 Gosling 2003_6'].replace(response_mapping)
    chinese_data['Big5 Gosling 2003_7'] = chinese_data['Big5 Gosling 2003_7'].replace(response_mapping)
    chinese_data['Big5 Gosling 2003_8'] = chinese_data['Big5 Gosling 2003_8'].replace(response_mapping)
    chinese_data['Big5 Gosling 2003_9'] = chinese_data['Big5 Gosling 2003_9'].replace(response_mapping)
    chinese_data['Big5 Gosling 2003_10'] = chinese_data['Big5 Gosling 2003_10'].replace(response_mapping)

    chinese_data.rename(columns={"Phone#": "recruiter_id"}, inplace=True)

    chinese_data.to_csv(file_name)
    return chinese_data




