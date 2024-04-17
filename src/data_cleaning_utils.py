"""
Data cleaning functions for the cross-cultural dictator games
"""

#=============================================================
# Imports
#=============================================================

import pandas as pd
import numpy as np

#=============================================================
# Data Cleaning Functions 
#=============================================================

def clean_consent(data: pd.DataFrame):
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
    
    # Filtering out rows where Consent is 'No'
    initial_count = data.shape[0]
    cleaned_data = data[data['Consent'] != 'No']
    final_count = cleaned_data.shape[0]
    dropped_count = initial_count - final_count
    
    return cleaned_data, dropped_count


def combine_identical_columns(data: pd.DataFrame):
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
        ('nationality', 'nationality.1'),
        ('gender', 'gender.1'),
        *[(f'Big5 Gosling 2003_{i}', f'Big5 Gosling 2003_{i}.1') for i in range(1, 11)],
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

def deduplicate_by_recruiter_id(data: pd.DataFrame):
    """
    Deduplicates the dataset based on 'recruiter_id', keeping only the first entry for each recruiter
    based on the earliest 'StartDate', ensuring that the 'decision' column is not NaN.
    
    Parameters:
    - data: pandas DataFrame, the cleaned dataset with combined columns.
    
    Returns:
    - deduplicated_data: pandas DataFrame, the dataset after deduplication.
    """
    # Ensure 'StartDate' is in datetime format for correct sorting
    data['StartDate'] = pd.to_datetime(data['StartDate'])
    
    # First, filter out rows where 'decision' is NaN
    data = data.dropna(subset=['decision'])
    
    # Sort by 'StartDate' to ensure that the first entry per 'recruiter_id' is the earliest
    sorted_data = data.sort_values(by='StartDate')
    
    # Drop duplicates based on 'recruiter_id', keeping the first (earliest) entry
    deduplicated_data = sorted_data.drop_duplicates(subset='recruiter_id', keep='first')
    
    return deduplicated_data



def data_cleaning_one_click(data_path: str, csv_name: str):
    """
    Runs all of the data cleaning and reading functions and produces a 
    *.csv file 

    :param data_path: name of the data set to be cleaned
    :param csv_name: desired name for the output cleaned dataset
    """
    data = pd.read_csv(data_path)
    cleaned_data, dropped_count = clean_consent(data)
    print(dropped_count)

    cleaned_columns_data = combine_identical_columns(cleaned_data)

    cleaned_columns_no_dups_data = deduplicate_by_recruiter_id(cleaned_columns_data)

    cleaned_columns_no_dups_data.to_csv(csv_name)

    return cleaned_columns_no_dups_data
