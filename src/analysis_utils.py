"""
Code to conduct the analysis for the Dictator Games
"""

#=============================================================
# Imports
#=============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from scipy.stats import chi2_contingency
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from scipy.stats import ks_2samp
import itertools
from scipy.stats import epps_singleton_2samp
from scipy.stats import kstest
from scipy.stats import chisquare
from scipy.stats import brunnermunzel 

#=============================================================
# Summary Statistics
#=============================================================

def summary_statistics(data: pd.DataFrame, sample: str):
    sample_df = data[data['Location'] == sample]

    # 'gender' counts
    gender_counts = sample_df['gender'].value_counts()

    # 'nationality' counts
    nationality_counts = sample_df['nationality'].value_counts()

    # 'Treatment' counts
    treatment_counts = sample_df['Treatment'].value_counts()

    # 'Partner' counts
    partner_counts = sample_df['Partner'].value_counts()

    # 'decision' counts
    decision_counts = sample_df['decision'].value_counts()

    print(f"Gender counts among {sample} participants:\n", gender_counts)
    print(f"\nNationality counts among {sample} participants:\n", nationality_counts)
    print(f"\nTreatment counts among {sample} participants:\n", treatment_counts)
    print(f"\nPartner counts among {sample} participants:\n", partner_counts)
    print(f"\Decision counts among {sample} participants:\n", decision_counts)


def summary_statistics_individual(data: pd.DataFrame, sample: str):
    sample_df = data[data['Location'] == sample]

    # 'gender' counts
    gender_counts = sample_df['gender'].value_counts()

    # 'nationality' counts
    nationality_counts = sample_df['nationality'].value_counts()

    # 'Treatment' counts
    treatment_counts = sample_df['Treatment'].value_counts()

    # 'decision' counts
    decision_counts = sample_df['decision'].value_counts()

    print(f"Gender counts among {sample} participants:\n", gender_counts)
    print(f"\nNationality counts among {sample} participants:\n", nationality_counts)
    print(f"\nTreatment counts among {sample} participants:\n", treatment_counts)
    print(f"\Decision counts among {sample} participants:\n", decision_counts)


def response_times(data: pd.DataFrame, sample: str):
    sample_df = data[data['Location'] == sample]

    mean_duration = sample_df['Duration (in seconds)'].mean()
    std_duration = sample_df['Duration (in seconds)'].std()
    median_duration = sample_df['Duration (in seconds)'].median()

    print(f"Mean Duration (in seconds): {mean_duration}")
    print(f"Standard Deviation of Duration (in seconds): {std_duration}")
    print(f"Median Duration (in seconds): {median_duration}")


def count_observations_by_combinations(df, combinations):
    counts_dict = {}
    
    for location, treatment, partner in combinations:
        # Create a filter for the current combination
        subset_df = df[(df['Location'] == location) & 
                       (df['Treatment'] == treatment) & 
                       (df['Partner'] == partner)]
        
        # Count the number of participants
        count = len(subset_df)
        
        # Add to dictionary
        counts_dict[f'Location: {location}, Treatment: {treatment}, Partner: {partner}'] = count
    
    return counts_dict


#=============================================================
# Probability Mass Functions
#=============================================================

def calculate_pmf(df, column):
    pmf = df[column].value_counts(normalize=True)
    pmf = pmf.sort_index()  # Sort by index for readability
    return pmf


def plot_pmf_by_combinations(df, combinations):

    # Initialize an empty list to store all PMFs
    all_pmfs = []

    # Calculate PMF for each combination and store in the list
    for location, treatment, partner in combinations:
        subset_df = df[(df['Location'] == location) & (df['Treatment'] == treatment) & (df['Partner'] == partner)]
        pmf = subset_df['decision'].value_counts(normalize=True).sort_index()
        all_pmfs.append(pmf)

    # Determine the global max probability for consistent y-axis scaling
    max_probability = max(pmf.max() for pmf in all_pmfs)

    # Create the figure and subplots
    fig, axs = plt.subplots(len(combinations)//2, 2, figsize=(15, 20))
    axs = axs.flatten()  # Flatten to iterate easily
    fig.subplots_adjust(hspace=0.5, wspace=0.3)

    for i, (location, treatment, partner) in enumerate(combinations):
        pmf = all_pmfs[i]
        
        # Ensure every plot has bars for decisions 1 through 5, even if some have 0 probability
        decisions = range(1, 6)  # Assuming decisions go from 1 to 5
        probabilities = [pmf.get(decision, 0) for decision in decisions]
        
        axs[i].bar(decisions, probabilities, color='skyblue')
        axs[i].set_title(f'Location: {location}, Treatment: {treatment}, Partner: {partner}')
        axs[i].set_xlabel('Decision')
        axs[i].set_ylabel('Probability')
        axs[i].set_ylim(0, max_probability + 0.05)  # Add a little padding to the max probability
        axs[i].set_xticks(decisions)

    plt.tight_layout()
    plt.show()


def plot_pmf_by_combinations_individual_task(df, combinations):   
    # Initialize an empty list to store all PMFs
    all_pmfs = []

    # Calculate PMF for each combination and store in the list
    for location, treatment in combinations:
        subset_df = df[(df['Location'] == location) & (df['Treatment'] == treatment)]
        pmf = subset_df['decision'].value_counts(normalize=True).sort_index()
        all_pmfs.append(pmf)

    # Determine the global max probability for consistent y-axis scaling
    max_probability = max(pmf.max() for pmf in all_pmfs)

    # Create the figure and subplots
    fig, axs = plt.subplots(len(combinations)//2, 2, figsize=(15, 20))
    axs = axs.flatten()  # Flatten to iterate easily
    fig.subplots_adjust(hspace=0.5, wspace=0.3)

    for i, (location, treatment) in enumerate(combinations):
        pmf = all_pmfs[i]
        
        # Ensure every plot has bars for decisions 1 through 5, even if some have 0 probability
        decisions = range(1, 6)  # Assuming decisions go from 1 to 5
        probabilities = [pmf.get(decision, 0) for decision in decisions]
        
        axs[i].bar(decisions, probabilities, color='skyblue')
        axs[i].set_title(f'Location: {location}, Treatment: {treatment}')
        axs[i].set_xlabel('Decision')
        axs[i].set_ylabel('Probability')
        axs[i].set_ylim(0, max_probability + 0.05)  # Add a little padding to the max probability
        axs[i].set_xticks(decisions)

    plt.tight_layout()
    plt.show()


def plot_pmf_by_combinations_individual_task_reversed(df, combinations):
    # Initialize an empty list to store all PMFs
    all_pmfs = []

    # Calculate PMF for each combination and store in the list
    for location, treatment in combinations:
        subset_df = df[(df['Location'] == location) & (df['Treatment'] == treatment)]
        pmf = subset_df['decision'].value_counts(normalize=True).sort_index()
        all_pmfs.append(pmf)

    # Determine the global max probability for consistent y-axis scaling
    max_probability = max(pmf.max() for pmf in all_pmfs)

    # Create the figure and subplots
    fig, axs = plt.subplots(len(combinations)//2, 2, figsize=(15, 20))
    axs = axs.flatten()  # Flatten to iterate easily
    fig.subplots_adjust(hspace=0.5, wspace=0.3)

    for i, (location, treatment) in enumerate(combinations):
        pmf = all_pmfs[i]
        
        # Ensure every plot has bars for decisions 1 through 5, even if some have 0 probability
        original_decisions = range(1, 6)  # Assuming decisions go from 1 to 5
        reversed_decisions = list(reversed(original_decisions))
        probabilities = [pmf.get(decision, 0) for decision in original_decisions]

        # Plot with reversed decisions
        axs[i].bar(reversed_decisions, probabilities, color='skyblue')
        axs[i].set_title(f'Location: {location}, Treatment: {treatment}')
        axs[i].set_xlabel('Reversed Decision')
        axs[i].set_ylabel('Probability')
        axs[i].set_ylim(0, max_probability + 0.05)  # Add a little padding to the max probability
        axs[i].set_xticks(reversed_decisions)
        axs[i].set_xticklabels(reversed_decisions)  # Update x-tick labels to reflect the reversed order

    plt.tight_layout()
    plt.show()


#=============================================================
# Statistical Tests
#=============================================================

# Chi-Squared

def perform_chi_squared_tests_for_pairs(df, group_by_col, decision_col='decision'):
    """
    Perform Chi-squared tests on the distribution of a decision variable across pairwise combinations of treatments.
    
    Parameters:
    - df: The pandas DataFrame containing your data.
    - group_by_col: Column name for grouping, typically 'Treatment' or similar.
    - decision_col: The decision column to test, defaults to 'decision'.
    
    Returns:
    - results: A list of dictionaries containing results for each pair.
    """
    results = []
    treatments = df[group_by_col].unique()
    pairs = itertools.combinations(treatments, 2)  # Get all unique pairs of treatments

    for treatment1, treatment2 in pairs:
        # Filter data for each treatment in the pair
        sub_df = df[df[group_by_col].isin([treatment1, treatment2])]
        
        # Create a contingency table
        contingency_table = pd.crosstab(sub_df[decision_col], sub_df[group_by_col])
        
        # Perform the Chi-squared test
        chi2, p, dof, expected = chi2_contingency(contingency_table)
        
        # Append the results
        results.append({
            'Treatment Pair': f'{treatment1} vs {treatment2}',
            'Chi2 Statistic': chi2,
            'P-value': p,
            #'Degrees of Freedom': dof,
            #'Expected Frequencies': expected
        })

    return results


#Create a new column for the treatment combination
def get_combination(row):
    loc = 'C' if row['Location'] == 'Chapman' else 'W'
    part = 'C' if row['Partner'] == 'Chapman' else 'W'
    return loc + 'x' + part


def multi_sample_chi_squared(data: pd.DataFrame, treatment: str):
    treatment_df = data[data['Treatment'] == treatment]

    treatment_df['Combination'] = treatment_df.apply(get_combination, axis=1)

    contingency_table = pd.crosstab(treatment_df['Combination'], treatment_df['decision'])

    data_array = contingency_table.values

    chi2, p, dof, expected = chi2_contingency(data_array)

    print("Contingency Table:\n", contingency_table)
    print("\nChi-squared value:", chi2)
    print("p-value:", p)
    print("Degrees of freedom:", dof)
    print("Expected frequencies:\n", expected)


# Epps-Singleton

def gen_epps_singleton(data: pd.DataFrame, 
                       loc1: str, game1: str, partner1: str, 
                       loc2: str, game2: str, partner2: str):
    sample1 = data[(data['Location'] == loc1) & (data['Treatment'] == game1) & (data['Partner'] == partner1)]['decision']
    sample2 = data[(data['Location'] == loc2) & (data['Treatment'] == game2) & (data['Partner'] == partner2)]['decision']

    statistic, p_value = epps_singleton_2samp(sample1, sample2)
    print(f"Epps-Singleton Test Statistic: {statistic}, P-value: {p_value}")


def gen_epps_singleton_individual(data: pd.DataFrame, 
                       loc1: str, game1: str, 
                       loc2: str, game2: str):
    sample1 = data[(data['Location'] == loc1) & (data['Treatment'] == game1)]['decision']
    sample2 = data[(data['Location'] == loc2) & (data['Treatment'] == game2)]['decision']

    statistic, p_value = epps_singleton_2samp(sample1, sample2)
    print(f"Epps-Singleton Test Statistic: {statistic}, P-value: {p_value}")


# One-Sample Chi-Squared

def gen_one_sample_chi_squared(data: pd.DataFrame, 
                               treatment: str,
                               decision_list: list):
    treatment_df = data[(data['Treatment'] == treatment) & (data['decision'].isin(decision_list))]

    num_decisions = len(decision_list)

    # Perform the Chi-squared test for each group where Treatment == 'Lie'
    results = []
    for group, group_df in treatment_df.groupby(['Location', 'Partner']):
        # Count occurrences of each decision
        observed_frequencies = group_df['decision'].value_counts().reindex(decision_list, fill_value=0)
        expected_frequencies = [len(group_df) / num_decisions] * num_decisions  # Uniform expected frequencies

        # Chi-squared test
        chi2_stat, p_value = chisquare(f_obs=observed_frequencies, f_exp=expected_frequencies)

        results.append({
            'Group': group,
            'Chi2 Statistic': chi2_stat,
            'P-value': p_value
        })

    for result in results:
        print(result)


def gen_one_sample_chi_squared_individual(data: pd.DataFrame, 
                               treatment: str,
                               decision_list: list):
    treatment_df = data[(data['Treatment'] == treatment) & (data['decision'].isin(decision_list))]

    num_decisions = len(decision_list)

    # Perform the Chi-squared test for each group where Treatment == 'Lie'
    results = []
    for group, group_df in treatment_df.groupby('Location'):
        # Count occurrences of each decision
        observed_frequencies = group_df['decision'].value_counts().reindex(decision_list, fill_value=0)
        expected_frequencies = [len(group_df) / num_decisions] * num_decisions  # Uniform expected frequencies

        # Chi-squared test
        chi2_stat, p_value = chisquare(f_obs=observed_frequencies, f_exp=expected_frequencies)

        results.append({
            'Group': group,
            'Chi2 Statistic': chi2_stat,
            'P-value': p_value
        })

    for result in results:
        print(result)


#=============================================================
# Statistical Tests – Miscellaneous
#=============================================================


def gen_tukey(data, alpha):
    tukey_results = pairwise_tukeyhsd(endog=data['decision'], groups=data['Group'], alpha=alpha)
    return tukey_results


def plot_tukey(results):
    results.plot_simultaneous(figsize=(12, 8)) 
    plt.show()


def perform_anova(df, groups):
    """
    Perform ANOVA on specified groups within the DataFrame.
    
    Parameters:
    - df: The pandas DataFrame containing your data.
    - groups: A list of tuples, where each tuple contains the filter criteria
              for a specific group (e.g., Location and Treatment).
    
    Returns:
    - The F-value and p-value from the ANOVA test.
    """
    group_data = [df[(df['Location'] == location) & (df['Treatment'] == treatment) & (df['Partner'] == partner)]['decision'] 
                  for location, treatment, partner in groups]
    f_val, p_val = stats.f_oneway(*group_data)
    return f_val, p_val


def run_ks_tests_with_correction(df, combinations):
    results = []
    n_combinations = len(combinations)
    # Calculate the number of tests for Bonferroni correction
    n_tests = n_combinations * (n_combinations - 1) / 2
    corrected_alpha = 0.05 / n_tests
    
    for (loc1, treat1, part1), (loc2, treat2, part2) in itertools.combinations(combinations, 2):
        group1 = df[(df['Location'] == loc1) & (df['Treatment'] == treat1) & (df['Partner'] == part1)]['decision']
        group2 = df[(df['Location'] == loc2) & (df['Treatment'] == treat2) & (df['Partner'] == part2)]['decision']
        
        ks_stat, p_value = ks_2samp(group1, group2)
        
        # Apply Bonferroni correction
        significant = 'Yes' if p_value < corrected_alpha else 'No'
        
        results.append({
            'Group 1': f'{loc1}, {treat1}, {part1}',
            'Group 2': f'{loc2}, {treat2}, {part2}',
            'KS Statistic': ks_stat,
            'P-value': p_value,
            'Significant After Correction': significant
        })
        
    return pd.DataFrame(results)


def uniform_cdf(x):
    return np.where(x < 1, 0, np.where(x > 5, 1, (x - 1) / 4))


def one_sample_ks(data: pd.DataFrame, treatment: str):
    treatment_df = data[data['Treatment'] == treatment]

    results = []
    
    for group, group_df in treatment_df.groupby(['Location', 'Partner']):
        # group_df['decision'] contains the decisions for this subgroup
        statistic, p_value = kstest(group_df['decision'], uniform_cdf)
        results.append({'Group': group, 'KS Statistic': statistic, 'P-value': p_value})

    # Display the results
    for result in results:
        print(result)


def gen_bm(data: pd.DataFrame, 
                       loc1: str, game1: str, partner1: str, 
                       loc2: str, game2: str, partner2: str):
    sample1 = data[(data['Location'] == loc1) & (data['Treatment'] == game1) & (data['Partner'] == partner1)]['decision']
    sample2 = data[(data['Location'] == loc2) & (data['Treatment'] == game2) & (data['Partner'] == partner2)]['decision']

    statistic, p_value = brunnermunzel(sample1, sample2)
    print(f"Brunner-Munzel Test Statistic: {statistic}, P-value: {p_value}")