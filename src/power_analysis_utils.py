"""
Code to conduct the power analysis on the Kolmogorov-Smirnov and Chi-Squared tests
"""

#=============================================================
# Imports
#=============================================================

import numpy as np
import scipy.stats as stats

#=============================================================
# Kolmogorov-Smirnov
#=============================================================


def ks_power_analysis_custom_distributions(alpha=0.05, power=0.8, n_sim=1000):
    """
    Estimates the sample size needed to achieve a specified power for custom distributions in a K-S test.

    Parameters:
    - alpha: Significance level
    - power: Desired power of the test
    - n_sim: Number of simulations to perform

    Returns:
    - Estimated sample size needed
    """
    sample_sizes = np.arange(10, 501, 10)  # Sample sizes to explore

    # Distributions specifications
    null_distribution = np.array([1, 2, 3, 4, 5])  # Uniform distribution over 1, 2, 3, 4, 5
    alt_distribution_probs = np.array([1, 2, 3, 2, 1]) / 9  # Inverted V distribution probabilities

    for n in sample_sizes:
        n_rejections = 0
        for _ in range(n_sim):
            # Generate samples from the specified distributions
            null_sample = np.random.choice(null_distribution, size=n, replace=True)
            alt_sample = np.random.choice(null_distribution, size=n, replace=True, p=alt_distribution_probs)
            
            # Perform the K-S test
            _, p_value = stats.ks_2samp(null_sample, alt_sample)
            
            # Check if the null hypothesis is rejected
            if p_value < alpha:
                n_rejections += 1
        
        # Calculate the power for this sample size
        current_power = n_rejections / n_sim
        
        # If the power is at least the desired power, return the current sample size
        if current_power >= power:
            return n, current_power

    return None, None  # If no sample size was sufficient, return None


# Modify the function to record the power achieved at each sample size
def ks_power_analysis_custom_distributions_with_recording(alpha=0.05, power=0.8, n_sim=1000):
    sample_sizes = np.arange(10, 501, 10)  # Sample sizes to explore
    powers = []  # List to store the power achieved at each sample size

    # Distributions specifications
    null_distribution = np.array([1, 2, 3, 4, 5])
    alt_distribution_probs = np.array([1, 2, 3, 2, 1]) / 9

    for n in sample_sizes:
        n_rejections = 0
        for _ in range(n_sim):
            null_sample = np.random.choice(null_distribution, size=n, replace=True)
            alt_sample = np.random.choice(null_distribution, size=n, replace=True, p=alt_distribution_probs)
            _, p_value = stats.ks_2samp(null_sample, alt_sample)
            if p_value < alpha:
                n_rejections += 1
        current_power = n_rejections / n_sim
        powers.append(current_power)

    return sample_sizes, powers


# Adjusting the analysis to include more extreme cases for the steepness and height of the inverted V distribution

def ks_power_analysis_extreme_cases(alpha=0.05, power=0.8, n_sim=500, steepness_factors=[1, 2, 4, 8, 16]):
    results = []  # List to store results for different steepness factors
    sample_sizes = np.arange(10, 301, 10)  # Range of sample sizes

    null_distribution = np.array([1, 2, 3, 4, 5])

    for factor in steepness_factors:
        powers = []
        # Adjust the probabilities for the inverted V distribution based on the steepness factor
        alt_distribution_probs = np.array([1, 2, 3, 2, 1]) * factor
        alt_distribution_probs = alt_distribution_probs / alt_distribution_probs.sum()

        for n in sample_sizes:
            n_rejections = 0
            for _ in range(n_sim):
                null_sample = np.random.choice(null_distribution, size=n, replace=True)
                alt_sample = np.random.choice(null_distribution, size=n, replace=True, p=alt_distribution_probs)
                _, p_value = stats.ks_2samp(null_sample, alt_sample)
                if p_value < alpha:
                    n_rejections += 1
            current_power = n_rejections / n_sim
            powers.append(current_power)
        
        results.append((factor, sample_sizes, powers))
    
    return results


# Adjusting the analysis for an alternative hypothesis with a more extreme distribution of mass
# The alternative distribution will place a large amount of mass at 1, decreasing towards 5.

def ks_power_analysis_extreme_prior(alpha=0.05, power=0.8, n_sim=500, mass_factors=[1, 2, 4, 8, 16]):
    results = []  # List to store results for different mass distributions
    sample_sizes = np.arange(10, 301, 10)  # Range of sample sizes

    null_distribution = np.array([1, 2, 3, 4, 5])

    for factor in mass_factors:
        powers = []
        # Create an extreme distribution for the alternative hypothesis
        alt_distribution_probs = np.array([4, 3, 2, 1, 0.5]) * factor
        alt_distribution_probs = alt_distribution_probs / alt_distribution_probs.sum()

        for n in sample_sizes:
            n_rejections = 0
            for _ in range(n_sim):
                null_sample = np.random.choice(null_distribution, size=n, replace=True)
                alt_sample = np.random.choice(null_distribution, size=n, replace=True, p=alt_distribution_probs)
                _, p_value = stats.ks_2samp(null_sample, alt_sample)
                if p_value < alpha:
                    n_rejections += 1
            current_power = n_rejections / n_sim
            powers.append(current_power)
        
        results.append((factor, sample_sizes, powers))
    
    return results


# Adjusting the analysis for a bi-modal alternative hypothesis
# This alternative distribution will place a large amount of mass at 1 and 5, less on 2 and 4, and the least on 3.

def ks_power_analysis_bimodal(alpha=0.05, power=0.8, n_sim=500, mass_factors=[1, 2, 4, 8, 16]):
    results = []  # List to store results for different bi-modal distributions
    sample_sizes = np.arange(10, 301, 10)  # Range of sample sizes

    null_distribution = np.array([1, 2, 3, 4, 5])

    for factor in mass_factors:
        powers = []
        # Create a bi-modal distribution for the alternative hypothesis
        alt_distribution_probs = np.array([4, 2, 1, 2, 4]) * factor
        alt_distribution_probs = alt_distribution_probs / alt_distribution_probs.sum()

        for n in sample_sizes:
            n_rejections = 0
            for _ in range(n_sim):
                null_sample = np.random.choice(null_distribution, size=n, replace=True)
                alt_sample = np.random.choice(null_distribution, size=n, replace=True, p=alt_distribution_probs)
                _, p_value = stats.ks_2samp(null_sample, alt_sample)
                if p_value < alpha:
                    n_rejections += 1
            current_power = n_rejections / n_sim
            powers.append(current_power)
        
        results.append((factor, sample_sizes, powers))
    
    return results


#=============================================================
# Chi-Squared (One-Sample)
#=============================================================

def chi_squared_power_analysis_extreme_cases(alpha=0.05, power=0.8, n_sim=500, steepness_factors=[1, 2, 4, 8, 16]):
    results = []  # List to store results for different steepness factors
    sample_sizes = np.arange(10, 301, 10)  # Range of sample sizes
    
    null_distribution = np.array([1, 2, 3, 4, 5])
    null_probs = np.ones_like(null_distribution) / len(null_distribution)  # Equal probabilities for null distribution

    for factor in steepness_factors:
        powers = []
        # Adjust the probabilities for the inverted V distribution based on the steepness factor
        alt_distribution_probs = np.array([1, 2, 3, 2, 1], dtype=float) * factor
        alt_distribution_probs /= alt_distribution_probs.sum()

        for n in sample_sizes:
            n_rejections = 0
            for _ in range(n_sim):
                null_sample = np.random.choice(null_distribution, size=n, replace=True, p=null_probs)
                alt_sample = np.random.choice(null_distribution, size=n, replace=True, p=alt_distribution_probs)
                
                # Calculate observed frequencies for the null and alternative samples
                null_freqs = np.array([np.sum(null_sample == category) for category in null_distribution])
                alt_freqs = np.array([np.sum(alt_sample == category) for category in null_distribution])

                # Combine frequencies to get observed and expected frequencies
                observed = alt_freqs
                expected = null_freqs
                
                # Chi-squared test
                chi2_stat, p_value = stats.chisquare(f_obs=observed, f_exp=expected)
                
                if p_value < alpha:
                    n_rejections += 1
            
            current_power = n_rejections / n_sim
            powers.append(current_power)
        
        results.append((factor, sample_sizes, powers))
    
    return results


def chi_squared_power_analysis_extreme_prior(alpha=0.05, power=0.8, n_sim=500, mass_factors=[1, 2, 4, 8, 16]):
    results = []  # List to store results for different mass distributions
    sample_sizes = np.arange(10, 301, 10)  # Range of sample sizes
    
    null_distribution = np.array([1, 2, 3, 4, 5])
    null_probs = np.ones_like(null_distribution) / len(null_distribution)  # Equal probabilities for null distribution

    for factor in mass_factors:
        powers = []
        # Create an extreme distribution for the alternative hypothesis
        alt_distribution_probs = np.array([4, 3, 2, 1, 0.5]) * factor
        alt_distribution_probs /= alt_distribution_probs.sum()

        for n in sample_sizes:
            n_rejections = 0
            for _ in range(n_sim):
                null_sample = np.random.choice(null_distribution, size=n, replace=True, p=null_probs)
                alt_sample = np.random.choice(null_distribution, size=n, replace=True, p=alt_distribution_probs)
                
                # Calculate observed frequencies for the null and alternative samples
                null_freqs = np.array([np.sum(null_sample == category) for category in null_distribution])
                alt_freqs = np.array([np.sum(alt_sample == category) for category in null_distribution])

                # Combine frequencies to get observed and expected frequencies
                observed = alt_freqs
                expected = null_freqs
                
                # Chi-squared test
                chi2_stat, p_value = stats.chisquare(f_obs=observed, f_exp=expected)
                
                if p_value < alpha:
                    n_rejections += 1
            
            current_power = n_rejections / n_sim
            powers.append(current_power)
        
        results.append((factor, sample_sizes, powers))
    
    return results


def chi_squared_power_analysis_bimodal(alpha=0.05, power=0.8, n_sim=500, mass_factors=[1, 2, 4, 8, 16]):
    results = []  # List to store results for different bi-modal distributions
    sample_sizes = np.arange(10, 301, 10)  # Range of sample sizes
    
    null_distribution = np.array([1, 2, 3, 4, 5])
    null_probs = np.ones_like(null_distribution) / len(null_distribution)  # Equal probabilities for null distribution

    for factor in mass_factors:
        powers = []
        # Create a bi-modal distribution for the alternative hypothesis
        alt_distribution_probs = np.array([4, 2, 1, 2, 4], dtype=float) * factor
        alt_distribution_probs /= alt_distribution_probs.sum()

        for n in sample_sizes:
            n_rejections = 0
            for _ in range(n_sim):
                null_sample = np.random.choice(null_distribution, size=n, replace=True, p=null_probs)
                alt_sample = np.random.choice(null_distribution, size=n, replace=True, p=alt_distribution_probs)
                
                # Calculate observed frequencies for the null and alternative samples
                null_freqs = np.array([np.sum(null_sample == category) for category in null_distribution])
                alt_freqs = np.array([np.sum(alt_sample == category) for category in null_distribution])

                # Combine frequencies to get observed and expected frequencies
                observed = alt_freqs
                expected = null_freqs
                
                # Chi-squared test
                chi2_stat, p_value = stats.chisquare(f_obs=observed, f_exp=expected)
                
                if p_value < alpha:
                    n_rejections += 1
            
            current_power = n_rejections / n_sim
            powers.append(current_power)
        
        results.append((factor, sample_sizes, powers))
    
    return results
