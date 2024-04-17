"""
Code to pair participants in dictator and receiver pairs according to
their partner's location specifier (Chapman or Wuhan)
"""

#=============================================================
# Imports
#=============================================================

import pandas as pd

#=============================================================
# Matching Functionality 
#=============================================================

# ALL CHAPMAN PARTICIPANTS (PILOT ALPHA)

def match_subjects_all_chapman(data_path: str, file_name: str):
    """
    Matches subjects when all participants are from Chapman

    :param data_path: the path of the data set
    :return: the data set to pay subjects based on their pairings
    """

    df = pd.read_csv(data_path)
    df.rename(columns={"decision": "Decision"}, inplace=True)
    matched_df = pd.DataFrame(columns=['recruiter_id', 'Role', 'Treatment', 'Location', 'Partner', 'Decision', 'Player Payoff', 'Partner Payoff', 'Matched Player ID'])
    
    for treatment in ['Lie', 'Dict']:
        df_treatment = df[df['Treatment'] == treatment]
        
        dictators = df_treatment[df_treatment['Role'] == 'Dictator']
        receivers = df_treatment[df_treatment['Role'] == 'Receiver']
        
        print(f"Number of Dictators in {treatment}: {len(dictators)}")
        print(f"Number of Receivers in {treatment}: {len(receivers)}")
        
        for index, dictator in dictators.iterrows():
            match = receivers[receivers['Location'] == dictator['Partner']]
            print(f"Matching Dictator: {dictator['recruiter_id']}, Partner: {dictator['Partner']}, Location: {dictator['Location']}")
            print(f"Potential Matches: {match['recruiter_id'].tolist()}")
            if not match.empty:
                receiver = match.iloc[0]
                print(f"Matched Receiver: {receiver['recruiter_id']}")
                
                dictator_payoff = 6 - dictator['Decision']
                receiver_payoff = dictator['Decision']
                
                matched_df = matched_df.append({
                    'recruiter_id': dictator['recruiter_id'],
                    'Role': 'Dictator',
                    'Treatment': treatment,
                    'Location': dictator['Location'],
                    'Partner': dictator['Partner'],
                    'Decision': dictator['Decision'],
                    'Player Payoff': dictator_payoff,
                    'Partner Payoff': receiver_payoff,
                    'Matched Player ID': receiver['recruiter_id']
                }, ignore_index=True)
                
                matched_df = matched_df.append({
                    'recruiter_id': receiver['recruiter_id'],
                    'Role': 'Receiver',
                    'Treatment': treatment,
                    'Location': receiver['Location'],
                    'Partner': receiver['Partner'],
                    'Decision': receiver['Decision'],
                    'Player Payoff': receiver_payoff,
                    'Partner Payoff': '',
                    'Matched Player ID': dictator['recruiter_id']
                }, ignore_index=True)
                
                receivers = receivers.drop(receiver.name)
            else:
                print("No suitable match found for Dictator. Matching to redundant receiver.")
                # Match dictator to a redundant receiver (one that has already been matched)
                matched_receivers = matched_df[matched_df['Role'] == 'Receiver']
                if not matched_receivers.empty:
                    receiver = matched_receivers.iloc[0]  # Select the first redundant receiver
                    receiver_payoff = 0  # Dictator's decision doesn't affect this receiver's payoff
                    matched_df = matched_df.append({
                        'recruiter_id': dictator['recruiter_id'],
                        'Role': 'Dictator',
                        'Treatment': treatment,
                        'Location': dictator['Location'],
                        'Partner': dictator['Partner'],
                        'Decision': dictator['Decision'],
                        'Player Payoff': dictator_payoff,
                        'Partner Payoff': receiver_payoff,
                        'Matched Player ID': receiver['Matched Player ID']
                    }, ignore_index=True)
                else:
                    print("No redundant receiver found. Dictator cannot be matched.")
        
        for index, receiver in receivers.iterrows():
            match = dictators[dictators['Partner'] == receiver['Location']]
            print(f"Matching Receiver: {receiver['recruiter_id']}, Partner: {receiver['Partner']}, Location: {receiver['Location']}")
            print(f"Potential Matches: {match['recruiter_id'].tolist()}")
            if not match.empty:
                dictator = match.iloc[0]
                print(f"Matched Dictator: {dictator['recruiter_id']}")
                
                dictator_payoff = 6 - dictator['Decision']
                receiver_payoff = dictator['Decision']
                
                matched_df = matched_df.append({
                    'recruiter_id': dictator['recruiter_id'],
                    'Role': 'Dictator',
                    'Treatment': treatment,
                    'Location': dictator['Location'],
                    'Partner': dictator['Partner'],
                    'Decision': dictator['Decision'],
                    'Player Payoff': dictator_payoff,
                    'Partner Payoff': receiver_payoff,
                    'Matched Player ID': receiver['recruiter_id']
                }, ignore_index=True)
                
                matched_df = matched_df.append({
                    'recruiter_id': receiver['recruiter_id'],
                    'Role': 'Receiver',
                    'Treatment': treatment,
                    'Location': receiver['Location'],
                    'Partner': receiver['Partner'],
                    'Decision': receiver['Decision'],
                    'Player Payoff': receiver_payoff,
                    'Partner Payoff': '',
                    'Matched Player ID': dictator['recruiter_id']
                }, ignore_index=True)
                
                dictators = dictators.drop(dictator.name)
            else:
                print("No suitable match found for Receiver. Matching to redundant dictator.")
                # Match receiver to a redundant dictator (one that has already been matched)
                matched_dictators = matched_df[matched_df['Role'] == 'Dictator']
                if not matched_dictators.empty:
                    dictator = matched_dictators.iloc[0]  # Select the first redundant dictator
                    dictator_payoff = 6 - dictator['Decision']  # Receiver's decision doesn't affect this dictator's payoff
                    matched_df = matched_df.append({
                        'recruiter_id': receiver['recruiter_id'],
                        'Role': 'Receiver',
                        'Treatment': treatment,
                        'Location': receiver['Location'],
                        'Partner': receiver['Partner'],
                        'Decision': receiver['Decision'],
                        'Player Payoff': receiver_payoff,
                        'Partner Payoff': dictator_payoff,
                        'Matched Player ID': dictator['recruiter_id']
                    }, ignore_index=True)
                else:
                    print("No redundant dictator found. Receiver cannot be matched.")
    
    matched_df['Player Payoff in USD'] = matched_df['Player Payoff']*3
    matched_df.to_csv(file_name)
    return matched_df

# Chapman X Wuhan Participants 

def ensure_every_dictator_and_receiver_matches(df):
    df['Player Payoff'] = None
    df['Partner Payoff'] = None
    df['Matched ID'] = None
    
    # Track which dictators and receivers have been matched
    matched_dictators = set()
    matched_receivers = set()

    # Step 1: Guarantee a match for each dictator
    for index, dictator in df[df['Role'] == 'Dictator'].iterrows():
        potential_matches = df[(df['Treatment'] == dictator['Treatment']) & 
                               (df['Role'] == 'Receiver') & 
                               (df['Location'] == dictator['Partner']) & 
                               (~df['recruiter_id'].isin(matched_receivers))]

        # Allow matching with already matched Receivers if no unmatched ones are available
        if potential_matches.empty:
            potential_matches = df[(df['Treatment'] == dictator['Treatment']) & 
                                   (df['Role'] == 'Receiver') & 
                                   (df['Location'] == dictator['Partner'])]

        if not potential_matches.empty:
            receiver = potential_matches.iloc[0]
            matched_dictators.add(dictator['recruiter_id'])
            matched_receivers.add(receiver['recruiter_id'])

            # Update match and payoff details
            df.at[index, 'Player Payoff'] = 6 - dictator['decision']
            df.at[index, 'Partner Payoff'] = dictator['decision']
            df.at[index, 'Matched ID'] = receiver['recruiter_id']
            
            receiver_index = receiver.name
            df.at[receiver_index, 'Player Payoff'] = dictator['decision']
            df.at[receiver_index, 'Matched ID'] = dictator['recruiter_id']
    
    # Step 2: Ensure every receiver is matched, even if redundantly
    for index, receiver in df[df['Role'] == 'Receiver'].iterrows():
        if receiver['recruiter_id'] not in matched_receivers:
            # Look for any dictator that matches the treatment and location criteria
            potential_dictators = df[(df['Treatment'] == receiver['Treatment']) & 
                                     (df['Role'] == 'Dictator') & 
                                     (df['Partner'] == receiver['Location'])]

            if not potential_dictators.empty:
                dictator = potential_dictators.iloc[0]
                # Update only receiver's information for redundant matches
                df.at[index, 'Player Payoff'] = dictator['decision']
                df.at[index, 'Matched ID'] = dictator['recruiter_id']

    return df

