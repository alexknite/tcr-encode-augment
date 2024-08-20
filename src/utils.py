"""
Title: Utils
Author: Alex Nite (me@alexnite.com) & Sarah Huang (shuan146@ucsc.edu)
Date: July 2, 2024
Description:
    Establish useful methods for this project
"""

### Imports ###
import pandas as pd
import numpy as np
from Bio import SeqIO
from Levenshtein import distance
from mhc_pseudo import mhc_pseudo

def parse_fasta(fasta_file):
    '''
    Parse FASTA files and extract V allele names and their full amino acid
    sequences
    
    parameters:
        - fasta_file (str) : FASTA file location
    returns: 
        - (dictionary) V allele names and full amino acid sequences
    '''
    seq_dict = {}
    for record in SeqIO.parse(fasta_file, 'fasta'):
        desc = record.description
        allele_name = desc.split('|')[1]
        seq = str(record.seq)
        seq_dict[allele_name] = seq
    return seq_dict

def extract_cdr_sequences(v_dict, cdr1_seqs, cdr2_seqs):
    '''
    Iterate over TRAV/TRBV dictionaries and create a dictionary containing
    V allele names and CDR1,2 sequences

    parameters: 
        - v_dict (dictionary) :  contains V allele names and full sequences
        - cdr1_sequences (list) : contains CDR1 sequences from NetTCR2.2
        - cdr2_sequences (list) : contains CDR2 sequences from NetTCR2.2
    returns: 
        - (dictionary) containing V allele names and mapped CDR1,2 
          sequences from NetTCR2.2's dataset
    '''
    cdr_dict = {}

    for v_name, full_seq in v_dict.items():
        cdr1_found = False
        cdr2_found = False
        cdr1 = ''
        cdr2 = ''

        for a1_seq in cdr1_seqs:
            if a1_seq in full_seq:
                cdr1 = a1_seq
                cdr1_found = True
                break

        for a2_seq in cdr2_seqs:
            if a2_seq in full_seq:
                cdr2 = a2_seq
                cdr2_found = True
                break

        if cdr1_found or cdr2_found:
            cdr_dict[v_name] = {
                'cdr1': cdr1,
                'cdr2': cdr2
            }

    return cdr_dict

def find_alt_mhc(mhc, max_allele=100):
    '''
    Attempt to find alternate allele variants 

    parameters:
        - mhc (str) : MHC allele name
        - max_alelle (int) : Max allele variant to search for
    returns:
        - (str) Either modified or original MHC alelle name
    '''
    if ':' not in mhc:
        for i in range(1, max_allele + 1):
            new_mhc = f"{mhc}:{i if i >= 10 else f'0{i}'}"
            if new_mhc in mhc_pseudo:
                return new_mhc
        return mhc
    
    if mhc not in mhc_pseudo:
        base = mhc.split(':')[0]
        for i in range(1, max_allele + 1):
            new_mhc = f"{base}:{i if i >= 10 else f'0{i}'}"
            if new_mhc in mhc_pseudo:
                return new_mhc
    return mhc

def clean_mhcs(df):
    '''
    Remove MHCs that are MHC class II, modify MHCs allele variants, and
    remove remaining MHCs that are not present in reference dictionary 

    parameters: 
        - df (DataFrame) : Original VDJdb dataset
    return: 
        - (DataFrame) filtered VDJdb dataset
    '''
    result_df = df.copy()
    
    # Remove MHCs with 'D'
    result_df = result_df[~result_df['MHC A'].str.contains('D', na=False)]

    # Locate MHCs with >= 2 colons
    has_multiple_colons = result_df['MHC A'].str.count(':') >= 2

    # Remove second colon and anything after
    result_df.loc[has_multiple_colons, 'MHC A'] = result_df.loc[
        has_multiple_colons, 'MHC A'].apply(
        lambda x: ':'.join(x.split(':')[:2])
    )

    # Locate MHCs with no colons
    no_colons = ~result_df['MHC A'].str.contains(':')

    # Append x- allele to HLAs without colons
    result_df.loc[no_colons, 'MHC A'] = result_df.loc[
        no_colons, 'MHC A'].apply(find_alt_mhc)

    result_df.loc[~result_df['MHC A'].isin(mhc_pseudo.keys()), 'MHC A'] = result_df.loc[~result_df['MHC A'].isin(mhc_pseudo.keys()), 'MHC A'].apply(find_alt_mhc)

    result_df = result_df[result_df['MHC A'].isin(mhc_pseudo.keys())]
    
    return result_df

def get_novel_targets(validation_df, training_df, max_distance=2):
    validation_epitopes = validation_df['Epitope'].to_numpy()
    training_epitopes = training_df['Epitope'].to_numpy()

    novel_mask = np.ones(len(validation_epitopes), dtype=bool)

    for i, validation_epitope in enumerate(validation_epitopes):
        distances = np.array([distance(validation_epitope, training_epitope) \
                               for training_epitope in training_epitopes])
        
        if np.any(distances < max_distance):
            novel_mask[i] = False

    novel_targets = validation_df[novel_mask].reset_index(drop=True)

    return novel_targets

def get_cdr1_cdr2(v_name, cdr_a_dict, cdr_b_dict):
    '''
    Search alpha and beta CDR dictionaries for V allele names, and extract
    CDR1,2 sequences

    parameters:
        - v_name (str) : V allele name
        - cdr_a_dict (dictionary) : containing of TRAV names and CDR1,2 sequences
        - cdr_b_dict (dictionary) : containing TRBV names and CDR1,2 sequences
    returns:
        - (tuple) of CDR1,2 sequences for a specific V allele name
    '''
    if v_name in cdr_a_dict: 
        return cdr_a_dict[v_name]['cdr1'], cdr_a_dict[v_name]['cdr2']
    elif v_name in cdr_b_dict:
        return cdr_b_dict[v_name]['cdr1'], cdr_b_dict[v_name]['cdr2']
    else:
        return None, None 
    
def get_pmhc_sequence(mhc_name):
    '''
    Search dictionary containing MHC allele names and their pseudo sequences

    parameters:
        - mhc_name (dictionary ): containing MHC allele names and pseudo sequences
    returns: 
        - (str) If found, MHC pseudo sequence for specific MHC allele, or None
    '''
    return mhc_pseudo.get(mhc_name, None)

def generate_negatives(positive_df, n_copies=3, n_shuffles=3):
    '''
    From positive training dataset, generate negative examples by shuffling
    the pMHC column n_shuffle-times

    parameters:
        - positive_df (DataFrame) : containing positive training data
        - n_shuffles (int) : number of shuffling to complete (default: 3)
    returns: 
        - (DataFrame) containing negative training data
    '''
    positive_copies = [positive_df.copy() for _ in range(n_copies)]

    negative_df = pd.concat(positive_copies, ignore_index=True)

    shuffled_pmhc = None
    
    for _ in range(n_shuffles):
        shuffled_pmhc = np.random.permutation(negative_df['pmhc_encoded'])

    negative_df['pmhc_encoded'] = shuffled_pmhc

    common_rows = pd.merge(negative_df, positive_df, how='inner')
    print(f'Number of common rows found: {common_rows.shape[0]}')

    negative_df = negative_df.merge(
        positive_df, 
        how='left', 
        indicator=True).query('_merge == "left_only"').drop(columns='_merge')
    
    negative_df['pos_neg'] = 0

    return negative_df

def generate_variations(row, pos_neg):
    '''
    From current positive OR negative DataFrame, generate more positive OR negative variations by
    creating masked alpha chains, beta chains, V genes, HLA alleles, and 
    epitope sequences for each positive or negative example

    parameters: 
        - row : a single row in training DataFrame
        - pos_neg (int) : 0 (negative) or 1 (positive)
    returns: 
        - list of positive OR negative variations for a single row
    '''
    variations = []
    V_components = row['tcr_encoded'].split('-')
    target_components = row['pmhc_encoded'].split('-')

    variations.append({
        'V_alpha': row['V_alpha'],
        'V_beta': row['V_beta'],
        'tcr_encoded': 'x-x-x-' + '-'.join(V_components[3:]),
        'pmhc_encoded': row['pmhc_encoded'],
        'pos_neg': pos_neg
        })
    variations.append({
        'V_alpha': row['V_alpha'],
        'V_beta': row['V_beta'],
        'tcr_encoded': '-'.join(V_components[3:]) + '-x-x-x',
        'pmhc_encoded': row['pmhc_encoded'],
        'pos_neg': pos_neg
        })
    variations.append({
        'V_alpha': row['V_alpha'],
        'V_beta': row['V_beta'],
        'tcr_encoded': 'x-x-' + V_components[2] + '-x-x-' + V_components[5],
        'pmhc_encoded': row['pmhc_encoded'],
        'pos_neg': pos_neg
        })

    variations.append({
        'V_alpha': row['V_alpha'],
        'V_beta': row['V_beta'],
        'tcr_encoded': row['tcr_encoded'],
        'pmhc_encoded': 'x-' + target_components[1],
        'pos_neg': pos_neg
        })
    variations.append({
        'V_alpha': row['V_alpha'],
        'V_beta': row['V_beta'],
        'tcr_encoded': row['tcr_encoded'],
        'pmhc_encoded': target_components[0] + '-x',
        'pos_neg': pos_neg
        })
    
    return variations

