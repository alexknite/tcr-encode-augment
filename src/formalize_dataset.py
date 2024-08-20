"""
Title: Formalize Dataset
Author: Alex Nite (me@alexnite.com) & Sarah Huang (shuan146@ucsc.edu)
Date: July 2, 2024
Description:
    (1) Collect TCR-antigen paired sequences from VDJdb, (2) pair CDR1,2
    sequences to TCRs and pseudo HLA sequences to MHC allele names from
    proprietary datasets, (3) encode TCR and target sequences, (4) perform
    data augmentation, and finally (5) export validation, full training, and
    limited training datasets to be inputted into machine learning model
    to predict TCR-antigen interactivity.
"""

### Imports ###
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'data'))

import pandas as pd
import utils

# File paths
nettcr_file = './data/nettcr_2_2_full_dataset.csv'
tratcr_file = './data/proprietary/TRAV_aa.fasta'
trbtcr_file = './data/proprietary/TRBV_aa.fasta'
vdjdb_file = './data/VDJdb_Human_TRA_TRB_append.tsv'
missing_hla_output_file = './out/missing_hla_set.tsv'
validation_output_file = './out/validation_dataset.tsv'
full_training_output_file = './out/full_training_dataset.tsv'
limited_training_output_file = './out/limited_training_dataset.tsv'

# Fraction to split validation dataset
VALIDATION_FRACTION = 0.1

### Load 'data' files ###
nettcr_df = pd.read_csv(nettcr_file)
print(f'[NetTCR2.2] {nettcr_df.shape[0]} entries')

original_vdjdb_df = pd.read_csv(vdjdb_file, sep='\t')
print(f'[VDJdb] {original_vdjdb_df.shape[0]} entries')

# nettcr_unique_targets = len(nettcr_df['peptide'].unique())
# print(f'[NetTCR2.2] {nettcr_unique_targets} unique targets')
# vdjdb_unique_targets = len(original_vdjdb_df['Epitope'].unique())
# print(f'[VDJdb] {vdjdb_unique_targets} unique targets')

tratcr_dict = utils.parse_fasta(tratcr_file)
print(f'[References] {len(tratcr_dict)} TRAV entries')
trbtcr_dict = utils.parse_fasta(trbtcr_file)
print(f'[References] {len(trbtcr_dict)} TRBV entries')

# Curate alpha and beta CDR dictionaries
cdr_dict_alpha = utils.extract_cdr_sequences(tratcr_dict, nettcr_df['A1'],
                                        nettcr_df['A2'])
print(f'[References] {len(cdr_dict_alpha)} TRAV-CDR1,2 matches')
cdr_dict_beta = utils.extract_cdr_sequences(trbtcr_dict, nettcr_df['B1'], 
                                      nettcr_df['B2'])
print(f'[References] {len(cdr_dict_beta)} TRBV-CDR1,2 matches')

### Data Preparation ###

# Remove unwanted MHCs 
original_vdjdb_df = utils.clean_mhcs(original_vdjdb_df)
print(f'[VDJdb] {original_vdjdb_df.shape[0]} entries after cleaning MHCs')

# Split into alpha and beta DataFrames
alpha_df = original_vdjdb_df[original_vdjdb_df['Gene'] == 'TRA']
print(f'[VDJdb] {alpha_df.shape[0]} alpha genes')
beta_df = original_vdjdb_df[original_vdjdb_df['Gene'] == 'TRB']
print(f'[VDJdb] {beta_df.shape[0]} beta genes')

# Split into paired and unpaired alpha and beta DataFrames
paired_alpha_df = alpha_df[alpha_df['complex.id'] != 0]
print(f'[VDJdb] {paired_alpha_df.shape[0]} paired alpha genes')
paired_beta_df = beta_df[beta_df['complex.id'] != 0]
print(f'[VDJdb] {paired_beta_df.shape[0]} paired beta genes')
unpaired_alpha_df = alpha_df[alpha_df['complex.id'] == 0]
print(f'[VDJdb] {unpaired_alpha_df.shape[0]} unpaired alpha genes')
unpaired_beta_df = beta_df[beta_df['complex.id'] == 0]
print(f'[VDJdb] {unpaired_beta_df.shape[0]} unpaired beta genes')

# Rename columns in unpaired alpha and beta DataFrames for proper concatenation
unpaired_alpha_df = unpaired_alpha_df.rename(
    columns={
        'V': 'V_alpha',
        'CDR3': 'CDR3_alpha'
    }
)
unpaired_beta_df = unpaired_beta_df.rename(
    columns ={
        'V': 'V_beta',
        'CDR3': 'CDR3_beta'
    }
)

# Set alpha/beta columns to None for consistency
unpaired_alpha_df[['V_beta', 'CDR3_beta']] = None
unpaired_beta_df[['V_alpha', 'CDR3_alpha']] = None 

# Merge paired alpha and beta chain DataFrames because information is already
# complete
merged_paired_df = pd.merge(
    paired_alpha_df,
    paired_beta_df,
    how='outer',
    on=['complex.id', 'MHC A','Epitope', 'Score'],
    suffixes=['_alpha', '_beta'])

# Randomly split merged paired DataFrame into validation (10%) and 
# training (90%) DataFrames
paired_validation_df = merged_paired_df.sample(frac=VALIDATION_FRACTION)
print(f'[Validation] {paired_validation_df.shape[0]} paired entries')
paired_training_df = merged_paired_df.drop(paired_validation_df.index)
print(f'[Training] {paired_training_df.shape[0]} paired entries')

# Separately split unpaired DataFrames using the same fraction to ensure an 
# equal representation of paired and unpaired entries in our validation and
# training DataFrames
unpaired_alpha_validation_df = unpaired_alpha_df.sample(frac=VALIDATION_FRACTION)
print(f'[Validation] {unpaired_alpha_validation_df.shape[0]} unpaired alpha entries')
unpaired_beta_validation_df = unpaired_beta_df.sample(frac=VALIDATION_FRACTION)
print(f'[Validation] {unpaired_beta_validation_df.shape[0]} unpaired beta entries')
unpaired_alpha_training_df = unpaired_alpha_df.drop(unpaired_alpha_validation_df.index)
print(f'[Training] {unpaired_alpha_training_df.shape[0]} unpaired alpha entries')
unpaired_beta_training_df = unpaired_beta_df.drop(unpaired_beta_validation_df.index)
print(f'[Training] {unpaired_beta_training_df.shape[0]} unpaired beta entries')

# Concatenate all validation DataFrames
validation_df = pd.concat(
    [
        paired_validation_df,
        unpaired_alpha_validation_df,
        unpaired_beta_validation_df
    ],
    ignore_index=True
)
print(f'[Validation] {validation_df.shape[0]} complete entries')

# Concatenate all training datasets
training_df = pd.concat(
    [
        paired_training_df,
        unpaired_alpha_training_df,
        unpaired_beta_training_df
    ],
    ignore_index=True
)
print(f'[Training] {training_df.shape[0]} complete entries')

################################################################################
# Filter validation DataFrame to only contain of novel targets 

# validation_epitopes = validation_df['Epitope']
# training_epitopes = training_df['Epitope']
# novel_validation_df = validation_df.apply(
#     lambda validation_epitope: utils.is_novel(validation_epitope, training_df))
# validation_df = validation_df[novel_validation_df]

# validation_df = utils.get_novel_targets(validation_df, training_df)
# print(f'[Validation] {validation_df.shape[0]} novel target entries')
################################################################################

# Filter novel targets with a Score = 0
validation_df = validation_df[validation_df['Score'] != 0]
# print(f'[Validation] {validation_df.shape[0]} novel target, high confidence entries')
print(f'[Validation] {validation_df.shape[0]} high confidence entries')

# Load CDR1,2 sequences into validation and training DataFrames
validation_df[['CDR1_alpha', 'CDR2_alpha']] = \
    validation_df['V_alpha'].apply(
        lambda v: pd.Series(utils.get_cdr1_cdr2(v, cdr_dict_alpha, cdr_dict_beta)))
validation_df[['CDR1_beta', 'CDR2_beta']] = \
    validation_df['V_beta'].apply(
        lambda v: pd.Series(utils.get_cdr1_cdr2(v, cdr_dict_alpha, cdr_dict_beta)))
training_df[['CDR1_alpha', 'CDR2_alpha']] = \
    training_df['V_alpha'].apply(
        lambda v: pd.Series(utils.get_cdr1_cdr2(v, cdr_dict_alpha, cdr_dict_beta)))
training_df[['CDR1_beta', 'CDR2_beta']] = \
    training_df['V_beta'].apply(
        lambda v: pd.Series(utils.get_cdr1_cdr2(v, cdr_dict_alpha, cdr_dict_beta)))

# Load pseudo sequences into validation and training DataFrames
validation_df['pseudo_sequence'] = validation_df[
    'MHC A'].apply(utils.get_pmhc_sequence)
training_df['pseudo_sequence'] = training_df[
    'MHC A'].apply(utils.get_pmhc_sequence)

# Check for HLAs missing pseudo sequences
missing_pseudos_validation_df = validation_df[validation_df['pseudo_sequence'].isnull()]
missing_pseudos_training_df = training_df[training_df['pseudo_sequence'].isnull()]

missing_hla_validation_list = missing_pseudos_validation_df['MHC A'].tolist()
missing_hla_training_list = missing_pseudos_training_df['MHC A'].tolist()

all_missing_hla_series = pd.Series(missing_hla_validation_list + missing_hla_training_list)
all_missing_hla_series = all_missing_hla_series.unique()
all_missing_hla_series = pd.Series(all_missing_hla_series)

all_missing_hla_series.to_csv(missing_hla_output_file, sep='\t', index=False, header=False)
print(f"Successfully saved {len(all_missing_hla_series)} entries to '{missing_hla_output_file}'!")

### Data Encoding ###

final_validation_df = pd.DataFrame()
final_training_df = pd.DataFrame()

final_validation_df['complex.id'] = validation_df['complex.id']
final_training_df['complex.id'] = training_df['complex.id']
final_validation_df['V_alpha'] = validation_df['V_alpha']
final_validation_df['V_beta'] = validation_df['V_beta']
final_training_df['V_alpha'] = training_df['V_alpha']
final_training_df['V_beta'] = training_df['V_beta']

# Encode paired TCR-antigen sequences
final_validation_paired = final_validation_df[final_validation_df['complex.id'] != 0].copy()
final_training_paired = final_training_df[final_training_df['complex.id'] != 0].copy()

final_validation_paired['tcr_encoded'] = (
    validation_df.loc[validation_df['complex.id'] != 0, 'CDR1_alpha'].astype(str)
    + '-' +
    validation_df.loc[validation_df['complex.id'] != 0, 'CDR2_alpha'].astype(str)
    + '-' +
    validation_df.loc[validation_df['complex.id'] != 0, 'CDR3_alpha'].astype(str)
    + '-' +
    validation_df.loc[validation_df['complex.id'] != 0, 'CDR1_beta'].astype(str)
    + '-' +
    validation_df.loc[validation_df['complex.id'] != 0, 'CDR2_beta'].astype(str)
    + '-' +
    validation_df.loc[validation_df['complex.id'] != 0, 'CDR3_beta'].astype(str)
)

final_validation_paired['pmhc_encoded'] = (
    validation_df.loc[validation_df['complex.id'] != 0, 'pseudo_sequence']
    + '-' +
    validation_df.loc[validation_df['complex.id'] != 0, 'Epitope']
)
print(f'[Paired Validation] {len(final_validation_df)} pre-augmentation entries')

final_training_paired['tcr_encoded'] = (
    training_df.loc[training_df['complex.id'] != 0, 'CDR1_alpha'].astype(str)
    + '-' +
    training_df.loc[training_df['complex.id'] != 0, 'CDR2_alpha'].astype(str)
    + '-' +
    training_df.loc[training_df['complex.id'] != 0, 'CDR3_alpha'].astype(str)
    + '-' +
    training_df.loc[training_df['complex.id'] != 0, 'CDR1_beta'].astype(str)
    + '-' +
    training_df.loc[training_df['complex.id'] != 0, 'CDR2_beta'].astype(str)
    + '-' +
    training_df.loc[training_df['complex.id'] != 0, 'CDR3_beta'].astype(str)
)

final_training_paired['pmhc_encoded'] = (
    training_df.loc[training_df['complex.id'] != 0, 'pseudo_sequence']
    + '-' +
    training_df.loc[training_df['complex.id'] != 0, 'Epitope']
)
print(f'[Paired Training] {len(final_training_df)} pre-augmentation entries')

# Encode unpaired TCR-antigen sequences, replace missing information with 'x'
final_validation_unpaired = final_validation_df[final_validation_df['complex.id'] == 0].copy()
final_training_unpaired = final_training_df[final_training_df['complex.id'] == 0].copy()

final_validation_unpaired['tcr_encoded'] = (
    validation_df.loc[validation_df['complex.id'] == 0, 'CDR1_alpha'].fillna('x').astype(str)
    + '-' +
    validation_df.loc[validation_df['complex.id'] == 0, 'CDR2_alpha'].fillna('x').astype(str)
    + '-' +
    validation_df.loc[validation_df['complex.id'] == 0, 'CDR3_alpha'].fillna('x').astype(str)
    + '-' +
    validation_df.loc[validation_df['complex.id'] == 0, 'CDR1_beta'].fillna('x').astype(str)
    + '-' +
    validation_df.loc[validation_df['complex.id'] == 0, 'CDR2_beta'].fillna('x').astype(str)
    + '-' +
    validation_df.loc[validation_df['complex.id'] == 0, 'CDR3_beta'].fillna('x').astype(str)
)

final_validation_unpaired['pmhc_encoded'] = (
    validation_df.loc[validation_df['complex.id'] == 0, 'pseudo_sequence']
    + '-' +
    validation_df.loc[validation_df['complex.id'] == 0, 'Epitope']
)
print(f'[Unpaired Validation] {len(final_validation_df)} pre-augmentation entries')

final_training_unpaired['tcr_encoded'] = (
    training_df.loc[training_df['complex.id'] == 0, 'CDR1_alpha'].fillna('x').astype(str)
    + '-' +
    training_df.loc[training_df['complex.id'] == 0, 'CDR2_alpha'].fillna('x').astype(str)
    + '-' +
    training_df.loc[training_df['complex.id'] == 0, 'CDR3_alpha'].fillna('x').astype(str)
    + '-' +
    training_df.loc[training_df['complex.id'] == 0, 'CDR1_beta'].fillna('x').astype(str)
    + '-' +
    training_df.loc[training_df['complex.id'] == 0, 'CDR2_beta'].fillna('x').astype(str)
    + '-' +
    training_df.loc[training_df['complex.id'] == 0, 'CDR3_beta'].fillna('x').astype(str)
)

final_training_unpaired['pmhc_encoded'] = (
    training_df.loc[training_df['complex.id'] == 0, 'pseudo_sequence']
    + '-' +
    training_df.loc[training_df['complex.id'] == 0, 'Epitope']
)
print(f'[Unpaired Training] {final_training_unpaired.shape[0]} pre-augmenation entries')

# Mask entries that are missing V gene information
# missing_validation_paired = final_validation_paired[final_validation_paired['tcr_encoded'].str.contains('None')].copy()
# missing_validation_unpaired = final_validation_unpaired[final_validation_unpaired['tcr_encoded'].str.contains('None')].copy()
missing_training_paired = final_training_paired[final_training_paired['tcr_encoded'].str.contains('None')].copy()
# missing_training_unpaired = final_training_unpaired[final_training_unpaired['tcr_encoded'].str.contains('None')].copy()

# print(f'[Paired Validation] {missing_validation_paired.shape[0]} entries missing V gene information')
# print(f'[Unpaired Validation] {missing_validation_unpaired.shape[0]} entries missing V gene information')
print(f'[Paired Training] {missing_training_paired.shape[0]} entries missing V gene information')
# print(f'[Unpaired Training] {missing_training_unpaired.shape[0]} entries missing V gene information')

missing_training_paired['tcr_encoded'] = missing_training_paired['tcr_encoded'].str.replace('None', 'x')

final_training_unpaired = pd.concat(
    [
        final_training_unpaired,
        missing_training_paired
    ], ignore_index=True    
)
print(f'[Unpaired Training] {final_training_unpaired.shape[0]} entries after fixing missing information')

### Data Augmentation ###
final_validation_paired[['pos_neg']] = 1
final_training_paired[['pos_neg']] = 1
final_validation_unpaired[['pos_neg']] = 1
final_training_unpaired[['pos_neg']] = 1

# Generate additional positive examples for paired training dataset
pos_training_variations = [variation for _, row in final_training_paired.iterrows() \
                  for variation in utils.generate_variations(row, 1)]

# Generate negative training examples
negative_validation_paired_df = utils.generate_negatives(final_validation_paired)
negative_training_paired_df = utils.generate_negatives(final_training_paired)

# Generate even more negative training examples
neg_validation_paired_variations = [variation for _, row in negative_validation_paired_df.iterrows() \
                                    for variation in utils.generate_variations(row, 0)]
neg_training_paired_variations = [variation for _, row in negative_training_paired_df.iterrows() \
                                    for variation in utils.generate_variations(row, 0)]

pos_training_variations_df = pd.DataFrame(pos_training_variations)
neg_validation_paired_variations_df = pd.DataFrame(neg_validation_paired_variations)
neg_training_paired_variations_df = pd.DataFrame(neg_training_paired_variations)

# Concatenate negative examples to final validation DataFrame
final_validation_df = pd.concat(
    [
        final_validation_paired,
        final_validation_unpaired,
        negative_validation_paired_df,
        neg_validation_paired_variations_df
    ], ignore_index=True
)
print(f'[Validation] {len(final_validation_df)} post-augmentation entries')

# Concatenate paired and unpaired training dataframes into limited final training DataFrame
limited_final_training_df = pd.concat(
    [
        final_training_paired,
        final_training_unpaired,
        negative_training_paired_df
    ], ignore_index=True
)
print(f'[Limited Training] {len(limited_final_training_df)} post-augmentation entries')

# Concatenate positive and negative examples to final full training DataFrame
full_final_training_df = pd.concat(
    [
        limited_final_training_df,
        pos_training_variations_df,
        neg_training_paired_variations_df
        
    ], ignore_index=True
)
print(f'[Full Training] {len(full_final_training_df)} post-augmentation entries')

final_validation_df = final_validation_df.merge(
    full_final_training_df,
    how='left',
    indicator=True).query('_merge == "left_only"').drop(columns='_merge')
print(f'[Validation] {final_validation_df.shape[0]} entries after removing overlap')

final_validation_df.drop_duplicates(inplace=True)
print(f'[Validation] {len(final_validation_df)} entries after removing duplicates')
full_final_training_df.drop_duplicates(inplace=True)
print(f'[Training] {len(full_final_training_df)} entries after removing duplicates')

validation_unique_targets = set(final_validation_df['pmhc_encoded'])
print(f'[Validation] {len(validation_unique_targets)} unique targets')
training_unique_targets = set(full_final_training_df['pmhc_encoded'])
print(f'[Training] {len(training_unique_targets)} unique targets')
combined_unique_targets = validation_unique_targets.intersection(training_unique_targets) # TODO: This isn't working
# combined_targets = final_validation_df[final_validation_df['pmhc_encoded'].isin(full_final_training_df['pmhc_encoded'])]
print(f'[Validation + Training] {len(combined_unique_targets)} unique targets')

# Save final DataFrames to output files
final_validation_df.to_csv(validation_output_file, sep='\t', index=False)
print(f"Successfully saved {final_validation_df.shape[0]} entries to '{validation_output_file}'!")
full_final_training_df.to_csv(full_training_output_file, sep='\t', index=False)
print(f"Successfully saved {full_final_training_df.shape[0]} entries to '{full_training_output_file}'!")
limited_final_training_df.to_csv(limited_training_output_file, sep='\t', index=False)
print(f"Successfully saved {limited_final_training_df.shape[0]} entries to '{limited_training_output_file}'!")