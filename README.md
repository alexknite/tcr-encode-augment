# Summer 2024 NHGRI GREAT Summer Research Project

In collaboration with the [Jönsson Lab](https://jonssonlab.com/) , we created methods to do the following:
1. Collect TCR-antigen paired data from [VDJdb](https://vdjdb.cdr3.net/search) public database
2. Pair CDR1,2 and pseudo HLA sequences from proprietary datasets to matching V and MHC allele names collected
3. Encode TCR seqences using alpha and beta CDR1-3 sequences and target sequences with MHC pseudo and epitope sequences
4. Augment datasets by generating additional positive examples and generating negative examples
5. Export validation, limited, and full training datasets to be inputted into machine learning model to predict TCR-antigen interactivity

## Installation
### Establish virtual environment
```
python3 -m venv YOUR_VENV
```
### Activate virtual environment
```
source YOURVENV/bin/activate
```
### Install dependencies
```
pip install -r dependencies.txt
```

## Execute Program
```
python3 src/formalize_dataset.py
```

<!-- ## License

[MIT](https://choosealicense.com/licenses/mit/) -->
