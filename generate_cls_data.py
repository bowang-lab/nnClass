import os
import json
import pandas as pd
from sklearn.model_selection import StratifiedKFold, StratifiedGroupKFold, train_test_split

def generate_cls_data(df, identifier_column, label_column, n_splits, output_path):
    identifiers = df[identifier_column].tolist()
    labels = df[label_column].tolist()
    cls_data = pd.DataFrame({
        'identifier': identifiers,
        'label': labels
    })
    cls_data.to_csv(os.path.join(output_path, 'cls_data.csv'), index=False)
    df['age_bin'] = pd.qcut(df['Age_at_StudyDate'], q=4, labels=False)
    df['stratify_col'] = df['Gender'].astype(str) + '_' + df['age_bin'].astype(str) +\
        '_' + df[label_column].astype(str)
    # df['stratify_col'] = df[label_column].astype(str)
    train_df, test_df = train_test_split(df, test_size=0.2, stratify=df['stratify_col'], random_state=42)
    # train_df['stratify_col'] = train_df['tumor_subtype'].astype(str) + \
    #     '_' + train_df[label_column].astype(str)
    print(f"Train size: {len(train_df)}, Test size: {len(test_df)}")
    print(f"Train label distribution:\n{train_df[label_column].value_counts()}")
    print(f"Test label distribution:\n{test_df[label_column].value_counts()}")
    test_df.to_csv(os.path.join(output_path, 'test_data.csv'), index=False)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    splits = []
    for fold, (train_idx, val_idx) in enumerate(skf.split(train_df, train_df['stratify_col'])):
        train_fold = train_df.iloc[train_idx]
        val_fold = train_df.iloc[val_idx]
        print(f'Fold {fold + 1}: Train size: {len(train_fold)}, Val size: {len(val_fold)}')
        print(f'Train labels distribution:\n{train_fold[label_column].value_counts(normalize=True)}')
        print(f'Val labels distribution:\n{val_fold[label_column].value_counts(normalize=True)}')
        splits.append({
            'train': train_fold[identifier_column].tolist(),
            'val': val_fold[identifier_column].tolist(),
        })
    
    with open(os.path.join(output_path, 'splits_final.json'), 'w') as f:
        json.dump(splits, f, indent=4)

    return


if __name__ == "__main__":
    import argparse
    import json

    argparser = argparse.ArgumentParser(description="Generate classification data and splits")
    argparser.add_argument('--input_path', '-i', type=str, required=True, help='Path to the input csv/excel file containing clinical and imaging info')
    argparser.add_argument('--output_path', '-o', type=str, required=True, help='Path to save the classification data and splits')
    argparser.add_argument('--identifier_column', '-id', type=str, default='patient_id', help='Column name for patient identifiers')
    argparser.add_argument('--label_column', '-label', type=str, default='label', help='Column name for classification labels')
    args = argparser.parse_args()

    # Load the dataset
    if args.input_path.endswith('.xlsx'):
        df = pd.read_excel(args.input_path)
    elif args.input_path.endswith('.csv'):
        df = pd.read_csv(args.input_path)
    else:
        raise ValueError("Input file must be a CSV or Excel file.")

    # Define the output path
    output_path = args.output_path
    os.makedirs(output_path, exist_ok=True)
    with open ('/home/jma/Documents/CY/cvpr26/hecktor25/Dataset723_hecktor25/no_segs.json') as f:
        no_segs = json.load(f)
    df = df[~df['PatientID'].isin(no_segs)]
    # Generate classification data
    generate_cls_data(df, args.identifier_column, args.label_column, n_splits=5, output_path=output_path)       




