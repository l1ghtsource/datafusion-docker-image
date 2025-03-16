import pandas as pd
from utils import analyze_category_tree
from attr_parser import parse_attr_to_dict

WORKDIR = '/kaggle/input/data-fusion-2025/'

labeled_train = pd.read_parquet(WORKDIR + 'labeled_train.parquet')
unlabeled_train = pd.read_parquet(WORKDIR + 'unlabeled_train.parquet')
cat_tree = pd.read_csv(WORKDIR + 'category_tree.csv')

cat_tree = analyze_category_tree(cat_tree)
labeled_train = labeled_train.merge(cat_tree, on='cat_id', how='left')

unlabeled_train['cat_id'] = 'unlabeled'
unlabeled_train['parent_id'] = 'unlabeled'
unlabeled_train['cat_name'] = 'unlabeled'
unlabeled_train['depth'] = 'unlabeled'
unlabeled_train['direct_children_count'] = 'unlabeled'
unlabeled_train['all_children_count'] = 'unlabeled'
unlabeled_train['full_path'] = 'unlabeled'

train = pd.concat([labeled_train, unlabeled_train])
train['attributes'] = train['attributes'].apply(parse_attr_to_dict)

train.drop(columns=['hash_id'], axis=1, inplace=True)

key_values = {}
for attr_dict in train['attributes']:
    for key, value in attr_dict.items():
        key_values.setdefault(key, set()).add(value)

result_df = pd.DataFrame({
    'key': list(key_values.keys()),
    'typical_values': [list(values)[:10] for values in key_values.values()]
})

result_df.to_csv('keys_and_values.csv', index=False)