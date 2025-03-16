import pandas as pd
import numpy as np

WORKDIR = '/kaggle/input/data-fusion-2025/'

labeled_train = pd.read_parquet(WORKDIR + 'labeled_train.parquet')
cat = pd.read_csv(WORKDIR + 'category_tree.csv')

cat['cat_name_lower'] = cat['cat_name'].str.lower()

groups = cat.groupby('cat_name_lower').agg({
    'cat_name': list,
    'cat_id': list
}).reset_index()

duplicate_groups = groups[groups['cat_name'].apply(len) > 1]

id_counts = labeled_train['cat_id'].value_counts().reset_index()
id_counts.columns = ['cat_id', 'count']

def get_most_popular(cat_ids, id_counts):
    group_counts = id_counts[id_counts['cat_id'].isin(cat_ids)]
    if not group_counts.empty:
        return group_counts.loc[group_counts['count'].idxmax(), 'cat_id']
    return None

duplicate_groups['most_popular'] = duplicate_groups['cat_id'].apply(
    lambda x: get_most_popular(x, id_counts)
)

duplicate_groups.to_csv('duplicate_groups.csv', index=False)