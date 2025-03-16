import pandas as pd
from collections import defaultdict
import json
import math


def analyze_category_tree(cat_df):
    children = defaultdict(list)
    for _, row in cat_df.iterrows():
        if pd.notna(row['parent_id']):
            children[row['parent_id']].append(row['cat_id'])

    def get_depth(cat_id, memo={}):
        if cat_id in memo:
            return memo[cat_id]

        parent = cat_df[cat_df['cat_id'] == cat_id]['parent_id'].values

        if len(parent) == 0 or pd.isna(parent[0]):
            memo[cat_id] = 0
            return 0
        
        depth = get_depth(parent[0], memo) + 1
        memo[cat_id] = depth
        return depth
    
    cat_df['depth'] = cat_df['cat_id'].apply(get_depth)
    
    def get_all_children(cat_id, all_children=None):
        if all_children is None:
            all_children = set()
        
        all_children.add(cat_id)
        for child in children.get(cat_id, []):
            get_all_children(child, all_children)
        
        return all_children
    
    def get_path_to_root(cat_id):
        path = [cat_id]
        current_id = cat_id
        
        while True:
            parent = cat_df[cat_df['cat_id'] == current_id]['parent_id'].values
            
            if len(parent) == 0 or pd.isna(parent[0]):
                break
                
            path.append(parent[0])
            current_id = parent[0]
        
        return path[::-1]
    
    cat_df['direct_children_count'] = cat_df['cat_id'].apply(lambda x: len(children.get(x, [])))
    
    cat_df['all_children_count'] = cat_df['cat_id'].apply(lambda x: len(get_all_children(x)) - 1)
    
    def get_full_path(cat_id):
        path_ids = get_path_to_root(cat_id)
        path_names = []
        
        for cid in path_ids:
            name = cat_df[cat_df['cat_id'] == cid]['cat_name'].values
            if len(name) > 0:
                path_names.append(name[0])
        
        return ' > '.join(path_names)
    
    cat_df['full_path'] = cat_df['cat_id'].apply(get_full_path)
    
    return cat_df


def build_label_to_path(df):
    df = df.copy()
    df['parent_id'] = df['parent_id'].apply(
        lambda x: None if (isinstance(x, float) and math.isnan(x)) else int(x) if x is not None else None
    )
    parent_map = dict(zip(df['cat_id'], df['parent_id']))
    
    label_to_path = {}
    
    def get_path(cat):
        path = []
        cur = cat
        while cur is not None:
            path.append(cur)
            cur = parent_map.get(cur, None)
        return list(reversed(path))
    
    for cat in df['cat_id'].unique():
        label_to_path[cat] = get_path(cat)
    return label_to_path


def lca_depth(path1, path2):
    d = 0
    for a, b in zip(path1, path2):
        if a == b:
            d += 1
        else:
            break
    return d