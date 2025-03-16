import pandas as pd
from collections import defaultdict
import json


def parse_attr_to_dict(s):
    fixed_data_str = s.replace('""', '"')
    data = json.loads(fixed_data_str)
    
    result = {}
    for item in data:
        key = item.get('attribute_name')
        value = item.get('attribute_value', '')
        if 'attribute_measure' in item:
            value = f"{value} {item['attribute_measure']}"
        result[key] = value
    
    return result
    

def parse_attr_to_str(s):
    return str(parse_attr_to_dict(s))


def parse_attr_keys_to_str(s):
    fixed_data_str = s.replace('""', '"')
    data = json.loads(fixed_data_str)
    
    result = {}
    for item in data:
        key = item.get('attribute_name')
        if key is None:
            continue
        value = item.get('attribute_value', '')
        if 'attribute_measure' in item:
            value = f"{value} {item['attribute_measure']}"
        result[key] = value
    
    return ', '.join(filter(None, result.keys()))


def parse_attr_selected_to_str(s, keys_with_good_values, only_good_keys):
    fixed_data_str = s.replace('""', '"')
    data = json.loads(fixed_data_str)
    
    result = []
    for item in data:
        key = item.get('attribute_name')
        value = item.get('attribute_value', '')
        if 'attribute_measure' in item:
            value = f"{value} {item['attribute_measure']}"
        
        if key in keys_with_good_values:
            result.append(f'{key}: {value}')
        elif key in only_good_keys:
            result.append(f'{key}: <some_value>')
    
    return '</s>'.join(result)