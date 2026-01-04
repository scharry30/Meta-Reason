import json
import os


def read_json(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error reading JSON file {file_path}: {e}")
        return []


def dump_json(file_path, data):
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)

        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=4)
        return True
    except Exception as e:
        print(f"Error writing JSON file {file_path}: {e}")
        return False


def merge_json_files(file_paths, output_path):
    merged_data = []

    for file_path in file_paths:
        data = read_json(file_path)
        if isinstance(data, list):
            merged_data.extend(data)
        else:
            merged_data.append(data)

    return dump_json(output_path, merged_data)


def filter_json_by_key(data, key, value):
    if isinstance(data, list):
        return [item for item in data if item.get(key) == value]
    elif isinstance(data, dict):
        return data if data.get(key) == value else {}
    return []


def extract_json_field(data, field):
    if isinstance(data, list):
        return [item.get(field) for item in data if field in item]
    elif isinstance(data, dict):
        return data.get(field)
    return None
