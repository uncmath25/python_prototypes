import argparse
from collections import defaultdict
import csv
from functools import reduce
import json
import pandas as pd
import time


def is_collection(x):
    """
    Determines if the given object is a collection which is not a dictionary or a string
    """
    return hasattr(x, '__iter__') and (not isinstance(x, dict)) and (not isinstance(x, str))

def reduce_path(path, data, return_info=False):
    """
    Returns the traversed data according to the first collection splitting point of the given path
    """
    value = data
    for i, key in enumerate(path):
        try:
            value = value[key]
        # flake8: noqa
        except:
            raise KeyError(f'"{key}" not found in "{value}"')

        if is_collection(value):
            if return_info:
                return value
            else:
                return (path[:(i+1)], path[(i+1):], value)

    return value

def flatten(collection):
    """
    Flatten a potentially nested collection into a one-dimensional list
    """
    result = []

    for x in collection:
        if is_collection(x):
            result.extend(x)
        else:
            result.append(x)

    return result

def merge_items(x, y, return_info=False):
    """
    Merge the two collections based on shared keys
    """
    if is_collection(x) and is_collection(y):
        result = [merge_items(a, b) for a in x for b in y]
    elif is_collection(x):
        result = [merge_items(a, y) for a in x]
    elif is_collection(y):
        result = [merge_items(x, b) for b in y]
    else:
        result = x.copy()
        result.update(y)

    if is_collection(result) and len(result) == 1:
        return result[0]
    else:
        return result

def build_key_value_dict(key_path_pairs, data, is_initial=True):
    """
    Build the list of parsed dictionaries based upon recursive deduction
    """
    final_values = {}
    collections = defaultdict(list)

    for key, path, is_collection in key_path_pairs:
        value = reduce_path(path, data)
        if isinstance(value, tuple):
            parsed_path, leftover_path, value = value
            if is_collection and len(leftover_path) == 0:
                final_values[key] = value
            else:
                collections['.'.join(parsed_path)].append((key, leftover_path, is_collection))
        else:
            final_values[key] = value

    if collections:
        collections_dicts = []
        for sub_base_path, sub_field_map in collections.items():
            sub_base_path = sub_base_path.split('.')
            collection = reduce_path(sub_base_path, data, return_info=True)
            collection_dicts = flatten([build_key_value_dict(sub_field_map, x) for x in collection])
            collections_dicts.append(collection_dicts)
        reduced_collections = reduce(merge_items, collections_dicts)
        results = [merge_items(final_values, d) for d in reduced_collections]
    else:
        results = [final_values]

    return results


class JSONToDataFrameParser():

    def __init__(self, field_info_path, should_log):
        """
        Initialize necessary parsing state variables
        """
        self._should_log = bool(should_log)

        self._field_info = []
        with open(field_info_path, 'r') as f:
            reader = csv.reader(f)
            for row in reader:
                self._field_info.append((str(row[0]), str(row[1]), bool(row[2])))
        self._field_info = tuple(self._field_info)

        self._data_dict = None
        self._parsed_results = None

    def load_data(self, json_data_path):
        """
        Load the json data from the specified path
        """
        with open(json_data_path, 'r') as f:
            self._data_dict = json.load(f)

    def parse(self):
        """
        Parse the json data into a collection of dictionaries based upon the specified field info of the form (field_name, json_path, is_collection)
        """
        field_info = [(field, path_str.split('.'), is_collection) for field, path_str, is_collection in self._field_info]
        self._parsed_results = build_key_value_dict(field_info, self._data_dict)

    def export_results(self, dataframe_output_path):
        """
        Export the parsed dataframe to the specified path
        """
        df = pd.DataFrame.from_records(self._parsed_results)
        df.to_csv(dataframe_output_path, header=True, index=False)


def main(json_input_path, field_info_path, dataframe_output_path, should_log):
    """
    Utilizes the json parser to convert the given json to a dataframe based upon the specified field info
    """
    start_time = time.time()

    parser = JSONToDataFrameParser(field_info_path, should_log)
    parser.load_data(json_input_path)
    parser.parse()
    parser.export_results(dataframe_output_path)

    print("--- Parsing the DataFrame from the JSON File took {0} seconds ---".format(round(time.time() - start_time, 2)))


def run(json_input_path, field_info_path, dataframe_output_path):
    """
    Run the program using the cli inputs
    """
    JSON_INPUT_PATH = str(json_input_path)
    FIELD_INFO_PATH = str(field_info_path)
    DATAFRAME_OUTPUT_PATH = str(dataframe_output_path)

    SHOULD_LOG = False

    main(JSON_INPUT_PATH, FIELD_INFO_PATH, DATAFRAME_OUTPUT_PATH, SHOULD_LOG)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='JSON to DataFrame Parser')
    parser.add_argument('json_input_path', help='JSON input path to be parsed')
    parser.add_argument('field_info_path', help='CSV input path for the field info to be parsed')
    parser.add_argument('dataframe_output_path', help='Output path for the parsed dataframe to be exported')
    args = parser.parse_args()

    try:
        run(args.json_input_path, args.field_info_path, args.dataframe_output_path)
    except Exception as e:
        print(e)
