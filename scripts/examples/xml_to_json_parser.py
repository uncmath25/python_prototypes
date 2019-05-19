import argparse
import json
import time
import xml.sax


class XMLToJSONHandler(xml.sax.ContentHandler):

    def __init__(self, should_log):
        """
        Initialize necessary parsing state variables
        """
        self._json_dict = {}
        self._pointer_path = []

        self._depth_dict_counts = {}
        self._current_depth = 0
        self._was_last_tag_end = False
        self._current_content = ''

        self._should_log = bool(should_log)

        self._ATTRIBUTE_PREFIX = '@'
        self._CONTENT_PREFIX = '#text'

    def startElement(self, tag, attributes):
        """
        Hook method called during start tags
        """
        self._current_depth += 1
        if self._current_depth - 1 not in self._depth_dict_counts:
            self._depth_dict_counts[self._current_depth - 1] = {}
        if tag not in self._depth_dict_counts[self._current_depth - 1]:
            self._depth_dict_counts[self._current_depth - 1][tag] = 1
        else:
            self._depth_dict_counts[self._current_depth - 1][tag] += 1
        self._pointer_path.append([tag, int(self._depth_dict_counts[self._current_depth - 1][tag]) - 1])
        self._was_last_tag_end = False
        self._init_with_pointer_path()
        self._add_attributes_with_pointer_path(attributes.items())

    def endElement(self, tag):
        """
        Hook method called during end tags
        """
        if len(self._current_content) != 0:
            self._assign_with_pointer_path(self._current_content)
            self._current_content = ''
        if self._was_last_tag_end:
            del self._depth_dict_counts[len(self._depth_dict_counts) - 1]
        self._pointer_path.pop()
        self._was_last_tag_end = True
        self._current_depth -= 1

    def characters(self, content):
        """
        Hook method called between tags
        """
        self._current_content += str(content).strip()

    def _init_with_pointer_path(self):
        """
        Initializes or updates the current node
        """
        assignment_string = 'self._json_dict'
        for key_tuple in self._pointer_path[:-1]:
            assignment_string += '["' + key_tuple[0] + '"]' if key_tuple[1] == 0 else '["' + key_tuple[0] + '"][' + str(key_tuple[1]) + ']'
        assignment_string += '["' + self._pointer_path[-1][0] + '"]'
        if self._pointer_path[-1][1] == 0:
            assignment_string += '= {}'
        elif self._pointer_path[-1][1] == 1:
            assignment_string += ' = [str(' + assignment_string + ') if isinstance(' + assignment_string + ', str) else dict(' + assignment_string + '), {}]'
        else:
            assignment_string += '.append({})'

        if self._should_log:
            print('---------- INIT ----------')
            print(self._pointer_path)
            print(assignment_string)
            # exec('print(self._json_dict)')
            print('--------------------------')

        exec(assignment_string)

    def _add_attributes_with_pointer_path(self, attribute_pairs):
        """
        Add attributes at the current node
        """
        if len(attribute_pairs) == 0:
            return()

        attr_dict = {}
        assignment_string = 'self._json_dict'
        for key_tuple in self._pointer_path[:-1]:
            assignment_string += '["' + key_tuple[0] + '"]' if key_tuple[1] == 0 else '["' + key_tuple[0] + '"][' + str(key_tuple[1]) + ']'
        assignment_string += '["' + self._pointer_path[-1][0] + '"]'

        exec('global attr_dict; attr_dict = {}')
        for key, value in attribute_pairs:
            exec('attr_dict["' + self._ATTRIBUTE_PREFIX + '" + "' + key + '"] = "' + value + '"')

        if self._pointer_path[-1][1] == 0:
            assignment_string += ' = dict(' + str(attr_dict) + ')'
        elif self._pointer_path[-1][1] == 1:
            assignment_string += ' = [dict(' + assignment_string + '[0]), dict(' + str(attr_dict) + ')]'
        else:
            # assignment_string +=  '.append(dict(' + str(attr_dict) + '))'
            assignment_string += '[len(' + assignment_string + ')-1] = dict(' + str(attr_dict) + ')'

        if self._should_log:
            print('********** ATTRS **********')
            print(self._pointer_path)
            print(assignment_string)
            # exec('print(self._json_dict)')
            print('***************************')

        exec(assignment_string)

    def _assign_with_pointer_path(self, content):
        """
        Add content at the current node
        """
        assignment_string = 'self._json_dict'
        for key_tuple in self._pointer_path[:-1]:
            assignment_string += '["' + key_tuple[0] + '"]' if key_tuple[1] == 0 else '["' + key_tuple[0] + '"][' + str(key_tuple[1]) + ']'
        assignment_string += '["' + self._pointer_path[-1][0] + '"]'

        is_empty = None
        exec('global is_empty; is_empty = len(' + assignment_string + ') == 0')
        if self._pointer_path[-1][1] == 0:
            if not is_empty:
                assignment_string += '["' + self._CONTENT_PREFIX + '"]'
            assignment_string += ' = str(content)'
        elif self._pointer_path[-1][1] == 1:
            if is_empty:
                assignment_string += ' = [dict(assignment_string), str(content)]'
            else:
                assignment_string += '[1]["' + self._CONTENT_PREFIX + '"]  = str(content)'
        else:
            if is_empty:
                assignment_string += '.append(str(content))'
            else:
                assignment_string += '[len(' + assignment_string + ')-1]["' + self._CONTENT_PREFIX + '"]  = str(content)'

        if self._should_log:
            print('########## ASSIGN ##########')
            print(self._pointer_path)
            print(content)
            print(assignment_string)
            # exec('print(self._json_dict)')
            print('############################')

        exec(assignment_string)

    def get_json_dict(self):
        """
        Return the json dictionary
        """
        return self._json_dict


def main(xml_input_path, json_output_path, should_log):
    """
    Utilizes the xml parser to convert the given xml to json
    """
    start_time = time.time()

    parser = xml.sax.make_parser()
    parser.setFeature(xml.sax.handler.feature_namespaces, 0)
    xml_handler = XMLToJSONHandler(should_log)
    parser.setContentHandler(xml_handler)
    parser.parse(xml_input_path)

    with open(json_output_path, 'w') as f:
        json.dump(xml_handler.get_json_dict(), f)

    print("--- Parsing the Data from the XML File took {0} seconds ---".format(round(time.time() - start_time, 2)))


def run(xml_input_path, json_output_path):
    """
    Run the program using the cli inputs
    """
    XML_INPUT_PATH = str(xml_input_path)
    JSON_OUTPUT_PATH = str(json_output_path)

    SHOULD_LOG = False

    main(XML_INPUT_PATH, JSON_OUTPUT_PATH, SHOULD_LOG)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='XML to JSON Parser')
    parser.add_argument('xml_input_path', help='XML input path to be parsed')
    parser.add_argument('json_output_path', help='Output path for parsed json to be dumped')
    args = parser.parse_args()

    try:
        run(args.xml_input_path, args.json_output_path)
    except Exception as e:
        print(e)
