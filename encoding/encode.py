import json 
import re
from pattern.text.en import singularize

with open('../data/metadata/gqa_all_class.json') as f:
    categories = json.load(f)
class_to_category = {}
for category, classes in categories.items():
    for c in classes:
        if c not in class_to_category:
            class_to_category[c] = [category]
        else:
            class_to_category[c].append(category)

with open('../data/metadata/gqa_all_attribute.json') as f:
    attributes = json.load(f)
value_to_attribute = {}
for attribute, values in attributes.items():
    for v in values:
        if v not in value_to_attribute:
            value_to_attribute[v] = [attribute]
        else:
            value_to_attribute[v].append(attribute)



def sanitize(name):
    # source: DFOL-VQA
    plurale_tantum = ['this', 'yes', 'pants', 'shorts', 'glasses', 'scissors', 'panties', 'trousers', 'binoculars', 'pliers', 'tongs',\
        'tweezers', 'forceps', 'goggles', 'jeans', 'tights', 'leggings', 'chaps', 'boxers', 'indoors', 'outdoors', 'bus', 'octapus', 'waitress',\
        'pasta', 'pita', 'glass', 'asparagus', 'hummus', 'dress', 'cafeteria', 'grass', 'class']

    irregulars = {'shelves': 'shelf', 'bookshelves': 'bookshelf', 'olives': 'olive', 'brownies': 'brownie', 'cookies': 'cookie'}
    
    temp = name.strip().lower()
    if temp in irregulars:
        temp = irregulars[temp]
    elif not (temp.split(' ')[-1] in plurale_tantum or temp[-2:] == 'ss'):
        temp = singularize(temp)
    cleanup_regex = r'[^\w]'

    return re.sub(cleanup_regex, '_', temp)

def encode_sample(question):
    return (encode_scene(question['sceneGraph']), encode_question(question))

def encode_scene(scene_graph):
    scene_encoding = ""
    for oid, object in scene_graph['objects'].items():
        scene_encoding += f"object({oid}).\n"

        scene_encoding += f"has_attribute({oid}, class, {sanitize(object['name'])}).\n"
        scene_encoding += f"has_attribute({oid}, name, {sanitize(object['name'])}).\n"
       
        for category in class_to_category.get(sanitize(object['name']), []):
            scene_encoding += f"has_attribute({oid}, class, {category}).\n"    

        for value in object['attributes']:
            if value in value_to_attribute:
                for att in value_to_attribute[value]:
                    scene_encoding += f"has_attribute({oid}, {sanitize(att)}, {sanitize(value)}).\n"
            else:
                scene_encoding += f"has_attribute({oid}, any, {sanitize(value)}).\n"

        if (object['x'] + object['w']/2) > scene_graph['width']/3*2:
            scene_encoding += f"has_attribute({oid}, hposition, right).\n"
        elif (object['x'] + object['w']/2) > scene_graph['width']/3:
            scene_encoding += f"has_attribute({oid}, hposition, middle).\n"
        else:
            scene_encoding += f"has_attribute({oid}, hposition, left).\n"

        if (object['y'] + object['h']/2) > scene_graph['height']/3*2:
            scene_encoding += f"has_attribute({oid}, vposition, bottom).\n"
        if (object['y'] + object['h']/2) > scene_graph['height']/3:
            scene_encoding += f"has_attribute({oid}, vposition, middle).\n"
        else:
            scene_encoding += f"has_attribute({oid}, vposition, top).\n"

        for rel in object['relations']:
            scene_encoding += f"has_relation({oid}, {sanitize(rel['name'])}, {sanitize(rel['object'])}).\n"

    return scene_encoding

def encode_question(question):
    question_encoding = ""
    step_padding = 0
    ops_map = {}
    for i, operation in enumerate(question['semantic']):
        if len(operation['dependencies']) == 0:
            question_encoding += f"scene({i+step_padding}).\n"
            dependencies = [i + step_padding]
            step_padding = step_padding + 1
        else:
            dependencies = [ops_map[op] for op in operation['dependencies']]
        
        if operation['operation'] == 'select':
            target_class = sanitize(operation['argument'].split('(')[0])
            question_encoding += f"select({i+step_padding}, {dependencies[0]}, {target_class}).\n"

        elif operation['operation'] == 'relate':
            target_class = sanitize(operation['argument'].split(',')[0])
            relation_type = sanitize(operation['argument'].split(',')[1])
            if relation_type.startswith('same_'):
                question_encoding += f"relate_attr({i+step_padding}, {dependencies[0]}, {target_class}, {relation_type[5:]}).\n"
            else:
                position = 'subject' if operation['argument'].split(',')[2].startswith('s') else 'object'
                if target_class == '_':
                    question_encoding += f"relate_any({i+step_padding}, {dependencies[0]}, {relation_type}, {position}).\n"
                else:
                    question_encoding += f"relate({i+step_padding}, {dependencies[0]}, {target_class}, {relation_type}, {position}).\n"

        elif operation['operation'] == 'query':
            question_encoding += f"unique({i+step_padding}, {dependencies[0]}).\n"
            question_encoding += f"query_attr({i+step_padding+1}, {i+step_padding}, {operation['argument']}).\n"
            step_padding += 1

        elif operation['operation'] == 'exist':
            question_encoding += f"exist({i+step_padding}, {dependencies[0]}).\n"

        elif operation['operation'] == 'and':
            question_encoding += f"and({i+step_padding}, {dependencies[0]}, {dependencies[1]}).\n"

        elif operation['operation'] == 'or':
            question_encoding += f"or({i+step_padding}, {dependencies[0]}, {dependencies[1]}).\n"

        elif operation['operation'] == 'common':
            question_encoding += f"common({i+step_padding}, {dependencies[0]}, {dependencies[1]}).\n"

        elif operation['operation'] == 'filter':
            if operation['argument'].startswith('not('):
                value = sanitize(operation['argument'][4:-1])
                question_encoding += f"filter_any({i+step_padding}, {dependencies[0]}, {value}).\n"
                question_encoding += f"negate({i+step_padding+1}, {i+step_padding}, {dependencies[0]}).\n"
                step_padding = step_padding + 1
            else:
                value = sanitize(operation['argument'])
                question_encoding += f"filter_any({i+step_padding}, {dependencies[0]}, {value}).\n"

        elif operation['operation'] == 'choose':
            option0 = sanitize(operation['argument'].split('|')[0])
            option1 = sanitize(operation['argument'].split('|')[1])
            question_encoding += f"choose_attr({i+step_padding}, {dependencies[0]}, any, {option0}, {option1}).\n"

        elif operation['operation'] == 'choose rel':
            target_class = sanitize(operation['argument'].split(',')[0])
            option0 = sanitize(operation['argument'].split(',')[1].split('|')[0])
            option1 = sanitize(operation['argument'].split(',')[1].split('|')[1])
            position = 'subject' if operation['argument'].split(',')[2].startswith('s') else 'object'
            question_encoding += f"choose_rel({i+step_padding}, {dependencies[0]}, {target_class}, {option0}, {option1}, {position}).\n"

        elif operation['operation'] == 'same':
            if operation['argument'] == 'type':
                attr = 'class'
            else:
                attr = sanitize(operation['argument'])
            question_encoding += f"all_same({i+step_padding}, {dependencies[0]}, {attr}).\n"

        elif operation['operation'] == 'different':
            if operation['argument'] == 'type':
                attr = 'class'
            else:
                attr = sanitize(operation['argument'])
            question_encoding += f"all_different({i+step_padding}, {dependencies[0]}, {attr}).\n"

        elif operation['operation'].startswith('filter'):
            attr = sanitize(' '.join(operation['operation'].split(' ')[1:]))
            if operation['argument'].startswith('not('):
                value = sanitize(operation['argument'][4:-1])
                question_encoding += f"filter({i+step_padding}, {dependencies[0]}, {attr}, {value}).\n"
                question_encoding += f"negate({i+step_padding+1}, {i+step_padding}, {dependencies[0]}).\n"
                step_padding = step_padding + 1
            else:
                value = sanitize(operation['argument'])
                question_encoding += f"filter({i+step_padding}, {dependencies[0]}, {attr}, {value}).\n"

        elif operation['operation'] == 'verify':
            value = sanitize(operation['argument'])
            question_encoding += f"verify_attr({i+step_padding}, {dependencies[0]}, any, {value}).\n"
        
        elif operation['operation'] == 'verify rel':
            target_class = sanitize(operation['argument'].split(',')[0])
            relation_type = sanitize(operation['argument'].split(',')[1])
            position = 'subject' if operation['argument'].split(',')[2].startswith('s') else 'object'
            question_encoding += f"verify_rel({i+step_padding}, {dependencies[0]}, {target_class}, {relation_type}, {position}).\n"
        
        elif operation['operation'].startswith('verify'):
            attr = sanitize(' '.join(operation['operation'].split(' ')[1:]))
            value = sanitize(operation['argument'])
            question_encoding += f"verify_attr({i+step_padding}, {dependencies[0]}, {attr}, {value}).\n"

        elif operation['operation'].startswith('choose'):
            if operation['argument'] == '':
                op_tokens = operation['operation'].split(' ')
                if len(op_tokens) >= 3:
                    if sanitize(op_tokens[1]) == 'more':
                        question_encoding += f"compare({i+step_padding}, {dependencies[0]}, {dependencies[1]}, {sanitize(op_tokens[2])}, true).\n"
                    elif sanitize(op_tokens[1]) == 'less':
                        question_encoding += f"compare({i+step_padding}, {dependencies[0]}, {dependencies[1]}, {sanitize(op_tokens[2])}, false).\n"
                else:
                    token = sanitize(op_tokens[1])
                    if token.endswith('er'):
                        token = token[:-2]
                        if token.endswith('i'):
                            token = token[:-1] + 'y'

                    question_encoding += f"compare({i+step_padding}, {dependencies[0]}, {dependencies[1]}, {token}, true).\n"
            else:
                attr = sanitize(' '.join(operation['operation'].split(' ')[1:]))
                option0 = sanitize(operation['argument'].split('|')[0])
                option1 = sanitize(operation['argument'].split('|')[1])
                question_encoding += f"choose_attr({i+step_padding}, {dependencies[0]}, {attr}, {option0}, {option1}).\n"

        elif operation['operation'].startswith('same'):
            attr = sanitize(' '.join(operation['operation'].split(' ')[1:]))
            question_encoding += f"two_same({i+step_padding}, {dependencies[0]}, {dependencies[1]}, {attr}).\n"

        elif operation['operation'].startswith('different'):
            attr = sanitize(' '.join(operation['operation'].split(' ')[1:]))
            question_encoding += f"two_different({i+step_padding}, {dependencies[0]}, {dependencies[1]}, {attr}).\n" 
        
        ops_map[i] = i + step_padding

    question_encoding += f"end({len(question['semantic'])+step_padding-1})."
    return question_encoding

if __name__ == '__main__':
    question_and_scene_graph_file = '../data/questions/train_sampled_questions_10000.json'
    target_folder = '../data/encoded_questions'

    with open(question_and_scene_graph_file) as f:
        questions = json.load(f)
    
    for qid, question in questions.items():
        print(f"encoding question {qid}")
        scene_encoding, question_encoding = encode_question(question)
        with open(f"{target_folder}/{qid}.lp", "w") as f:
            f.write("% ------ scene encoding ------\n")
            f.write(scene_encoding)
            f.write("\n% ------ question encoding ------\n")
            f.write(question_encoding)

