import json 

question_and_scene_graph_file = '../../data/gqa/questions/train_sampled_questions.json'
target_folder = '../../data/gqa/encoded_questions'

with open('../../data/gqa/metadata/gqa_all_class.json') as f:
    categories = json.load(f)
class_to_category = {c: category for category, classes in categories.items() for c in classes}

with open('../../data/gqa/metadata/gqa_all_attribute.json') as f:
    attributes = json.load(f)
value_to_attribute = {v: attribute for attribute, values in attributes.items() for v in values}

with open(question_and_scene_graph_file) as f:
    questions = json.load(f)

for qid, question in questions.items():
    scene_encoding = ""
    for oid, object in question['sceneGraph']['objects'].items():
        scene_encoding += f"object({oid}).\n"

        scene_encoding += f"has_attribute({oid}, class, {object['name'].replace(' ', '_')}).\n"
        if object['name'] in class_to_category:
            scene_encoding += f"has_attribute({oid}, class, {class_to_category[object['name']].replace(' ', '_')}).\n"

        for att in object['attributes']:
            scene_encoding += f"has_attribute({oid}, {value_to_attribute.get(att, 'any')}, {att}).\n"

        for rel in object['relations']:
            scene_encoding += f"has_relation({oid}, {rel['name'].replace(' ', '_')}, {rel['object']}).\n"

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
            target_class = operation['argument'].split('(')[0].strip().replace(' ', '_')
            question_encoding += f"select({i+step_padding}, {dependencies[0]}, {target_class}).\n"

        elif operation['operation'] == 'relate':
            target_class = operation['argument'].split(',')[0].strip().replace(' ', '_')
            relation_type = operation['argument'].split(',')[1].strip().replace(' ', '_')
            if target_class == '_':
                question_encoding += f"relate_any({i+step_padding}, {dependencies[0]}, {relation_type}).\n"
            else:
                question_encoding += f"relate({i+step_padding}, {dependencies[0]}, {target_class}, {relation_type}).\n"

        elif operation['operation'] == 'query':
            question_encoding += f"query_attr({i+step_padding}, {dependencies[0]}, {operation['argument']}).\n"

        elif operation['operation'] == 'exist':
            question_encoding += f"exist({i+step_padding}, {dependencies[0]}).\n"

        elif operation['operation'] == 'and':
            question_encoding += f"and({i+step_padding}, {dependencies[0]}, {dependencies[1]}).\n"

        elif operation['operation'] == 'or':
            question_encoding += f"or({i+step_padding}, {dependencies[0]}, {dependencies[1]}).\n"

        elif operation['operation'] == 'filter':
            question_encoding += f"% filter - unsupported" 

        elif operation['operation'] == 'choose':
            option0 = operation['argument'].split('|')[0].strip().replace(' ', '_')
            option1 = operation['argument'].split('|')[1].strip().replace(' ', '_')
            question_encoding += f"choose_attr({i+step_padding}, {dependencies[0]}, any, {option0}, {option1}).\n"

        elif operation['operation'] == 'choose rel':
            target_class = operation['argument'].split(',')[0].strip().replace(' ', '_')
            option0 = operation['argument'].split(',')[1].split('|')[0].strip().replace(' ', '_')
            option1 = operation['argument'].split(',')[1].split('|')[1].strip().replace(' ', '_')
            question_encoding += f"choose_rel({i+step_padding}, {dependencies[0]}, {target_class}, {option0}, {option1}).\n"

        elif operation['operation'] == 'verify rel':
            target_class = operation['argument'].split(',')[0].strip().replace(' ', '_')
            relation_type = operation['argument'].split(',')[1].strip().replace(' ', '_')
            question_encoding += f"verify_rel({i+step_padding}, {dependencies[0]}, {target_class}, {relation_type}).\n"

        elif operation['operation'] == 'same':
            question_encoding += f"% all_same - unsupported" 

        elif operation['operation'] == 'different':
            question_encoding += f"% all_different - unsupported" 

        elif operation['operation'].startswith('filter'):
            attr = '_'.join(operation['operation'].split(' ')[1:])
            value = operation['argument'].strip().replace(' ', '_')
            question_encoding += f"filter({i+step_padding}, {dependencies[0]}, {attr}, {value}).\n"

        elif operation['operation'].startswith('verify'):
            attr = '_'.join(operation['operation'].split(' ')[1:])
            value = operation['argument'].strip().replace(' ', '_')
            question_encoding += f"verify_attr({i+step_padding}, {dependencies[0]}, {attr}, {value}).\n"

        elif operation['operation'].startswith('choose'):
            attr = '_'.join(operation['operation'].split(' ')[1:])
            option0 = operation['argument'].split('|')[0].strip().replace(' ', '_')
            option1 = operation['argument'].split('|')[1].strip().replace(' ', '_')
            question_encoding += f"choose_attr({i+step_padding}, {dependencies[0]}, {attr}, {option0}, {option1}).\n"

        elif operation['operation'].startswith('same'):
            attr = '_'.join(operation['operation'].split(' ')[1:]) 
            question_encoding += f"two_same({i+step_padding}, {dependencies[0]}, {dependencies[1]}, {attr}).\n"

        elif operation['operation'].startswith('different'):
            attr = '_'.join(operation['operation'].split(' ')[1:]) 
            question_encoding += f"two_different({i+step_padding}, {dependencies[0]}, {dependencies[1]}, {attr}).\n" 
        
        ops_map[i] = i + step_padding

    question_encoding += f"end({len(question['semantic'])+step_padding-1})."

    with open(f"{target_folder}/{qid}.lp", "w") as f:
        f.write("% ------ scene encoding ------\n")
        f.write(scene_encoding)
        f.write("\n% ------ question encoding ------\n")
        f.write(question_encoding)

