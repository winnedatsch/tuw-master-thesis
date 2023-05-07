import json 
from pipeline.utils import sanitize

with open('../data/metadata/gqa_all_attribute.json') as f:
    all_attributes = json.load(f)


def extract_attributes(question):
    attributes = set()
    standalone_values = set()
    for operation in question['semantic']:
        op = operation['operation']
        if op == 'relate':
            relation_type = operation['argument'].split(',')[1]
            if relation_type.startswith('same '):
                attributes.add(relation_type[5:])
        elif op == 'query':
            attributes.add(operation['argument'])
        elif op == 'common':
            attributes.update(all_attributes.keys())
        elif (op == 'same' or op == 'different') and \
             operation['argument'] != 'type':
            attributes.add(operation['argument'])
        elif op == 'filter':
            if operation['argument'].startswith('not('):
                standalone_values.add(operation['argument'][4:-1])
            else:
                standalone_values.add(operation['argument'])
        elif op == 'choose':
            standalone_values.add(operation['argument'].split('|')[0])
            standalone_values.add(operation['argument'].split('|')[1])
        elif operation['operation'] == 'verify':
            standalone_values.add(operation['argument'])
        elif op.startswith('filter') or \
             (op.startswith('verify') and op != 'verify rel') or \
             op.startswith('same') or \
             op.startswith('different'):
            attributes.add(' '.join(op.split(' ')[1:]))
        elif op.startswith('choose') and \
             op != 'choose rel' and operation['argument'] != '':
            attributes.add(' '.join(op.split(' ')[1:]))
        
    return {sanitize(a) for a in attributes if a != 'name' and a != 'vposition' and a != 'hposition'}, \
           {sanitize(v) for v in standalone_values}


def extract_classes(question):
    classes = set()
    for operation in question['semantic']:
        if operation['operation'] == 'select':
            classes.add(operation['argument'].split('(')[0])
        elif operation['operation'] == 'relate':
            target_class = operation['argument'].split(',')[0]
            if target_class != '_':
                classes.add(target_class)
        elif operation['operation'] == 'choose rel':
            classes.add(operation['argument'].split(',')[0])
        elif operation['operation'] == 'verify rel':
            classes.add(operation['argument'].split(',')[0])
    return {sanitize(c) for c in classes}


def extract_relations(question):
    relations = set()
    for operation in question['semantic']:
        if operation['operation'] == 'relate':
            relation = operation['argument'].split(',')[1]
            if not relation.startswith('same '):
                relations.add(relation)
        elif operation['operation'] == 'choose rel':
            relations.add(operation['argument'].split(',')[1].split('|')[0])
            relations.add(operation['argument'].split(',')[1].split('|')[1])
        elif operation['operation'] == 'verify rel':
            relations.add(operation['argument'].split(',')[1])
    return {sanitize(r) for r in relations}