from torchvision.io import read_image, ImageReadMode
from torchvision.transforms.functional import crop, resize, pad
from pipeline.concept_extraction import extract_classes, extract_attributes, extract_relations
from pipeline.bounding_box_optimization import get_object_bboxes, get_pair_bboxes
from pipeline.encoding.utils import cleanup_whitespace, sanitize_asp
import math
import torch
import re
import json

def bboxes_to_image_crops(bboxes, image, model, mode="pad"):
    bbox_crops = []
    for bbox in bboxes:
        y, x, h, w = int(bbox[0]), int(bbox[1]), int(bbox[2]-bbox[0]), int(bbox[3]-bbox[1])
        bbox_crop = crop(image, y, x, h, w)

        if mode == "pad":
            # resize and scale (maintain aspect ratio)
            if h > w:
                resize_dimensions = (
                    model.img_size, 2*round((model.img_size*w/h)/2))
            else:
                resize_dimensions = (
                    2*round((model.img_size*h/w)/2), model.img_size)
            bbox_crop = resize(bbox_crop, resize_dimensions, antialias=True)

            # pad the image to square dimensions
            bbox_crop = pad(bbox_crop, ((
                model.img_size - resize_dimensions[1])//2, (model.img_size - resize_dimensions[0])//2))

        elif mode == "scale":
            # resize and scale the image to the target dimensions
            bbox_crop = resize(
                bbox_crop, (model.img_size, model.img_size), antialias=True)
        else:
            raise RuntimeError("Unsupported image processing mode!")

        bbox_crops.append(bbox_crop)

    return bbox_crops


def prob_to_asp_weight(prob):
    return int(min(-1000*math.log(prob), 5000))


with open('../data/metadata/gqa_all_attribute.json') as f:
    all_attributes = json.load(f)

@torch.no_grad()
def encode_scene(question, model):
    scene_encoding = ""
    
    scene_graph = question['sceneGraph']
    objects = scene_graph['objects'].values()
    object_items = scene_graph['objects'].items()
    num_objects = len(objects)
    attributes, standalone_values = extract_attributes(question)
    num_attr_values = sum(len(all_attributes.get(attr, [])) for attr in attributes)
    num_standalone_values = len(standalone_values)
    classes = extract_classes(question)
    num_classes = len(classes)
    relations = extract_relations(question)
    num_relations = len(relations)

    for attr in attributes:
        for val in all_attributes.get(attr, []):
            scene_encoding += f"is_attribute_value({cleanup_whitespace(attr)}, {cleanup_whitespace(val)}).\n"
        scene_encoding += "\n"


    image = read_image(f"../data/images/{question['imageId']}.jpg", ImageReadMode.RGB)

    if len(attributes) > 0 or len(standalone_values) > 0 or len(classes) > 0:
        object_bboxes = get_object_bboxes(question)
        obj_bbox_crops = bboxes_to_image_crops(object_bboxes, image, model)
        attr_prompts = [f"a bad photo of a {val} object"
                        for object in objects
                        for attr in attributes
                        for val in all_attributes.get(attr, [])]
        attr_prompts.append("a bad photo of an object")
        standalone_value_prompts = [f"a bad photo of a {val} object"
                                    for object in objects
                                    for val in standalone_values]
        standalone_value_prompts.append("a bad photo of an object")
        class_prompts = [f"a bad photo of a {class_}" for class_ in classes]
        class_prompts.append("a bad photo of an object")

        obj_logits_per_image = model.score(
            obj_bbox_crops, [*attr_prompts, *standalone_value_prompts, *class_prompts])
            
    if len(relations) > 0:
        rel_bboxes, rel_bbox_indices = get_pair_bboxes(question, merge_threshold=0.6)
        rel_bbox_crops = bboxes_to_image_crops(rel_bboxes, image, model)
    
    # add attributes derived from object detection (names, vposition/hposition)
    for o1, (oid1, object1) in enumerate(object_items):
        scene_encoding += f"object({oid1}).\n"

        # scene_encoding += f"has_attribute({oid1}, class, {sanitize_asp(object['name'])}).\n"
        scene_encoding += f"has_attribute({oid1}, name, {sanitize_asp(object1['name'])}).\n"

        if (object1['x'] + object1['w']/2) > scene_graph['width']/3*2:
            scene_encoding += f"has_attribute({oid1}, hposition, right).\n"
        elif (object1['x'] + object1['w']/2) > scene_graph['width']/3:
            scene_encoding += f"has_attribute({oid1}, hposition, middle).\n"
        else:
            scene_encoding += f"has_attribute({oid1}, hposition, left).\n"

        if (object1['y'] + object1['h']/2) > scene_graph['height']/3*2:
            scene_encoding += f"has_attribute({oid1}, vposition, bottom).\n"
        elif (object1['y'] + object1['h']/2) > scene_graph['height']/3:
            scene_encoding += f"has_attribute({oid1}, vposition, middle).\n"
        else:
            scene_encoding += f"has_attribute({oid1}, vposition, top).\n"
        scene_encoding += "\n"

        if len(attributes) > 0:    
            attr_scores = torch.stack([
                obj_logits_per_image[o1, o1*(num_attr_values):(o1+1)*num_attr_values],
                obj_logits_per_image[o1, num_objects*num_attr_values].expand(num_attr_values)
            ])
            attr_probs = torch.nn.functional.softmax(attr_scores, dim=0)  

            # print(f"{num_attr_values}, {attr_probs.shape}")

            j = 0
            for attr in attributes:
                for val in all_attributes.get(attr, []): 
                    scene_encoding += f"{{has_attribute({oid1}, {cleanup_whitespace(attr)}, {cleanup_whitespace(val)})}}.\n"
                    scene_encoding += f":~ has_attribute({oid1}, {cleanup_whitespace(attr)}, {cleanup_whitespace(val)}). [{prob_to_asp_weight(attr_probs[0,j])}, ({oid1}, {cleanup_whitespace(attr)}, {cleanup_whitespace(val)})]\n"
                    scene_encoding += f":~ not has_attribute({oid1}, {cleanup_whitespace(attr)}, {cleanup_whitespace(val)}). [{prob_to_asp_weight(attr_probs[1,j])}, ({oid1}, {cleanup_whitespace(attr)}, {cleanup_whitespace(val)})]\n"
                    j += 1
                
            scene_encoding += "\n"

            del attr_scores, attr_probs

        if len(standalone_values) > 0:
            attr_indices = num_objects*(num_attr_values)+1
            standalone_scores = torch.stack([
                obj_logits_per_image[o1, attr_indices+o1*num_standalone_values:attr_indices+(o1+1)*num_standalone_values],
                obj_logits_per_image[o1, attr_indices+num_objects*num_standalone_values].expand(num_standalone_values)
            ])
            standalone_probs = torch.nn.functional.softmax(standalone_scores, dim=0)

            # print(f"{num_standalone_values}, {standalone_probs.shape}")

            k = 0
            for standalone_value_ in standalone_values:
                scene_encoding += f"{{has_attribute({oid1}, any, {cleanup_whitespace(standalone_value_)})}}.\n"
                scene_encoding += f":~ has_attribute({oid1}, any, {cleanup_whitespace(standalone_value_)}). [{prob_to_asp_weight(standalone_probs[0,k])}, ({oid1}, any, {cleanup_whitespace(standalone_value_)})]\n"
                scene_encoding += f":~ not has_attribute({oid1}, any, {cleanup_whitespace(standalone_value_)}). [{prob_to_asp_weight(standalone_probs[1,k])}, ({oid1}, any, {cleanup_whitespace(standalone_value_)})]\n"
                k += 1

            del standalone_scores, standalone_probs

        if len(classes) > 0:
            attr_standalone_indices = num_objects*(num_attr_values+num_standalone_values)+2
            class_scores = torch.stack([
                obj_logits_per_image[o1, attr_standalone_indices:attr_standalone_indices+num_classes],
                obj_logits_per_image[o1, attr_standalone_indices+num_classes].expand(num_classes)
            ])
            class_probs = torch.nn.functional.softmax(class_scores, dim=0)

            l = 0
            for class_ in classes:
                scene_encoding += f"{{has_attribute({oid1}, class, {cleanup_whitespace(class_)})}}.\n"
                scene_encoding += f":~ has_attribute({oid1}, class, {cleanup_whitespace(class_)}). [{prob_to_asp_weight(class_probs[0,l])}, ({oid1}, class, {cleanup_whitespace(class_)})]\n"
                scene_encoding += f":~ not has_attribute({oid1}, class, {cleanup_whitespace(class_)}). [{prob_to_asp_weight(class_probs[1,l])}, ({oid1}, class, {cleanup_whitespace(class_)})]\n"
                l += 1
            scene_encoding += "\n"

            del class_scores, class_probs

        # get cosine similarities between relations and every object pair's image crop
        if len(relations) > 0:
            rel_prompts = []
            for oid2, object2 in object_items:
                if oid2 != oid1:
                    for rel in relations:
                        rel_prompts.append(f"a bad photo of a {object1['name']} {rel} a {object2['name']}")
                    rel_prompts.append(f"a bad photo of a {object1['name']} and a {object2['name']}")

            rel_logits_per_image = model.score(rel_bbox_crops, rel_prompts)
            
            m = 0
            for o2, (oid2, object2) in enumerate(object_items):
                if oid1 != oid2:
                    rel_scores = torch.stack([
                        rel_logits_per_image[rel_bbox_indices[o1, o2], m*(num_relations+1):m*(num_relations+1)+num_relations],
                        rel_logits_per_image[rel_bbox_indices[o1, o2], m*(num_relations+1)+num_relations].expand(num_relations)
                    ])
                    rel_probs = torch.nn.functional.softmax(rel_scores, dim=0)

                    n = 0
                    for rel in relations:
                        scene_encoding += f"{{has_relation({oid1}, {cleanup_whitespace(rel)}, {oid2})}}.\n"
                        scene_encoding += f":~ has_relation({oid1}, {cleanup_whitespace(rel)}, {oid2}). [{prob_to_asp_weight(rel_probs[0,n])}, ({oid1}, {cleanup_whitespace(rel)}, {oid2})]\n"
                        scene_encoding += f":~ not has_relation({oid1}, {cleanup_whitespace(rel)}, {oid2}). [{prob_to_asp_weight(rel_probs[1,n])}, ({oid1}, {cleanup_whitespace(rel)}, {oid2})]\n"

                        n += 1
                    
                    scene_encoding += "\n"
                    m += 1

                    del rel_scores, rel_probs
                
            del rel_logits_per_image

    try: 
        del obj_logits_per_image
    except:
        pass

    return scene_encoding