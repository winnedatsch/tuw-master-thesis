from models.base_model import BaseModel
from transformers import BlipModel, BlipImageProcessor, BertTokenizerFast, BatchEncoding

class BLIPModel(BaseModel):
    def __init__(self, gpu):
        super().__init__(img_size=384, gpu=gpu)

        self.model = BlipModel.from_pretrained("Salesforce/blip-itm-base-coco").to(gpu)
        self.image_processor = BlipImageProcessor.from_pretrained("Salesforce/blip-itm-base-coco")
        self.tokenizer = BertTokenizerFast.from_pretrained("Salesforce/blip-itm-base-coco")

    def image_preprocess(self, images):
        return self.image_processor(images, return_tensors="pt", do_resize=False).to(self.gpu)

    def text_preprocess(self, texts):
        text_inputs = self.tokenizer(texts, return_tensors="pt", padding=True)
        return BatchEncoding({k: text_inputs[k] for k in ("input_ids", "attention_mask")}).to(self.gpu)

    def score(self, images, texts):
        image_inputs = self.image_preprocess(images)
        text_inputs = self.text_preprocess(texts)
        
        result = self.model(**image_inputs, **text_inputs)
        return result["logits_per_image"]