from models.base_model import BaseModel
from transformers import CLIPModel as TCLIPModel, CLIPImageProcessor, CLIPTokenizer

class CLIPModel(BaseModel):
    def __init__(self, gpu):
        super().__init__(img_size=224, gpu=gpu)

        self.model = TCLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(gpu)
        self.image_processor = CLIPImageProcessor.from_pretrained("openai/clip-vit-base-patch32")
        self.tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")

    def image_preprocess(self, images):
        return self.image_processor(images, return_tensors="pt", do_resize=False, do_center_crop=False).to(self.gpu)

    def text_preprocess(self, texts):
        return self.tokenizer(texts, return_tensors="pt", padding=True).to(self.gpu)

    def score(self, images, texts):
        image_inputs = self.image_preprocess(images)
        text_inputs = self.text_preprocess(texts)
        
        result = self.model(**image_inputs, **text_inputs)
        return result["logits_per_image"]