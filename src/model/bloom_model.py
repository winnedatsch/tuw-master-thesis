
from model.language_model import LanguageModel
from transformers import BloomForCausalLM, AutoTokenizer
from perplexity import perplexity

class BloomModel(LanguageModel):
    def __init__(self, gpu):
        self.gpu = gpu
        self.tokenizer = AutoTokenizer.from_pretrained("bigscience/bloom-560m")
        self.model = BloomForCausalLM.from_pretrained("bigscience/bloom-560m").to(gpu)


    def score(self, texts):
        return perplexity(texts, self.model, self.tokenizer, self.gpu, batch_size=64)