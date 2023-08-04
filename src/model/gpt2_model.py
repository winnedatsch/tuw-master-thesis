
from model.language_model import LanguageModel
from transformers import GPT2LMHeadModel, AutoTokenizer
from perplexity import perplexity

class GPT2Model(LanguageModel):
    def __init__(self, gpu):
        self.gpu = gpu
        self.tokenizer = AutoTokenizer.from_pretrained("gpt2-medium")
        self.model = GPT2LMHeadModel.from_pretrained("gpt2-medium").to(gpu)


    def score(self, texts):
        return perplexity(texts, self.model, self.tokenizer, self.gpu, batch_size=64)