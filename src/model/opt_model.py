
from model.language_model import LanguageModel
from transformers import OPTForCausalLM, AutoTokenizer
from perplexity import perplexity
import torch


class OPTModel(LanguageModel):
    def __init__(self, gpu):
        self.gpu = gpu
        self.tokenizer = AutoTokenizer.from_pretrained("facebook/opt-1.3b")
        self.model = OPTForCausalLM.from_pretrained("facebook/opt-1.3b").to(gpu)


    def score(self, texts):
        return perplexity(texts, self.model, self.tokenizer, self.gpu, batch_size=64)