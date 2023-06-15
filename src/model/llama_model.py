
from model.language_model import LanguageModel
from transformers import LlamaForCausalLM, LlamaTokenizer
from perplexity import perplexity
import torch


class LlamaModel(LanguageModel):
    def __init__(self, path_to_weights):
        self.tokenizer = LlamaTokenizer.from_pretrained(path_to_weights)
        self.model = LlamaForCausalLM.from_pretrained(path_to_weights, low_cpu_mem_usage=True)


    def score(self, texts):
        return perplexity(texts, self.model, self.tokenizer, torch.device("cpu"))