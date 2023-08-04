
from model.language_model import LanguageModel
from transformers import AutoModelForCausalLM, AutoTokenizer
from perplexity import perplexity

class GPTNeoModel(LanguageModel):
    def __init__(self, gpu):
        self.gpu = gpu
        self.tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neo-1.3B")
        self.model = AutoModelForCausalLM.from_pretrained("EleutherAI/gpt-neo-1.3B").to(gpu)


    def score(self, texts):
        return perplexity(texts, self.model, self.tokenizer, self.gpu, batch_size=64)