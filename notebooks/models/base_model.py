from abc import ABC, abstractmethod

class BaseModel(ABC):
    def __init__(self, img_size, gpu):
        self.img_size = img_size
        self.gpu = gpu
    
    @abstractmethod
    def image_preprocess(self, images):
        pass

    @abstractmethod
    def text_preprocess(self, texts):
        pass

    @abstractmethod
    def score(self, images, texts):
        pass