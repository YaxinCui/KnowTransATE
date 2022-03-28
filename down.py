from transformers import AutoModel, AutoTokenizer, AutoConfig

for model in ['yjernite/retribert-base-uncased']:
    c = AutoConfig.from_pretrained(model)
    m = AutoModel.from_pretrained(model) 
    t= AutoTokenizer.from_pretrained(model)
