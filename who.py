# from transformers import (BertConfig, BertTokenizer)


# bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')     


from transformers import BertTokenizer

# Load the tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Example text and tokenization
text = "Transformers are amazing for NLP tasks!"
tokens = tokenizer.tokenize(text)

print("Tokens:", tokens)

# Convert tokens back to string
reconstructed_text = tokenizer.convert_tokens_to_string(tokens)
print("Reconstructed Text:", reconstructed_text)
