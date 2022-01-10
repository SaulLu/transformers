from transformers import MarkupLMTokenizer, MarkupLMTokenizerFast


slow_tokenizer = MarkupLMTokenizer.from_pretrained("SaulLu/markuplm-base")
fast_tokenizer = MarkupLMTokenizerFast.from_pretrained("SaulLu/markuplm-base")

text = "<html> HuggingFace is solving NLP one commit at a time </html>"

encoding = fast_tokenizer(text, max_length=6, return_overflowing_tokens=True, return_tensors="pt")

for k, v in encoding.items():
    print(k, v.shape)

# tokens = tokenizer.tokenize(text)

# print("Tokens:", tokens)


# SPECIAL_TOKEN_1 = "[SPECIAL_TOKEN_1]"
# SPECIAL_TOKEN_2 = "[SPECIAL_TOKEN_2]"

# tokenizer.add_tokens([SPECIAL_TOKEN_1], special_tokens=True)
# tokenizer.add_special_tokens({"additional_special_tokens": [SPECIAL_TOKEN_2]})

# print(tokenizer.special_tokens_map)

# token_1 = tokenizer.tokenize(SPECIAL_TOKEN_1)
# token_2 = tokenizer.tokenize(SPECIAL_TOKEN_2)

# print("Token_1:", token_1)
# print("Token_2:", token_2)
