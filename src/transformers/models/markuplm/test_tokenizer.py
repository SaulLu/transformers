from transformers import MarkupLMTokenizer
from transformers import MarkupLMTokenizerFast

tokenizer = MarkupLMTokenizerFast.from_pretrained("microsoft/markuplm-base")

SPECIAL_TOKEN_1 = "[SPECIAL_TOKEN_1]"
SPECIAL_TOKEN_2 = "[SPECIAL_TOKEN_2]"

tokenizer.add_tokens([SPECIAL_TOKEN_1], special_tokens=True)
tokenizer.add_special_tokens({"additional_special_tokens": [SPECIAL_TOKEN_2]})

print(tokenizer.unique_no_split_tokens)
print(tokenizer.special_tokens_map)

token_1 = tokenizer.tokenize(SPECIAL_TOKEN_1)
token_2 = tokenizer.tokenize(SPECIAL_TOKEN_2)

print("Token_1:", token_1)
print("Token_2:", token_2)