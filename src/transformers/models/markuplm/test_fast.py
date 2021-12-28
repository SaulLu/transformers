from transformers import MarkupLMTokenizerFast


tokenizer = MarkupLMTokenizerFast(
    vocab_file="/Users/NielsRogge/Documents/vocab.json",
    merges_file="/Users/NielsRogge/Documents/merges.txt",
    tags_dict="/Users/NielsRogge/Documents/tags_dict.json",
)

page_name_1 = "/Users/NielsRogge/Documents/python_projecten/transformers/docs/_build/html/quicktour.html"
page_name_2 = "/Users/NielsRogge/Documents/python_projecten/transformers/docs/_build/html/performance.html"
page_name_3 = "/Users/NielsRogge/Documents/python_projecten/transformers/docs/_build/html/philosophy.html"

with open(page_name_1) as f:
    single_html_string = f.read()

multi_html_strings = []

with open(page_name_2) as f:
    multi_html_strings.append(f.read())
with open(page_name_3) as f:
    multi_html_strings.append(f.read())

# test not batched
encoding = tokenizer(single_html_string, return_tensors="pt")

for k, v in encoding.items():
    print(k, v.shape)

# print(tokenizer.decode(encoding.input_ids.squeeze().tolist()))

# # test batched
encoding = tokenizer(multi_html_strings, padding="max_length", max_length=512, truncation=True, return_tensors="pt")

for k, v in encoding.items():
    print(k, v.shape)

#print(tokenizer.decode(encoding.input_ids[0].tolist()))
#print(tokenizer.decode(encoding.input_ids[1].tolist()))

# test pair not batched
question = "what's her name?"
encoding = tokenizer(
    question, single_html_string, padding="max_length", max_length=30, truncation=True, return_tensors="pt"
)

print(tokenizer.decode(encoding.input_ids.squeeze().tolist()))

# test pair batched

questions = ["what's her name?", "can you tell me the address?"]
encoding = tokenizer(
    questions, multi_html_strings, padding="max_length", max_length=30, truncation=True, return_tensors="pt"
)

print(tokenizer.decode(encoding.input_ids[0].tolist()))
print(tokenizer.decode(encoding.input_ids[1].tolist()))
