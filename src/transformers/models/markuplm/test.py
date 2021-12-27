from transformers import MarkupLMTokenizer


tokenizer = MarkupLMTokenizer(
    vocab_file="/Users/NielsRogge/Documents/TAPAS/tapas-base-masklm/vocab.txt",
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

# test batched
encoding = tokenizer(multi_html_strings, padding="max_length", truncation=True, return_tensors="pt")

for k, v in encoding.items():
    print(k, v.shape)
