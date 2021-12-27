from transformers import MarkupLMModel, MarkupLMTokenizer

tokenizer = MarkupLMTokenizer.from_pretrained("markuplm-base-uncased")

page_name_1 = "page1.html"
page_name_2 = "page2.html"
page_name_3 = "page3.html"

with open(page_name_1) as f:
    single_html_string = f.read()

multi_html_strings = []

with open(page_name_2) as f:
    multi_html_strings.append(f.read())
with open(page_name_3) as f:
    multi_html_strings.append(f.read())

encoding = tokenizer(single_html_string, return_tensors="pt")

for k,v in encoding.items():
    print(k,v.shape)