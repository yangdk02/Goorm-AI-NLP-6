import streamlit as st
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    BartForConditionalGeneration,
)

if 'tokenizer' not in st.session_state:
    classifier_tokenizer = AutoTokenizer.from_pretrained('klue/bert-base', use_fast = True)
    converter_tokenizer = AutoTokenizer.from_pretrained('gogamza/kobart-base-v2')
    st.session_state.tokenizer = classifier_tokenizer, converter_tokenizer
else:
    classifier_tokenizer, converter_tokenizer = st.session_state.tokenizer

@st.cache(hash_funcs={torch.nn.parameter.Parameter: lambda parameter: parameter.data.numpy()}, allow_output_mutation=True)
def get_classifier():
    model = AutoModelForSequenceClassification.from_pretrained('yangdk/bert-base-finetuned-spoken-written')
    model.eval()
    return model
@st.cache(hash_funcs={torch.nn.parameter.Parameter: lambda parameter: parameter.data.numpy()}, allow_output_mutation=True)
def get_spoken_to_written_converter():
    model = BartForConditionalGeneration.from_pretrained('yangdk/kobart-base-v2-finetuned-spoken-to-written-v2')
    model.eval()
    return model
@st.cache(hash_funcs={torch.nn.parameter.Parameter: lambda parameter: parameter.data.numpy()}, allow_output_mutation=True)
def get_written_to_spoken_converter():
    model = BartForConditionalGeneration.from_pretrained('yangdk/kobart-base-v2-finetuned-written-to-spoken-v2')
    model.eval()
    return model

classifier = get_classifier()
spoken_to_written_converter = get_spoken_to_written_converter()
written_to_spoken_converter = get_written_to_spoken_converter()

st.title("Spoken-Written Converter 🪄")

goal = st.radio("Set Goal 👇", ('Spoken 🗣️', 'Written ✍️'))

col1, col2 = st.columns(2)
with col1:
    input_text = st.text_area("Type your text here 👇")
    bnt = st.button('Convert 🪄')
with col2:
    if input_text and bnt:
        embeddings = classifier_tokenizer(input_text, return_attention_mask=False, return_token_type_ids=False, return_tensors='pt')
        output = classifier(**embeddings)
        preds = output.logits.argmax(dim=-1)
        predicted = 'Spoken 🗣️' if preds == 0 else 'Written ✍️'
        if predicted == goal:
            st.text_area(goal, value=input_text)
            st.success(f'No alerts 🥳')
        else:
            tokenized = converter_tokenizer(input_text, return_attention_mask=False, return_token_type_ids=False, return_tensors='pt')
            if predicted == 'Spoken 🗣️':
                out = spoken_to_written_converter.generate(**tokenized, max_length=128)[0, 1:-1]
                st.text_area("Written ✍️", value=converter_tokenizer.decode(out, skip_special_tokens=True))
            else:
                out = written_to_spoken_converter.generate(**tokenized, max_length=128)[0, 1:-1]
                st.text_area("Spoken 🗣️", value=converter_tokenizer.decode(out, skip_special_tokens=True))
            st.error(f'Your text is {predicted} style')