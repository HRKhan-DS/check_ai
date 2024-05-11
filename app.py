import streamlit as st
import numpy as np
import torch
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
from scipy.special import softmax
import re
from transformers import pipeline, set_seed

# Load the text generation pipeline with GPT-2 model
generator = pipeline('text-generation', model='gpt2')
set_seed(42)

def get_model():
    MODEL = "distilbert-base-uncased-finetuned-sst-2-english"
    tokenizer = DistilBertTokenizer.from_pretrained(MODEL)
    model = DistilBertForSequenceClassification.from_pretrained(MODEL)
    return tokenizer, model

def clean_text(txt):
    cleanText = re.sub('http\S+\s', ' ', txt)
    cleanText = re.sub('RT|cc', ' ', cleanText)
    cleanText = re.sub('#\S+\s', ' ', cleanText)
    cleanText = re.sub('@\S+', '  ', cleanText)
    cleanText = re.sub('[%s]' % re.escape("""!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~"""), ' ', cleanText)
    cleanText = re.sub(r'[^\x00-\x7f]', ' ', cleanText)
    cleanText = re.sub('\s+', ' ', cleanText)
    return cleanText

def generate_friendly_prompt(input_sentence):
    return f"That's fantastic! {input_sentence}"

def generate_funny_prompt(input_sentence):
    return f"You won't believe it, but {input_sentence}"

def generate_congratulating_prompt(input_sentence):
    return f"Congratulations on {input_sentence}"

def generate_questioning_prompt(input_sentence):
    return f"I'm curious, {input_sentence}"

def generate_disagreement_prompt(input_sentence):
    return f"Sorry, I have to disagree. {input_sentence}"

def main():
    tokenizer, model = get_model()

    st.title("Content Analysis and Comment Generation Tool")

    user_input = st.text_area('Enter Text to Analyze')
    button = st.button("Analyze")

    if user_input and button:
        text = clean_text(user_input)
        test_sample = tokenizer(text, padding=True, truncation=True, max_length=512, return_tensors='pt')
        output = model(**test_sample)
        scores = output[0][0].detach().numpy()
        scores = softmax(scores)
        
        d = {
            1: 'Positive',
            0: 'Negative'
        }
        
        y_pred = np.argmax(scores)
        
        st.write("Prediction: ", d[y_pred])
        st.write("Confidence: ", np.round(scores[y_pred], 4))

        # Define prompts for different types of comments
        prompts = {
            "Friendly": generate_friendly_prompt(user_input),
            "Funny": generate_funny_prompt(user_input),
            "Congratulating": generate_congratulating_prompt(user_input),
            "Questioning": generate_questioning_prompt(user_input),
            "Disagreement": generate_disagreement_prompt(user_input)
        }

        # Generate comments for each type
        # Generate comments for each type
        generated_comments = {}
        for comment_type, prompt in prompts.items():
            generated_comment = generator(prompt, max_length=50, num_return_sequences=1)[0]['generated_text']
            generated_comments[comment_type] = generated_comment

        # Print the generated comments for each type
        st.write("\nGenerated Comments:")
        for comment_type, comment in generated_comments.items():
            st.write(f"{comment_type}:")
            st.write(comment.strip())

if __name__ == "__main__":
    main()