import pandas as pd
import os
import numpy as np
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
import json
from keras.models import load_model
from keras.preprocessing.text import tokenizer_from_json


path_to_json_file_with_tokens = 'tokenizer.json'
path_to_model = "next_word_question_how_when_plus_where.h5"


def loading_data():
    loaded_model = load_model(path_to_model)

    with open(path_to_json_file_with_tokens, 'r', encoding='utf-8') as json_file:
        tokenizer_json = json_file.read()
        loaded_tokenizer = tokenizer_from_json(tokenizer_json)

    return loaded_model, loaded_tokenizer


def loading_dataframe():
    json_file_path = path_to_json_file_with_tokens

    with open(json_file_path, 'r', encoding='utf-8') as json_file:
        tokenizer_config = json.load(json_file)

    word_counts_json = tokenizer_config['config']['word_counts']

    word_counts = json.loads(word_counts_json)

    data = [(word, count) for word, count in word_counts.items()]

    data_frame = pd.DataFrame(data, columns=['Keyword', 'Volume'])
    return data_frame


def model_predict(seed_text):
    max_sequence_len = 25
    token_list = tokenizer.texts_to_sequences([seed_text])[0]
    token_list = pad_sequences([token_list], maxlen=max_sequence_len - 1, padding='pre')
    predicted = model.predict(token_list, verbose=0)
    return predicted


def model_results(num_predictions):
    top_n_predictions = np.argpartition(-predicted, num_predictions, axis=1)[:, :num_predictions]

    for i in range(num_predictions):
        predicted_word_indices = top_n_predictions[0, i]
        predicted_word = tokenizer.index_word[predicted_word_indices]
        print(f"Предсказанное следующее слово {i + 1}: {predicted_word}")


def find_top_words_result(letter):
    filtered_df = combined_df[
        combined_df['Keyword'].str.startswith(letter, na=False)]
    if not filtered_df.empty:
        sorted_df = filtered_df.sort_values(by='Volume', ascending=False)
        top_words = sorted_df.head(3)
        return top_words
    else:
        return False

print("Wait a few minutes, model is loading.")

model, tokenizer = loading_data()
combined_df = loading_dataframe()

full_text = ""
sentences_continue = ""
keyword = {}

while True:
    char = input("Char:").lower()
    full_text += char

    if char == " ":
        sentences_continue += full_text
        full_text = ""

        predicted = model_predict(sentences_continue)
        model_results(3)
    else:

        top_words = find_top_words_result(full_text)

        if top_words is not False:
            try:
                keyword[top_words.iloc[0]['Keyword']] = top_words.iloc[0]['Volume']
                keyword[top_words.iloc[1]['Keyword']] = top_words.iloc[1]['Volume']
                keyword[top_words.iloc[2]['Keyword']] = top_words.iloc[2]['Volume']
            except Exception as e:
                pass

            count = 1
            for i in keyword:
                print(count, i)
                count += 1

            keyword = {}
        else:
            print("No words")
