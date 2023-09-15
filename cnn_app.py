import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from keras.models import model_from_json
import json

st.title("Aplikasi Prediksi Kategori")
st.title("Barang Kiriman PJT")
st.write("")

#load model
json_file = open("model.json","r")
loaded_model_json = json_file.read()
json_file.close()

load_model = model_from_json(loaded_model_json)
load_model.load_weights("model.h5")

#membaca data dari file JSON
with open('index_word.json', 'r') as file:
    index_word = json.load(file)

teks_baru = st.text_input("Masukkan uraian barang (huruf kecil tanpa tanda baca):")

#membagi teks menjadi kata-kata
kata_kata = teks_baru.split()

#mengkodekan setiap kata menjadi indeks
indeks_kata = [index for kata in kata_kata for index, word in index_word.items() if kata == word]

# mengonversi elemen-elemen daftar menjadi integer
indeks_kata = [int(index) for index in indeks_kata]

#mengelompokkan indeks menjadi sublist
max_kata = 100
indeks_kata_nested = [indeks_kata[i:i+max_kata] for i in range(0, len(indeks_kata), max_kata)]

#pads sequences
X_test = pad_sequences(indeks_kata_nested, maxlen = max_kata, padding="post")

#eager execution
tf.config.run_functions_eagerly(True)

#membuat widget input dan prediksi
def predict_category(X_test):
    predicted_probabilities = load_model.predict(X_test)
    predicted_class = np.argmax(predicted_probabilities, axis=1)
    return predicted_class

if st.button("PREDIKSI"):
    input_text = X_test

    #memanggil fungsi predict_category untuk mendapatkan predicted_class
    predicted_class = predict_category(input_text)

    if predicted_class == 0:
        keterangan = "Tas, koper, dan sejenisnya"
    elif predicted_class == 1:
        keterangan = "Produk tekstil, garmen, dan sejenisnya"
    elif predicted_class == 2:
        keterangan = "Alas kaki, sepatu, dan sejenisnya"
    elif predicted_class == 3:
        keterangan = "Lainnya (Non MFN)"
    else :
        keterangan = "Kategori tidak diketahui"

    st.write(f"<span style='font-size: 24px;'>KATEGORI: {predicted_class} = {keterangan}</span>", unsafe_allow_html=True)

st.write("")
st.write("")
st.write("Keterangan: Apabila muncul ERROR maka data belum tersedia saat ini.")

