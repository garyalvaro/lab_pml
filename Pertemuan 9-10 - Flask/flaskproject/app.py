from pickle import TRUE
from flask import Flask, render_template, request, jsonify
import models as ml
from tqdm import tqdm
import numpy as np

app = Flask(__name__)

@app.route("/")
def index():
    return render_template('template.html')

@app.route('/training', methods=['POST','GET'])
def training():
	if request.method=="POST":
		file = request.files['file_train']

		#read_data
		raw_docs_train, raw_docs_val = ml.read_data(file)

		#preprocessing
		processed_docs_train = ml.preprocessing(raw_docs_train)
		processed_docs_val = ml.preprocessing(raw_docs_val)
        
		word_index, word_seq_train = ml.tokenize_training(processed_docs_train)
		word_seq_val = ml.tokenize_testing(processed_docs_val)

		#word-embedding
		ml.word_embedding(word_index)

		# contoh hasil preprocessing-tokenizing
		n = 100
		cth = processed_docs_train['teks'][n]
		cth_lower = processed_docs_train['lower'][n]
		cth_punctual = processed_docs_train['punctual'][n]
		cth_normalize = processed_docs_train['normalize'][n]
		cth_stemmed = processed_docs_train['stemmed'][n]
		cth_tokenized = str(word_seq_train[n])
		
		# Memasukkan contoh preprocessing ke format JSON
		output = {
			'cth':cth, 
			'cth_lower':cth_lower, 
			'cth_punctual':cth_punctual, 
			'cth_normalize':cth_normalize,
			'cth_stemmed':cth_stemmed, 
			'cth_tokenized':cth_tokenized
		}

		# Penentuan X dan Y
		X_train = word_seq_train
		X_test = word_seq_val

		Y_train = processed_docs_train['pengiriman']
		Y_test = processed_docs_val['pengiriman']

		# Pembuatan Model Identifikasi Aspek [KHUSUS PENGIRIMAN] --> BACA DISCLAIMER PERTEMUAN 7
		loss_train, accuracy_train, loss_val, accuracy_val = ml.model_building(X_train, Y_train, X_test, Y_test)

		# Menambahkan nilai loss dan akurasi ke output JSON
		output["loss_train"] = round(loss_train*100, 2)
		output["accuracy_train"] = round(accuracy_train*100, 2)
		output["loss_val"] = round(loss_val*100, 2)
		output["accuracy_val"] = round(accuracy_val*100, 2)

	return jsonify(output)


@app.route('/testing', methods=['POST','GET'])
def testing():
	if request.method=="POST":
		file = request.files['file_test']
		df_hasil, accuracy = ml.model_testing(file)
		data_output = []
		for i in range (len(df_hasil)):
			data_output.append({
					'id': str(i+1),
                    'teks': str(df_hasil['teks'][i]),
                    'pengiriman': str(df_hasil['pengiriman'][i]),
                    'pengiriman_pred': str(df_hasil['pengiriman_pred'][i]),
                })
		
		output_test = {
			'accuracy': round(accuracy*100, 2), 
			'data_output':data_output
		}

		return jsonify(output_test)



if __name__ == '__main__':
    app.run(debug=TRUE)
