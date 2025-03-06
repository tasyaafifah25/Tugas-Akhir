import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from werkzeug.utils import secure_filename
from flask import Flask, request, render_template
from preprocessing import final_tokenize, preproc_until_stopwords

## library

from gensim.models import FastText
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Embedding, GRU, Dropout, Dense
from tensorflow.keras.preprocessing.text import Tokenizer, tokenizer_from_json
from scikeras.wrappers import KerasClassifier
from sklearn.metrics import roc_curve

app = Flask(__name__, static_folder='static')
app.config['SECRET_KEY'] = 'secret-key-you-can-change'
app.config['UPLOAD_FOLDER'] = 'uploads'

if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'])

@app.route('/')
def index():
    return render_template('index.html', active_page="home")

# -------------------------------------------------
# Route Training
# -------------------------------------------------
@app.route("/training", methods=["GET", "POST"])
def training():
    """
    - GET  : menampilkan form upload (training.html).
    - POST : menerima file CSV, melakukan preprocessing,
             training, dan menampilkan hasil.
    """
    if request.method == "POST":
        file = request.files.get("csv_file", None)
        if file and file.filename.endswith(".csv"):
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)

            (df_preview, df_preview_stop, history_dict) = train_model(file_path)
            
            return render_template(
                "training.html",
                active_page="training",
                df_preview=df_preview,    
                df_preview_stop=df_preview_stop,
                history_dict=history_dict        
            )
        else:
            return render_template(
                "training.html",
                error_msg="File tidak valid. Mohon upload file CSV."
            )

    return render_template("training.html", active_page="training")

# ==============================
# FUNGSI TRAINING (INTI)
# ==============================
def train_model(csv_path):
    df = pd.read_csv(csv_path)

    df["stop_result"] = df["text"].apply(lambda x: preproc_until_stopwords(x) if isinstance(x, str) else "")

    df_preview_stop = df[["text", "stop_result"]].to_dict("records")

    df["tokens"] = df["stop_result"].apply(final_tokenize)

    df.dropna(subset=["label"], inplace=True)
    df = df[df["tokens"].apply(lambda x: isinstance(x, list) and len(x) > 0)]

    df_preview = df.to_dict("records")

    X = df["tokens"]
    y = df["label"]

    X_train, X_val, y_train, y_val = train_test_split(
        X, y,
        test_size=0.1,     
        random_state=42,
        stratify=y
    )

    # Fasttext
    all_sentences = df["tokens"].tolist()
    ft_model = FastText(
        vector_size=300,
        window=5,
        min_count=2,
        workers=4,
        sg=1,
        min_n=3,
        max_n=6
    )
    ft_model.build_vocab(all_sentences)
    ft_model.train(all_sentences, total_examples=len(all_sentences), epochs=20)

    tokenizer = Tokenizer(oov_token="<OOV>")
    tokenizer.fit_on_texts(X_train)
    
    tokenizer_json = tokenizer.to_json()
    with open("new_tokenizer.json", "w", encoding="utf-8") as f:
        f.write(tokenizer_json)

    X_train_seq = tokenizer.texts_to_sequences(X_train)
    X_val_seq   = tokenizer.texts_to_sequences(X_val)

    max_length = 50
    X_train_pad = pad_sequences(X_train_seq, maxlen=max_length, padding='post', truncating='post')
    X_val_pad   = pad_sequences(X_val_seq,   maxlen=max_length, padding='post', truncating='post')

    word_index = tokenizer.word_index
    vocab_size = len(word_index) + 1
    embedding_dim = 300

    embedding_matrix = np.zeros((vocab_size, embedding_dim))
    for word, idx in word_index.items():
        if idx < vocab_size:
            if word in ft_model.wv.key_to_index:
                embedding_matrix[idx] = ft_model.wv[word]
                
    ft_model.save("fasttext.bin")

    # GRU
    model = Sequential()
    model.add(Embedding(
        input_dim=vocab_size,
        output_dim=embedding_dim,
        weights=[embedding_matrix],
        input_length=max_length,
        trainable=False
    ))
    
    model.add(GRU(units=32, dropout=0.2, recurrent_dropout=0.2, return_sequences=True))
    model.add(GRU(units=16, dropout=0.2, recurrent_dropout=0.2, return_sequences=False))
    model.add(Dense(1, activation='sigmoid'))

    optimizer = Adam(learning_rate=0.005)
    model.compile(
        loss='binary_crossentropy',
        optimizer=optimizer,
        metrics=['accuracy']
    )

    # Training
    history = model.fit(
        X_train_pad,
        y_train,
        epochs=50,
        batch_size=32,
        validation_data=(X_val_pad, y_val),
        verbose=1
    )

    history_dict = history.history
    
    model.save("new_model.h5") 

    return df_preview, df_preview_stop, history_dict


@app.route('/testing', methods=["GET", "POST"])
def testing():
    if request.method == "POST":
        file = request.files.get("csv_file", None)
        if file and file.filename.endswith(".csv"):
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)

            df = pd.read_csv(file_path)

            df["stop_result"] = df["text"].apply(lambda x: preproc_until_stopwords(x) if isinstance(x, str) else "")

            df["tokens"] = df["stop_result"].apply(final_tokenize)

            df.dropna(subset=["text"], inplace=True)

            model_path = "new_model.h5" if os.path.exists("new_model.h5") else "models/gru_model_84.h5"
            model = load_model(model_path)

            if os.path.exists("new_tokenizer.json"):
                with open("new_tokenizer.json", "r", encoding="utf-8") as f:
                    tokenizer_json = f.read()
                tokenizer = tokenizer_from_json(tokenizer_json)
            else:
                with open("models/default_tokenizer.json", "r", encoding="utf-8") as f:
                    tokenizer_json = f.read()
                tokenizer = tokenizer_from_json(tokenizer_json)

            X_test = df["tokens"]
            X_test_seq = tokenizer.texts_to_sequences(X_test)
            max_length = 50
            X_test_pad = pad_sequences(X_test_seq, maxlen=max_length, padding='post', truncating='post')

            y_true = np.array(df["label"]).astype(np.int32).ravel()
            y_probs = model.predict(X_test_pad)
            fpr, tpr, thresholds = roc_curve(y_true, y_probs)
            J = tpr - fpr
            optimal_threshold = thresholds[np.argmax(J)]
            
            y_pred = (y_probs >= optimal_threshold).astype(int).ravel()

            if "label" in df.columns:
                cm_array = confusion_matrix(y_true, y_pred)
                cm_list = cm_array.tolist()
                classification_rep = classification_report(y_true, y_pred, output_dict=True)
                acc = accuracy_score(y_true, y_pred)
            else:
                y_true = None
                cm_list = None
                classification_rep = None
                acc = None

            plt.figure(figsize=(6, 4))
            ax = sns.heatmap(cm_array, annot=True, fmt="d", cmap="Blues", cbar=True)
            plt.ylabel("Actual")
            plt.xlabel("Predicted")
            plt.title("Confusion Matrix")
            plt.tight_layout()
            
            output_dir = "static/images"
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
                
            heatmap_path = os.path.join(output_dir, "cm_heatmap.png")
            plt.savefig(heatmap_path)
            plt.close()
            
            df["predicted_label"] = y_pred
            if "label" not in df.columns:
                df["label"] = ""
            df_display = df[["text", "stop_result", "label", "predicted_label"]].to_dict("records")

            return render_template("testing.html",
                                   df_display=df_display,
                                   confusion_matrix=cm_list,
                                   classification_report=classification_rep,
                                   accuracy=acc,
                                   model_used=model_path,
                                   active_page="testing",
                                   heatmap_image="images/cm_heatmap.png")
        else:
            return render_template("testing.html", active_page="testing", error_msg="File tidak valid. Mohon upload file CSV.")
    return render_template("testing.html", active_page="testing")

@app.route('/user-input', methods=["GET", "POST"])
def userInput():
    if request.method == "POST":
        sentence = request.form.get("sentence")
        if not sentence:
            error_msg = "Mohon masukkan kalimat untuk dicek."
            return render_template("user-input.html", active_page="user-input", error_msg=error_msg)
        
        preprocessed_text = preproc_until_stopwords(sentence)
        tokens = final_tokenize(preprocessed_text)
    
        token_str = " ".join(tokens)
        
        if os.path.exists("my_model.h5"):
            model_path = "my_model.h5"
        else:
            model_path = "models/gru_model_84.h5" 
        model = load_model(model_path)
        
        if os.path.exists("tokenizer.json"):
            with open("tokenizer.json", "r", encoding="utf-8") as f:
                tokenizer_json = f.read()
            tokenizer = tokenizer_from_json(tokenizer_json)
        else:
            with open("models/default_tokenizer.json", "r", encoding="utf-8") as f:
                tokenizer_json = f.read()
            tokenizer = tokenizer_from_json(tokenizer_json)
        
        max_length = 50
        seq = tokenizer.texts_to_sequences([token_str])
        padded_seq = pad_sequences(seq, maxlen=max_length, padding='post', truncating='post')
        
        y_prob = model.predict(padded_seq)
        y_pred = int((y_prob >= 0.5).astype(int)[0][0])
        result = "Provokasi Doxing" if y_pred == 1 else "Non-Provokasi Doxing"
        
        return render_template("user-input.html",
                               sentence=sentence,
                               preprocessed=preprocessed_text,
                               result=result,
                               active_page="user-input",
                               probability=y_prob[0][0],
                               model_used=model_path)
    
    return render_template("user-input.html", active_page="user-input")

app.run(debug=True)