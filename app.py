from flask import Flask
from flask import request, jsonify, render_template

# Import the necessary modules
import pickle
import numpy as np
from keras.models import load_model
from keras.preprocessing.sequence import pad_sequences

# 导入字典
with open('sentiment-analysis/word_dict.pk', 'rb') as f:
    word_dictionary = pickle.load(f)
with open('sentiment-analysis/label_dict.pk', 'rb') as f:
    output_dictionary = pickle.load(f)

result_dict = {1: "积极", 2: "消极", 3: "中性"}

app = Flask(__name__)
app.config['JSON_AS_ASCII'] = False


@app.route('/')
def index():
    return "Hello World!"


@app.route("/inference", methods=["POST"])
def inference():
    sent = request.form.to_dict()['content']
    print(sent)

    # 数据预处理
    input_shape = 180
    x = [[word_dictionary[word] for word in sent]]
    x = pad_sequences(maxlen=input_shape, sequences=x, padding='post', value=0)

    # 载入模型
    model_save_path = 'sentiment-analysis/train_model.h5'
    lstm_model = load_model(model_save_path)

    # 模型预测
    y_predict = lstm_model.predict(x)
    label_dict = {v: k for k, v in output_dictionary.items()}
    print('输入语句: %s' % sent)
    print('情感预测结果: %s' % label_dict[np.argmax(y_predict)])
    print(type(label_dict[np.argmax(y_predict)]))
    print(label_dict)
    print((y_predict))
    return jsonify({"Code": 200, "Result": result_dict[label_dict[np.argmax(y_predict)]]})


@app.route('/NLP', methods=['GET', 'POST'])
def NLP():
        return render_template("NLP.html")



if __name__ == '__main__':
    app.run(debug=True)
