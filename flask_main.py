import numpy as np
from flask import Flask, abort, jsonify, request,render_template
import pickle
import json #ajouter dans requirements

my_model = pickle.load(open("./Faker_serialized.pkl","rb"))

app = Flask(__name__)
@app.route('/',methods = ['GET'])
def index():
        return render_template('index.html')
@app.route('/predicttxt', methods=['POST'])
def classify():
        if request.method == 'POST':
                
                file = request.form['article']
                #print(file)
                #data = request.get_json(force = True)
                predict_request = [file]
                predict_request = np.array(predict_request)
                percent,y_hat = my_model.predict_proba(predict_request),my_model.predict(predict_request)
                #10 mots qui nous ont le plus fait tiltés
                predict_request = [i.split() for i in predict_request]
                pro=[]
                dicts = np.array([])
                pr=[]
                for e in predict_request:
                        pr = my_model.predict_proba(e)
                        pro.append(pr)
                        #dicts[e] = [pr]

                #pro = [x for x in pro]
                #pro.tolist()
                
                #final = json.dumps(pro)
                #dict(zip(predict_request,pro))
                #print(list(zip(predict_request,pro[0])))
                



                
                pro = pro[0].tolist()
                return jsonify({
                        'type' : y_hat[0],
                        'proba_false':percent[0][0],
                        'proba_true':percent[0][1],
                        'Tilt': pro

                })
                 
@app.route('/predictapi', methods=['POST'])
def classifyapi():
        if request.method == 'POST':
                data = request.get_json(force = True)
                predict_request = [data['article']]
                predict_request = np.array(predict_request)
                percent,y_hat = my_model.predict_proba(predict_request),my_model.predict(predict_request)
                return jsonify({
                        'type' : y_hat[0],
                        'proba_false':percent[0][0],
                        'proba_true':percent[0][1]

                })
if __name__ == '__main__':
        app.run(port= 9000, debug= True)