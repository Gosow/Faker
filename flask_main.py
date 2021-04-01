import numpy as np
from flask import Flask, abort, jsonify, request,render_template
import pickle
import json #ajouter dans requirements
import helpers as utilss
from textblob import TextBlob
from flask_mail import Mail, Message
my_model = pickle.load(open("./Faker_serialized.pkl","rb"))
stopwords = pickle.load(open("./Faker_stopwords_en","rb"))               
app = Flask(__name__)
mail= Mail(app)
app.config['MAIL_SERVER']='smtp.gmail.com'
app.config['MAIL_PORT'] = 465
app.config['MAIL_USERNAME'] = 'enteryourmail@gmail.com'
app.config['MAIL_PASSWORD'] = 'enteryourpassword'
app.config['MAIL_USE_TLS'] = False
app.config['MAIL_USE_SSL'] = True
mail = Mail(app)



@app.route("/send",methods =['POST'])
def index_mail():
        file = request.get_json(force = True)
        file = file['form']

        msg = Message('NEW', sender = 'FakerNewsmails@gmail.com', recipients = ['FakerNewsmails@gmail.com'])
        msg.body = file
        mail.send(msg)
        return "Sent"

@app.route('/',methods = ['GET'])
def index():
        return render_template('index.html')
@app.route('/predicttxt', methods=['POST'])
def classify():
        if request.method == 'POST':
                punc = '''!()-[]{};:'"\, <>./?@#$%^&*_~'''
                
                file = request.form['article']
                for ele in file:  
                        if ele in punc:  
                                file = file.replace(ele, " ")
                
                predict_request = [file]
                predict_request = [i.split() for i in predict_request]
                print(predict_request)

                predict_request= [' '.join([word for word in predict_request[0] if not word in list(stopwords)])]
                analysis = TextBlob(str(predict_request))



                predict_request = np.array(predict_request)
                percent,y_hat = my_model.predict_proba(predict_request),my_model.predict(predict_request)
             
                predict_request = [i.split() for i in predict_request][0]
                print(predict_request)
                pro=[]
                dicts =[]
                pr=[]
                dictio = {}
                
              
                for e in predict_request:
                        pr = my_model.predict_proba([e])
                        #print(my_model.predict_proba([e]))
                        pro.append(pr)
                        dicts.append(e)
                        if e not in dictio:
                                dictio[e] = 0 
                        dictio[e] += 1
            
                pro = pro[0].tolist()
                
                return jsonify({
                        
                        'type' : y_hat[0],
                        'proba_false':percent[0][0],
                        'proba_true':percent[0][1],
                        'polarity':analysis.sentiment.polarity,
                        'subjectivity':analysis.sentiment.subjectivity,
                        'Mots':dictio

                })
                 
@app.route('/predictapi', methods=['POST'])
def classifyapi():
        if request.method == 'POST':
               
                punc = '''!()-[]{};:'"\, <>./?@#$%^&*_“”‘’~'''
                
                file = request.get_json(force = True)
                file = file['article']
                

                for ele in file:  
                        if ele in punc:  
                                file = file.replace(ele, " ")

                predict_request = [file]
                predict_request = [i.split() for i in predict_request]
                
               
                predict_request= [' '.join([word for word in predict_request[0] if not word in list(stopwords)])]
                lang=TextBlob(file).detect_language()
                analysis = TextBlob(str(predict_request))
             

                predict_request = np.array(predict_request)
                percent,y_hat = my_model.predict_proba(predict_request),my_model.predict(predict_request)

                
          
                
                predict_request = [i.split() for i in predict_request][0]
                pro=[]
                dicts =[]
                pr=[]
                dictio = {}
                

                for e in predict_request:
                        pr = my_model.predict_proba([e])
                        pro.append(pr)
                        dicts.append(e)
                        if e not in dictio:
                                dictio[e] = 0 
                        dictio[e] += 1
          
                pro = pro[0].tolist()
                dictio=utilss.hdict(dictio,15)
                return jsonify({
                        'language':lang,
                        'type' : y_hat[0],
                        'proba_false':np.round((percent[0][0])*100),
                        'proba_true':np.round(100*(percent[0][1])),
                        'polarity':np.round(100*analysis.sentiment.polarity),
                        'subjectivity':np.round(100*analysis.sentiment.subjectivity),
                        'Mots': dictio

                })
if __name__ == '__main__':
        app.run(port= 9000, debug= True)
