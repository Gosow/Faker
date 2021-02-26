import numpy as np
from flask import Flask, abort, jsonify, request
import pickle
import requests,json


url =  "https://fakernews.herokuapp.com/predictapi"#"http://localhost:9000/predictapi"
data = json.dumps({"article":"mario gave 1000$ to all the he saw"})
r = requests.post(url,data)
print(r.json())
