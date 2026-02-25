import urllib.request
import json

url = "http://127.0.0.1:5000/predict"
data = {
    "Age": 45,
    "Income": 80000,
    "LoanAmount": 20000,
    "CreditScore": 720,
    "EmploymentLength": 10,
    "LoanPurpose": "Debt Consolidation",
    "PreviousDefaults": 0,
    "DTI": 3.0
}

req = urllib.request.Request(url)
req.add_header('Content-Type', 'application/json; charset=utf-8')
jsondata = json.dumps(data)
jsondataasbytes = jsondata.encode('utf-8')   # needs to be bytes
req.add_header('Content-Length', len(jsondataasbytes))

response = urllib.request.urlopen(req, jsondataasbytes)
result = json.loads(response.read().decode('utf-8'))
print(json.dumps(result, indent=4))
