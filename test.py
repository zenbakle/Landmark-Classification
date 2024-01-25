import requests

# url = 'http://localhost:8080/2015-03-31/functions/function/invocations'
url = "https://hp8l79kej6.execute-api.us-east-2.amazonaws.com/landmark-url/predict"
data = {'url':"https://drive.google.com/uc?export=download&id=1HlsdKnbseV55R9qajoit5EJa4g6gibrR"
}

result = requests.post(url, json=data).json()
print(result)
