import requests

request = requests.get('http://127.0.0.1:2000/result')
def getDocuments(dataset_list):
    documents = []
    keys = []
    for doc in dataset_list.iter():
        documents.append(doc[1])
        keys.append(doc[0])

def result() :
    output = request.get_json()
   
    if len(output.keys()) <2:
        return{ "Status":"BAD RESPONSE"}
    
    n1 = output['n1']
    n2 = output['n2']

    cal ={}
    cal['sum'] = int(n1) + int(n2)
    return (cal)