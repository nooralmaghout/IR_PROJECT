from flask import Flask,request

app = Flask(__name__)

@app.route("/result", methods = ["POST","GET"])
def result() :
    output = request.get_json()
   
    if len(output.keys()) <2:
        return{ "Status":"BAD RESPONSE"}
    
    n1 = output['n1']
    n2 = output['n2']

    cal ={}
    cal['sum'] = int(n1) + int(n2)
    return (cal)
    
@app.route("/preprocess", methods = ["POST","GET"])
def preprocess() :
    output = request.get_json()
   
    if len(output.keys()) <2:
        return{ "Status":"BAD RESPONSE"}
    
    n1 = output['n1']
    n2 = output['n2']

    cal ={}
    cal['sum'] = int(n1) + int(n2)
    return (cal)

if __name__ == '__main__':
    app.run(debug=True,port=2000)