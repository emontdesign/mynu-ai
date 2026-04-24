from flask import Flask, request, jsonify
from flask_cors import CORS
from duckduckgo_search import DDGS
import sys

app = Flask(__name__)
CORS(app)

@app.route('/', methods=['GET'])
def home():
    return "Maya AI Online - Pronto al servizio", 200

@app.route('/ask', methods=['POST'])
def chat():
    try:
        data = request.get_json(force=True)
        
        # Recupero dati dal bridge PHP
        query = data.get("query", "")
        nome_rist = data.get("nome", "il ristorante")
        menu_data = data.get("menu", "Menu non disponibile")
        hours_data = data.get("hours", "Orari non disponibili")
        giorno_oggi = data.get("giorno_settimana", "oggi") 

        if not query:
            return jsonify({"success": False, "error": "Domanda vuota"})

        # Il tuo prompt preferito adattato
        full_prompt = f"""
Sei Maya, l'assistente virtuale di {nome_rist}. Rispondi in italiano.
Oggi è {giorno_oggi}.

CONTESTO OBBLIGATORIO:
1. MENU (JSON): {menu_data}
2. ORARI (JSON): {hours_data}

REGOLE DI RISPOSTA:
- Per gli orari, l'array 'schedule' ha indici da 0 a 6: 0=Lunedì, 1=Martedì, 2=Mercoledì, 3=Giovedì, 4=Venerdì, 5=Sabato, 6=Domenica.
- Se chiedono 'oggi', guarda il campo 'is_open' e commenta gli orari del giorno corrente.
- Usa SOLO le informazioni fornite. Se non trovi un piatto o un orario, dì che non lo sai.
- Sii cordiale e usa le emoji 🍕.

DOMANDA DEL CLIENTE: {query}
"""

        # Chiamata a DuckDuckGo con la nuova sintassi corretta
        with DDGS() as ddgs:
            # Eseguiamo la richiesta chat
            response = ddgs.chat(full_prompt, model='gpt-4o-mini')
            
            # Se la risposta è valida, la restituiamo
            if response:
                return jsonify({
                    "success": True, 
                    "reply": str(response)
                })
            else:
                raise Exception("Risposta vuota dall'AI")

    except Exception as e:
        # Stampiamo l'errore nei log di Render per vederlo
        print(f"ERRORE MAYA: {str(e)}", file=sys.stderr)
        return jsonify({
            "success": False, 
            "error": str(e),
            "reply": "Scusami, ho avuto un piccolo rallentamento. Puoi riprovare?"
        })

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000)
