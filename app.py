from flask import Flask, request, jsonify
from flask_cors import CORS
from duckduckgo_search import DDGS
import sys

app = Flask(__name__)
CORS(app)

@app.route('/', methods=['GET'])
def home():
    return "Maya AI (DDG-Style) Online", 200

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

        # Manteniamo il tuo prompt preferito come base di conoscenza
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

        # Esecuzione con DuckDuckGo (Leggerissimo, niente SIGKILL)
        with DDGS() as ddgs:
            # Usiamo gpt-4o-mini che è il più bilanciato
            response = ddgs.chat(full_prompt, model='gpt-4o-mini')
            
            return jsonify({
                "success": True, 
                "reply": str(response)
            })

    except Exception as e:
        print(f"ERRORE PYTHON: {str(e)}", file=sys.stderr)
        return jsonify({
            "success": False, 
            "error": str(e),
            "reply": "Scusami, ho avuto un piccolo problema tecnico. Riprova!"
        })

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000)
