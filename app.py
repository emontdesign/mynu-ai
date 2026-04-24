import os
from flask import Flask, request, jsonify
from flask_cors import CORS
from huggingface_hub import InferenceClient
import sys

app = Flask(__name__)
CORS(app)

# Recuperiamo il token dalle variabili d'ambiente di Render
HF_TOKEN = os.getenv("HF_TOKEN")

# Inizializziamo il client
client = InferenceClient(token=HF_TOKEN)
# Specifichiamo il modello (Mistral 7B v0.3 è ottimo per le chat)
MODEL_ID = "mistralai/Mistral-7B-Instruct-v0.3"

@app.route('/', methods=['GET'])
def home():
    return "Maya AI (Conversational Mode) Online", 200

@app.route('/ask', methods=['POST'])
def chat():
    try:
        data = request.get_json(force=True)
        
        query = data.get("query", "")
        nome_rist = data.get("nome", "il ristorante")
        menu_data = data.get("menu", "Menu non disponibile")
        hours_data = data.get("hours", "Orari non disponibili")
        giorno_oggi = data.get("giorno_settimana", "oggi") 

        if not query or not HF_TOKEN:
            return jsonify({"success": False, "error": "Configurazione mancante o domanda vuota"})

        # Istruzioni di sistema (Il "cervello" di Maya)
        system_instructions = f"""
Sei Maya, l'assistente virtuale di {nome_rist}. Rispondi in italiano.
Oggi è {giorno_oggi}.

CONTESTO:
- MENU: {menu_data}
- ORARI: {hours_data}

REGOLE:
- Orari 'schedule' (0-6): 0=Lun, 1=Mar, 2=Mer, 3=Gio, 4=Ven, 5=Sab, 6=Dom.
- Se chiedono 'oggi', controlla 'is_open' e il giorno corrente.
- Usa SOLO i dati forniti. Se non sai qualcosa, dillo gentilmente.
- Sii amichevole e usa emoji 🍕.
"""

        # Usiamo chat_completion (il compito 'conversational')
        response = client.chat_completion(
            model=MODEL_ID,
            messages=[
                {"role": "system", "content": system_instructions},
                {"role": "user", "content": query}
            ],
            max_tokens=500,
            temperature=0.7
        )
        
        # Estraiamo la risposta testuale
        final_reply = response.choices[0].message.content
        
        return jsonify({
            "success": True, 
            "reply": final_reply
        })

    except Exception as e:
        print(f"ERRORE MAYA: {str(e)}", file=sys.stderr)
        return jsonify({
            "success": False, 
            "error": str(e),
            "reply": "Scusami, ho avuto un piccolo corto circuito. Puoi riprovare?"
        })

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000)
