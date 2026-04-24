import os
from flask import Flask, request, jsonify
from flask_cors import CORS
from huggingface_hub import InferenceClient
import sys

app = Flask(__name__)
CORS(app)

# --- CONFIGURAZIONE HUGGINGFACE ---
# Incolla qui il tuo Token (es: "hf_xxxxxxxxxxxxxxxxx")
HF_TOKEN = "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"

# Usiamo un modello potentissimo e gratuito: Mistral-7B-Instruct
client = InferenceClient("mistralai/Mistral-7B-Instruct-v0.3", token=HF_TOKEN)

@app.route('/', methods=['GET'])
def home():
    return "Maya AI (Stable Edition) Online", 200

@app.route('/ask', methods=['POST'])
def chat():
    try:
        data = request.get_json(force=True)
        
        query = data.get("query", "")
        nome_rist = data.get("nome", "il ristorante")
        menu_data = data.get("menu", "Menu non disponibile")
        hours_data = data.get("hours", "Orari non disponibili")
        giorno_oggi = data.get("giorno_settimana", "oggi") 

        if not query:
            return jsonify({"success": False, "error": "Domanda vuota"})

        # Il tuo prompt preferito intatto
        full_prompt = f"""<s>[INST] Sei Maya, l'assistente virtuale di {nome_rist}. Rispondi in italiano.
Oggi è {giorno_oggi}.

CONTESTO OBBLIGATORIO:
1. MENU (JSON): {menu_data}
2. ORARI (JSON): {hours_data}

REGOLE DI RISPOSTA:
- Per gli orari, l'array 'schedule' ha indici da 0 a 6: 0=Lunedì, 1=Martedì, 2=Mercoledì, 3=Giovedì, 4=Venerdì, 5=Sabato, 6=Domenica.
- Se chiedono 'oggi', guarda il campo 'is_open' e commenta gli orari del giorno corrente.
- Usa SOLO le informazioni fornite. Se non trovi un piatto o un orario, dì che non lo sai.
- Sii cordiale e usa le emoji 🍕.

DOMANDA DEL CLIENTE: {query} [/INST]"""

        # Chiamata ufficiale (Niente blocchi IP, niente SIGKILL)
        response = client.text_generation(
            full_prompt,
            max_new_tokens=500,
            temperature=0.7,
            repetition_penalty=1.2
        )
        
        return jsonify({
            "success": True, 
            "reply": str(response)
        })

    except Exception as e:
        print(f"ERRORE MAYA: {str(e)}", file=sys.stderr)
        return jsonify({
            "success": False, 
            "error": str(e),
            "reply": "Scusami, ho avuto un piccolo rallentamento. Riprova tra un istante!"
        })

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000)
