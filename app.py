import os
import json
from flask import Flask, request, jsonify
from flask_cors import CORS
from huggingface_hub import InferenceClient
import sys

app = Flask(__name__)
CORS(app)

HF_TOKEN = os.getenv("HF_TOKEN")
client = InferenceClient(token=HF_TOKEN)

# Modelli più piccoli e veloci (hanno meno probabilità di essere "occupati")
MODELS = [
    "Qwen/Qwen2.5-1.5B-Instruct",
    "meta-llama/Llama-3.2-1B-Instruct",
    "microsoft/Phi-3-mini-4k-instruct"
]

@app.route('/', methods=['GET'])
def home():
    return "Maya AI (Debug Mode) Online", 200

@app.route('/ask', methods=['POST'])
def chat():
    try:
        data = request.get_json(force=True)
        query = data.get("query", "")
        nome_rist = data.get("nome", "il ristorante")
        giorno_oggi = data.get("giorno_settimana", "oggi")
        
        # --- PULIZIA DATI ---
        menu_raw = data.get("menu", [])
        if isinstance(menu_raw, str):
            try: menu_raw = json.loads(menu_raw)
            except: menu_raw = []
        
        menu_clean = []
        if isinstance(menu_raw, list):
            for cat in menu_raw:
                if not isinstance(cat, dict): continue
                p_clean = []
                for p in cat.get("prodotti", []):
                    if p.get("note") == "Nota prodotto": p["note"] = ""
                    p_clean.append(p)
                if p_clean:
                    cat["prodotti"] = p_clean
                    menu_clean.append(cat)

        hours_raw = data.get("hours", {})
        if isinstance(hours_raw, str):
            try: hours_raw = json.loads(hours_raw)
            except: hours_raw = {}
        
        orari_stringa = "Non disponibili"
        is_open_now = False
        if isinstance(hours_raw, dict):
            idx = hours_raw.get("status", {}).get("day_index")
            is_open_now = hours_raw.get("status", {}).get("is_open", False)
            schedule = hours_raw.get("schedule", [])
            if idx is not None and len(schedule) > idx:
                turni = [f"{t.get('apertura', '')[:5]} - {t.get('chiusura', '')[:5]}" 
                         for t in schedule[idx] if t.get('is_closed') == 0]
                orari_stringa = " | ".join(turni) if turni else "Chiuso"

        # --- CHIAMATA AI ---
        system_instructions = f"Sei Maya, assistente di {nome_rist}. Oggi è {giorno_oggi}. Orari: {orari_stringa}. Stato: {'Aperto' if is_open_now else 'Chiuso'}. Sii breve e usa emoji 🍕."

        last_error = "Nessun modello ha risposto"
        for model_id in MODELS:
            try:
                response = client.chat_completion(
                    model=model_id,
                    messages=[{"role": "system", "content": system_instructions}, {"role": "user", "content": query}],
                    max_tokens=200,
                    temperature=0.2
                )
                return jsonify({"success": True, "reply": response.choices[0].message.content})
            except Exception as e:
                last_error = str(e)
                print(f"Modello {model_id} fallito: {last_error}", file=sys.stderr)
                continue

        # Se arriviamo qui, riportiamo l'errore REALE per capire cosa succede
        return jsonify({"success": False, "reply": f"Errore tecnico: {last_error}"})

    except Exception as e:
        return jsonify({"success": False, "reply": f"Errore critico: {str(e)}"})

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000)
