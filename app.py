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

MODELS = [
    "Qwen/Qwen2.5-1.5B-Instruct",
    "meta-llama/Llama-3.2-1B-Instruct",
    "microsoft/Phi-3-mini-4k-instruct"
]

@app.route('/', methods=['GET'])
def home():
    return "Maya AI (Hallucination Fix) Online", 200

@app.route('/ask', methods=['POST'])
def chat():
    try:
        data = request.get_json(force=True)
        query = data.get("query", "")
        nome_rist = data.get("nome", "il ristorante")
        giorno_oggi = data.get("giorno_settimana", "oggi")
        
        # --- 1. TRASFORMAZIONE MENU IN TESTO SEMPLICE ---
        menu_raw = data.get("menu", [])
        if isinstance(menu_raw, str):
            try: menu_raw = json.loads(menu_raw)
            except: menu_raw = []
        
        menu_text = ""
        if isinstance(menu_raw, list):
            for cat in menu_raw:
                prodotti = cat.get("prodotti", [])
                if prodotti:
                    menu_text += f"\nCATEGORIA: {cat.get('titolo', 'Altro')}\n"
                    for p in prodotti:
                        nota = p.get("note", "")
                        if nota == "Nota prodotto": nota = ""
                        prezzo = f"{p.get('prezzo')}€"
                        menu_text += f"- {p.get('titolo')} ({prezzo}) {nota}\n"

        if not menu_text:
            menu_text = "Menu al momento non disponibile."

        # --- 2. ESTRAZIONE ORARI ---
        hours_raw = data.get("hours", {})
        if isinstance(hours_raw, str):
            try: hours_raw = json.loads(hours_raw)
            except: hours_raw = {}
        
        orari_stringa = "Non disponibili"
        if isinstance(hours_raw, dict):
            idx = hours_raw.get("status", {}).get("day_index")
            schedule = hours_raw.get("schedule", [])
            if idx is not None and len(schedule) > idx:
                turni = [f"{t.get('apertura', '')[:5]} - {t.get('chiusura', '')[:5]}" 
                         for t in schedule[idx] if t.get('is_closed') == 0]
                orari_stringa = " | ".join(turni) if turni else "Chiuso"

        # --- 3. PROMPT CORAZZATO ---
        system_instructions = f"""Sei Maya, l'assistente di {nome_rist}. 
Oggi è {giorno_oggi}.
ORARI: {orari_stringa}.
MENU REALE:
{menu_text}

REGOLE ANTI-INVENZIONE:
1. USA SOLO IL MENU REALE SOPRA. È proibito inventare pizze, panini o prezzi.
2. Se l'utente chiede un piatto che NON è nel "MENU REALE", rispondi: "Mi dispiace, questo piatto non è disponibile".
3. Non aggiungere descrizioni poetiche se non sono nel testo.
4. Ogni piatto su una riga con emoji 🍕.
"""

        last_error = ""
        for model_id in MODELS:
            try:
                response = client.chat_completion(
                    model=model_id,
                    messages=[{"role": "system", "content": system_instructions}, {"role": "user", "content": query}],
                    max_tokens=300,
                    temperature=0.1 # Fondamentale: vicina allo zero per non inventare nulla
                )
                return jsonify({"success": True, "reply": response.choices[0].message.content})
            except Exception as e:
                last_error = str(e)
                continue

        return jsonify({"success": False, "reply": f"Errore tecnico: {last_error}"})

    except Exception as e:
        return jsonify({"success": False, "reply": f"Errore critico: {str(e)}"})

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000)
