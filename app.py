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
    "microsoft/Phi-3-mini-4k-instruct",
    "Qwen/Qwen2.5-7B-Instruct",
    "mistralai/Mistral-7B-Instruct-v0.3"
]

@app.route('/', methods=['GET'])
def home():
    return "Maya AI (Perfect Timing) Online", 200

@app.route('/ask', methods=['POST'])
def chat():
    try:
        data = request.get_json(force=True)
        query = data.get("query", "")
        nome_rist = data.get("nome", "il ristorante")
        giorno_oggi = data.get("giorno_settimana", "oggi")
        
        # --- 1. PULIZIA MENU ---
        menu_raw = data.get("menu", [])
        if isinstance(menu_raw, str): menu_raw = json.loads(menu_raw)
        
        menu_clean = []
        if isinstance(menu_raw, list):
            for cat in menu_raw:
                if not isinstance(cat, dict): continue
                prod_clean = []
                for p in cat.get("prodotti", []):
                    if p.get("note") == "Nota prodotto": p["note"] = ""
                    prod_clean.append(p)
                if prod_clean:
                    cat["prodotti"] = prod_clean
                    menu_clean.append(cat)

        # --- 2. ESTRAZIONE CHIRURGICA ORARI (IL FIX) ---
        hours_raw = data.get("hours", {})
        if isinstance(hours_raw, str): hours_raw = json.loads(hours_raw)
        
        orari_stringa = "Non disponibili"
        if isinstance(hours_raw, dict):
            idx = hours_raw.get("status", {}).get("day_index")
            schedule = hours_raw.get("schedule", [])
            
            if idx is not None and len(schedule) > idx:
                oggi = schedule[idx] # Prendiamo SOLO i dati di oggi
                if isinstance(oggi, list):
                    turni = []
                    for t in oggi:
                        if t.get("is_closed") == 0:
                            # Puliamo le stringhe HH:MM:SS in HH:MM
                            ap = t.get("apertura", "")[:5]
                            ch = t.get("chiusura", "")[:5]
                            if ap and ch: turni.append(f"{ap} - {ch}")
                    
                    if turni:
                        orari_stringa = " | ".join(turni)
                    else:
                        orari_stringa = "Chiuso oggi"

        is_open_now = hours_raw.get("status", {}).get("is_open", False)
        # -----------------------------------------------

        system_instructions = f"""
Sei Maya, l'assistente virtuale di {nome_rist}. Rispondi in italiano.
Oggi è {giorno_oggi}. 

CONTESTO DISPONIBILE (USA SOLO SE RICHIESTO):
- ORARI DI OGGI: {orari_stringa}
- STATO ATTUALE: {"Aperto" if is_open_now else "Chiuso"}
- MENU: {menu_json}

REGOLE DI COMPORTAMENTO:
1. SALUTI: Se l'utente ti saluta (es. "ciao", "buongiorno"), rispondi cordialmente al saluto senza elencare orari o menu a meno che non vengano chiesti.
2. PERTINENZA: Fornisci gli orari SOLO se l'utente chiede informazioni su aperture, chiusure o orari. Fornisci il menu SOLO se l'utente chiede cosa si mangia, i piatti o i prezzi.
3. ORARI: Quando parli di orari, usa solo questi: {orari_stringa}. Ogni turno su una riga separata con emoji 🍕.
4. MENU: Ogni prodotto su una riga dedicata. Vai SEMPRE a capo dopo ogni piatto. Ignora le note vuote.
5. NESSUN CONSIGLIO: Non aggiungere mai frasi tipo "ti consiglio di venire", "ti aspettiamo". Sii precisa e basta.

STILE:
- Breve, cordiale e asciutta.
- Usa emoji 🍕.
- Se chiedono di ordinare, dì che non è possibile farlo in chat.
"""

        last_error = ""
        for model_id in MODELS:
            try:
                response = client.chat_completion(
                    model=model_id,
                    messages=[
                        {"role": "system", "content": system_instructions},
                        {"role": "user", "content": query}
                    ],
                    max_tokens=200,
                    temperature=0.1 # Minima creatività = Massima precisione
                )
                return jsonify({"success": True, "reply": response.choices[0].message.content})
            except:
                continue

        return jsonify({"success": False, "reply": "Riprova tra un istante! 🤖"})

    except Exception as e:
        return jsonify({"success": False, "error": str(e)})

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000)
