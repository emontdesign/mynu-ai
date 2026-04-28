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
    return "Maya AI (Fixed Type Error) Online", 200

@app.route('/ask', methods=['POST'])
def chat():
    try:
        data = request.get_json(force=True)
        query = data.get("query", "")
        nome_rist = data.get("nome", "il ristorante")
        giorno_oggi = data.get("giorno_settimana", "oggi")
        
        # --- 1. GESTIONE SICURA DEI DATI IN INGRESSO ---
        menu_raw = data.get("menu", [])
        # Se menu_raw è una stringa (testo), proviamo a convertirlo in lista
        if isinstance(menu_raw, str):
            try:
                menu_raw = json.loads(menu_raw)
            except:
                menu_raw = []

        hours_raw = data.get("hours", {})
        # Se hours_raw è una stringa, proviamo a convertirlo in dizionario
        if isinstance(hours_raw, str):
            try:
                hours_raw = json.loads(hours_raw)
            except:
                hours_raw = {}

        # --- 2. PULIZIA DATI ---
        menu_clean = []
        if isinstance(menu_raw, list):
            for cat in menu_raw:
                if not isinstance(cat, dict): continue
                prodotti_raw = cat.get("prodotti", [])
                prodotti_clean = []
                if isinstance(prodotti_raw, list):
                    for p in prodotti_raw:
                        if not isinstance(p, dict): continue
                        if p.get("note") == "Nota prodotto":
                            p["note"] = ""
                        prodotti_clean.append(p)
                
                if prodotti_clean:
                    cat["prodotti"] = prodotti_clean
                    menu_clean.append(cat)
        
        # --- 3. ESTRAZIONE STATO APERTURA ---
        # Evitiamo il crash verificando che hours_raw sia un dizionario
        is_open_now = False
        if isinstance(hours_raw, dict):
            status = hours_raw.get("status", {})
            if isinstance(status, dict):
                is_open_now = status.get("is_open", False)
        
        menu_json = json.dumps(menu_clean)
        hours_json = json.dumps(hours_raw)
        # -----------------------------------------------

        if not query or not HF_TOKEN:
            return jsonify({"success": False, "error": "Configurazione mancante"})

        system_instructions = f"""
Sei Maya, l'assistente virtuale di {nome_rist}. Rispondi in italiano.
Oggi è {giorno_oggi}.
STATO ATTUALE: {"Aperto" if is_open_now else "Chiuso"}.

REGOLE PER GLI ORARI:
1. Se l'utente chiede "posso venire alle X?", controlla i turni in 'schedule' per oggi (day_index).
2. Se l'orario cade tra la fine del primo turno e l'inizio del secondo (es. 15:00), di' che è CHIUSO.
3. Riporta solo i dati. Niente consigli tipo "ti aspettiamo".
4. Ogni turno orario su una riga separata.

REGOLE PER IL MENU:
5. Ogni prodotto deve stare su una riga dedicata. Vai SEMPRE a capo dopo ogni piatto.
6. Se vedi una nota (non vuota), scrivila tra parentesi. Se è vuota, non scrivere nulla.
7. Non scambiare note tra piatti. Ogni {{ }} è indipendente.

STILE:
- Risposte brevi, amichevoli, elenchi puntati con emoji 🍕.
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
                    max_tokens=500,
                    temperature=0.2
                )
                final_reply = response.choices[0].message.content
                return jsonify({"success": True, "reply": final_reply})
            except Exception as e:
                last_error = str(e)
                continue

        return jsonify({"success": False, "reply": "Scusami, riprova tra un istante! 🤖"})

    except Exception as e:
        print(f"ERRORE CRITICO: {str(e)}", file=sys.stderr)
        return jsonify({"success": False, "error": str(e)})

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000)
