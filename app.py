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
    return "Maya AI (Regole Integrate) Online", 200

@app.route('/ask', methods=['POST'])
def chat():
    try:
        data = request.get_json(force=True)
        query = data.get("query", "")
        nome_rist = data.get("nome", "il ristorante")
        giorno_oggi = data.get("giorno_settimana", "oggi")
        
        # --- PRE-ELABORAZIONE MENU (Regole 6, 7, 8, 9) ---
        menu_raw = data.get("menu", [])
        if isinstance(menu_raw, str):
            try: menu_raw = json.loads(menu_raw)
            except: menu_raw = []
        
        menu_clean_text = ""
        if isinstance(menu_raw, list):
            for cat in menu_raw:
                prodotti = cat.get("prodotti", [])
                if prodotti: # Regola 9: Salta categorie vuote
                    menu_clean_text += f"\nCATEGORIA: {cat.get('titolo')}\n"
                    for p in prodotti:
                        # Regola 7: Filtro Note
                        nota = p.get("note", "")
                        if not nota or nota.lower() == "nota prodotto":
                            nota_str = ""
                        else:
                            nota_str = f" ({nota})"
                        
                        # Regola 8: A capo (costruiamo la stringa riga per riga)
                        menu_clean_text += f"- {p.get('titolo')}: {p.get('prezzo')}€{nota_str}\n"

        # --- PRE-ELABORAZIONE ORARI (Regole 1, 2, 4, 5) ---
        hours_raw = data.get("hours", {})
        if isinstance(hours_raw, str):
            try: hours_raw = json.loads(hours_raw)
            except: hours_raw = {}
        
        idx = hours_raw.get("status", {}).get("day_index")
        is_open_now = hours_raw.get("status", {}).get("is_open", False)
        
        orari_oggi_testo = ""
        if idx is not None:
            schedule = hours_raw.get("schedule", [])
            if len(schedule) > idx:
                turni = [t for t in schedule[idx] if t.get("is_closed") == 0]
                for i, t in enumerate(turni):
                    label = "Mattina" if i == 0 else "Sera"
                    orari_oggi_testo += f"- {label}: {t.get('apertura')[:5]} - {t.get('chiusura')[:5]}\n"

        # --- SISTEMA DI ISTRUZIONI (Prompt con le tue Regole) ---
        system_instructions = f"""
Sei Maya, l'assistente virtuale di {nome_rist}. Rispondi in italiano.
Oggi è {giorno_oggi}.

REGOLE RIGIDE PER GLI ORARI (SISTEMA BINARIO):
1. SOLO OGGI: Rispondi usando solo questi orari: 
{orari_oggi_testo if orari_oggi_testo else "Chiuso oggi"}
2. DIVIETO DI RAGGRUPPAMENTO: Non citare altri giorni. Leggi solo i dati di oggi sopra forniti.
3. DIVIETO DI CONSIGLIO: È categoricamente vietato scrivere "ti consigliamo di...", "ti aspettiamo", "torna più tardi". Devi solo riportare i dati.
4. STATO: Se l'utente chiede se siete aperti ora, rispondi: "{"In questo momento siamo aperti" if is_open_now else "In questo momento siamo chiusi"}".

REGOLE RIGIDE PER IL MENU:
5. ISOLAMENTO: Associa note e prezzi solo al prodotto corrispondente. 
6. MENU DISPONIBILE:
{menu_clean_text if menu_clean_text else "Menu non disponibile."}
7. A CAPO: Ogni prodotto deve stare su una riga dedicata.

STILE:
- Risposte asciutte e cordiali.
- Usa l'emoji 🍕 alla fine delle informazioni.
- Se chiedono di ordinare, dì che non è possibile farlo in chat.
"""

        # Ciclo di chiamata ai modelli
        for model_id in MODELS:
            try:
                response = client.chat_completion(
                    model=model_id,
                    messages=[{"role": "system", "content": system_instructions}, {"role": "user", "content": query}],
                    max_tokens=300,
                    temperature=0.1
                )
                return jsonify({"success": True, "reply": response.choices[0].message.content})
            except: continue

        return jsonify({"success": False, "reply": "Riprova tra un istante! 🤖"})

    except Exception as e:
        return jsonify({"success": False, "error": str(e)})

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000)
