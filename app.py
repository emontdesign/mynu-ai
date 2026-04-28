import os
import json
from flask import Flask, request, jsonify
from flask_cors import CORS
from huggingface_hub import InferenceClient
import sys

app = Flask(__name__)
CORS(app)

# Recuperiamo il token dalle variabili d'ambiente di Render
HF_TOKEN = os.getenv("HF_TOKEN")
client = InferenceClient(token=HF_TOKEN)

MODELS = [
    "microsoft/Phi-3-mini-4k-instruct",
    "Qwen/Qwen2.5-7B-Instruct",
    "mistralai/Mistral-7B-Instruct-v0.3"
]

@app.route('/', methods=['GET'])
def home():
    return "Maya AI (Resilient & Clean) Online", 200

@app.route('/ask', methods=['POST'])
def chat():
    try:
        data = request.get_json(force=True)
        query = data.get("query", "")
        nome_rist = data.get("nome", "il ristorante")
        giorno_oggi = data.get("giorno_settimana", "oggi")
        
        # --- 1. PULIZIA DATI IN PYTHON (Risolve il problema alla radice) ---
        menu_raw = data.get("menu", [])
        menu_clean = []
        if isinstance(menu_raw, list):
            for cat in menu_raw:
                prodotti_raw = cat.get("prodotti", [])
                # Filtriamo i prodotti: puliamo le note "Nota prodotto"
                prodotti_clean = []
                for p in prodotti_raw:
                    if p.get("note") == "Nota prodotto":
                        p["note"] = "" # Cancelliamo la nota inutile
                    prodotti_clean.append(p)
                
                # Aggiungiamo la categoria solo se ha prodotti
                if prodotti_clean:
                    cat["prodotti"] = prodotti_clean
                    menu_clean.append(cat)
        
        menu_json = json.dumps(menu_clean)
        hours_raw = data.get("hours", {})
        hours_json = json.dumps(hours_raw)
        
        # Estraiamo lo stato attuale per passarlo esplicitamente
        is_open_now = hours_raw.get("status", {}).get("is_open", False)
        # -----------------------------------------------------------------

        if not query or not HF_TOKEN:
            return jsonify({"success": False, "error": "Configurazione mancante"})

        system_instructions = f"""
Sei Maya, l'assistente virtuale di {nome_rist}. Rispondi in italiano.
Oggi è {giorno_oggi}.
STATO ATTUALE (AD ADESSO): {"Aperto" if is_open_now else "Chiuso"}.

REGOLE MATEMATICHE PER GLI ORARI:
1. VERIFICA ORARIO: Se l'utente chiede "posso venire alle X?" o "siete aperti alle X?", DEVI ignorare lo stato attuale e controllare i turni in 'schedule' per il 'day_index' di oggi.
2. LOGICA DI CHIUSURA: Se l'orario richiesto cade tra la fine del primo turno e l'inizio del secondo (es. alle 15:00), DEVI dire che il locale è CHIUSO.
3. NESSUN CONSIGLIO: Non dire "ti consigliamo di venire", "ti aspettiamo". Di' solo se è aperto o chiuso e riporta i turni orari.
4. FORMATO: Ogni turno orario deve stare su una riga separata.

REGOLE RIGIDE PER IL MENU:
5. ISOLAMENTO: Ogni prodotto è indipendente. Non scambiare note o allergeni tra i piatti.
6. A CAPO: Ogni prodotto deve stare su una riga dedicata. È OBBLIGATORIO andare a capo dopo ogni piatto.
7. NOTE: Se vedi una nota (diversa da vuoto), scrivila tra parentesi accanto al prezzo. Se non c'è nota, non scrivere nulla.
8. CATEGORIE: Non inventare categorie. Usa solo quelle fornite.

STILE:
- Risposte asciutte, cordiali, elenchi puntati con emoji 🍕.
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
                    max_tokens=500,
                    temperature=0.2 # Abbassata per rendere l'AI meno creativa e più precisa
                )
                final_reply = response.choices[0].message.content
                return jsonify({"success": True, "reply": final_reply})
            except Exception as e:
                last_error = str(e)
                continue

        return jsonify({"success": False, "reply": "Scusami, riprova tra un minuto! 🤖"})

    except Exception as e:
        print(f"ERRORE CRITICO: {str(e)}", file=sys.stderr)
        return jsonify({"success": False, "error": str(e)})

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000)
