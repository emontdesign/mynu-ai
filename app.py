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
    return "Maya AI (Conversational Mode) Online", 200

@app.route('/ask', methods=['POST'])
def chat():
    try:
        data = request.get_json(force=True)
        query = data.get("query", "")
        nome_rist = data.get("nome", "il ristorante")
        giorno_oggi = data.get("giorno_settimana", "oggi")
        
        # --- 1. PREPARAZIONE MENU ---
        menu_raw = data.get("menu", [])
        if isinstance(menu_raw, str): menu_raw = json.loads(menu_raw)
        
        menu_text = ""
        if isinstance(menu_raw, list) and len(menu_raw) > 0:
            for cat in menu_raw:
                prodotti = cat.get("prodotti", [])
                if prodotti:
                    menu_text += f"\n- {cat.get('titolo')}:\n"
                    for p in prodotti:
                        nota = p.get("note", "")
                        nota_str = f"({nota})" if nota and nota.lower() != "nota prodotto" else ""
                        menu_text += f"  * {p.get('titolo')} a {p.get('prezzo')}€ {nota_str}\n"

        # --- 2. PREPARAZIONE ORARI ---
        hours_raw = data.get("hours", {})
        if isinstance(hours_raw, str): hours_raw = json.loads(hours_raw)
        
        orari_oggi = "purtroppo non ho i dati sottomano"
        is_open_now = False
        if isinstance(hours_raw, dict):
            status = hours_raw.get("status", {})
            is_open_now = status.get("is_open", False)
            idx = status.get("day_index")
            schedule = hours_raw.get("schedule", [])
            if idx is not None and len(schedule) > idx:
                turni = [f"dalle {t.get('apertura')[:5]} alle {t.get('chiusura')[:5]}" 
                         for t in schedule[idx] if t.get('is_closed') == 0]
                orari_oggi = " e ".join(turni) if turni else "oggi siamo chiusi"

        # --- 3. SYSTEM INSTRUCTIONS (Persona: Maya) ---
        system_instructions = f"""
Sei Maya, l'anima e l'assistente solare di {nome_rist}. Il tuo obiettivo è far sentire il cliente benvenuto.

PERSONALITÀ:
- Sei amichevole, educata e un pizzico spiritosa. 
- Rispondi come se fossi una persona reale al bancone del ristorante.
- Non elencare dati a meno che non ti venga chiesto qualcosa di specifico.

CONTESTO PER TE (NON RIPETERE SE NON RICHIESTO):
- Oggi è {giorno_oggi}.
- I nostri orari di oggi: {orari_oggi}.
- In questo momento siamo {"Aperti" if is_open_now else "Chiusi"}.
- Menu reale: {menu_text if menu_text else "Non disponibile al momento"}.

REGOLE DI CONVERSAZIONE:
1. SALUTI: Se ti dicono 'Ciao', rispondi calorosamente e chiedi come puoi essere utile. NON dire che giorno è o che orari facciamo a meno che non te lo chiedano.
2. ORARI: Se chiedono degli orari, rispondi in modo discorsivo (es: "Oggi siamo qui {orari_oggi}! 😊").
3. MENU: Se chiedono del menu, presenta i piatti in modo invitante. USA SOLO I PIATTI DEL MENU REALE. Non inventare nulla.
4. NESSUN ROBOT: Evita elenchi puntati freddi se puoi usare una frase gentile.
5. ORDINI: Se vogliono ordinare, spiega gentilmente che per ora non puoi prendere ordini in chat.
"""

        for model_id in MODELS:
            try:
                response = client.chat_completion(
                    model=model_id,
                    messages=[{"role": "system", "content": system_instructions}, {"role": "user", "content": query}],
                    max_tokens=400,
                    temperature=0.7 # Aumentata per dare più naturalezza
                )
                return jsonify({"success": True, "reply": response.choices[0].message.content})
            except Exception as e:
                continue

        return jsonify({"success": False, "reply": "Scusami, ho un piccolo calo di zuccheri! Riprova tra un secondo. 🍬"})

    except Exception as e:
        return jsonify({"success": False, "reply": "Ops! Qualcosa è andato storto nei miei circuiti. 🤖"})

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000)
