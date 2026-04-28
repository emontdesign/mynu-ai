import os
from flask import Flask, request, jsonify
from flask_cors import CORS
from huggingface_hub import InferenceClient
import sys

app = Flask(__name__)
CORS(app)

# Recuperiamo il token dalle variabili d'ambiente di Render
HF_TOKEN = os.getenv("HF_TOKEN")
client = InferenceClient(token=HF_TOKEN)

# Lista di modelli "sicuri" su HuggingFace (in ordine di priorità)
# Phi-3 è di Microsoft ed è estremamente stabile e veloce su HF
MODELS = [
    "microsoft/Phi-3-mini-4k-instruct",
    "Qwen/Qwen2.5-7B-Instruct",
    "mistralai/Mistral-7B-Instruct-v0.3"
]

@app.route('/', methods=['GET'])
def home():
    return "Maya AI (Resilient Mode) Online", 200

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
            return jsonify({"success": False, "error": "Configurazione mancante"})

        system_instructions = f"""
Sei Maya, l'assistente virtuale di {nome_rist}. Rispondi in italiano.
CONTESTO: Menu: {menu_data} | Orari: {hours_data} | Oggi è: {giorno_oggi}.

REGOLE RIGIDE PER IL MENU:
1. ISOLAMENTO DATI: Ogni oggetto nel JSON (delimitato da {{ }}) è indipendente. È SEVERAMENTE VIETATO associare la 'note' o gli 'allergeni' di un prodotto a un altro titolo. Se il prodotto ID 6 ha una nota, usa solo quella. Non guardare le note degli altri ID.
2. FILTRO NOTE: 
   - Se il campo 'note' contiene "Nota prodotto", ignoralo completamente (non scriverlo).
   - Se il campo 'note' è vuoto o null, non scrivere nulla.
   - Scrivi la nota solo se contiene informazioni reali sul piatto (es. "Aggiunta di Basilico").
3. ALLERGENI: Elenca gli allergeni solo se presenti nell'array 'allergeni'. Non inventare mai descrizioni come "Aggiunta di..." basandoti sugli allergeni; riporta solo i nomi degli allergeni se presenti.
4. CATEGORIE VUOTE: Se una categoria nel JSON non ha prodotti (lista []), NON DEVI assolutamente nominarla o includerla nella risposta.

REGOLE RIGIDE PER GLI ORARI:
5. SELEZIONE GIORNO: Nel JSON orari, il campo 'day_index' indica quale elemento dell'array 'schedule' devi leggere. Se 'day_index' è 1, DEVI guardare solo schedule[1]. Non prendere mai orari di altri giorni.
6. TURNI: Riporta esattamente le ore di 'apertura' e 'chiusura'. Se ci sono due turni (due oggetti nell'array), elencali entrambi. Non dire mai "aperto tutto il giorno" se c'è una pausa pomeridiana.
7. CHIUSURA: Se 'is_closed' è 1 per il giorno selezionato, di' che siamo chiusi.

ALTRO:
- Presenta i piatti del menu come elenco puntato.
- Sii amichevole, usa le emoji 🍕 e risposte non troppo lunghe.
- Se chiedono di ordinare, dì che per ora non è possibile farlo direttamente in chat.
"""

        # Tentiamo i modelli uno alla volta finché uno non risponde
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
                    temperature=0.7
                )
                final_reply = response.choices[0].message.content
                
                # Se arriviamo qui, il modello ha risposto!
                return jsonify({"success": True, "reply": final_reply})
            
            except Exception as e:
                last_error = str(e)
                print(f"Modello {model_id} fallito, provo il prossimo... Errore: {last_error}", file=sys.stderr)
                continue # Passa al prossimo modello nella lista

        # Se arriviamo qui, tutti i modelli hanno fallito
        return jsonify({
            "success": False, 
            "error": "Tutti i provider sono offline",
            "reply": "Scusami, i miei circuiti sono un po' sovraccarichi. Riprova tra un minuto! 🤖"
        })

    except Exception as e:
        print(f"ERRORE CRITICO: {str(e)}", file=sys.stderr)
        return jsonify({"success": False, "error": str(e)})

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000)
