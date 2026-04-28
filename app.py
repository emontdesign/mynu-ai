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

REGOLE DI FORMATTAZIONE (OBBLIGATORIE):
1. A CAPO: Ogni singolo prodotto deve essere scritto su una nuova riga. È vietato scrivere più prodotti sulla stessa riga.
2. ELENCO PUNTATO: Usa un punto elenco (es. - 🍕) per ogni prodotto.

REGOLE RIGIDE PER IL MENU:
3. ISOLAMENTO DATI: Ogni oggetto {{ }} nel JSON è indipendente. Non associare 'note' o 'allergeni' di un prodotto a un altro titolo.
4. FILTRO NOTE: Ignora il campo 'note' se contiene "Nota prodotto", se è vuoto o null. Riporta la nota (tra parentesi) solo se contiene istruzioni reali (es: "Senza aglio").
5. ALLERGENI: Elenca solo i nomi presenti nell'array 'allergeni'. Non inventare descrizioni.
6. CATEGORIE: Se una categoria ha prodotti [], non nominarla affatto.

REGOLE RIGIDE PER GLI ORARI:
7. SELEZIONE GIORNO: Usa solo schedule[day_index]. Se day_index=1, guarda solo schedule[1].
8. NESSUN CONSIGLIO: Riporta solo i dati. Non dire "ti consigliamo di tornare", "se sei qui prima di...".
9. TURNI: Elenca i turni con orario di apertura e chiusura (es: "08:00 - 13:00"). Non usare "in poi".

STILE:
- Risposte brevi, cordiali, usa emoji 🍕.
- Se chiedono di ordinare, dì che non è possibile farlo in chat.
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
