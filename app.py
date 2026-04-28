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

REGOLE RIGIDE PER GLI ORARI (SISTEMA BINARIO):
1. SOLO OGGI: Se l'utente chiede degli orari senza specificare un giorno, DEVI rispondere usando solo il 'day_index' di oggi. Non elencare altri giorni della settimana a meno che non venga chiesto esplicitamente.
2. DIVIETO DI RAGGRUPPAMENTO: Non scrivere mai "da lunedì a venerdì" o simili. Leggi ogni giorno come un'entità separata.
3. DIVIETO DI CONSIGLIO: È categoricamente vietato scrivere "ti consigliamo di...", "ti aspettiamo", "torna più tardi". Devi solo riportare i dati.
4. FORMATO TURNI: Ogni turno deve stare su una riga separata. 
   Esempio:
   - Mattina: 08:00 - 13:00
   - Sera: 16:00 - 00:00
5. STATO: Se is_open è true, scrivi "In questo momento siamo aperti". Se false, "In questo momento siamo chiusi".

REGOLE RIGIDE PER IL MENU:
6. ISOLAMENTO: Associa 'note' e 'allergeni' solo al prodotto corrispondente.
7. FILTRO NOTE: Non scrivere "Nota prodotto". Se la nota è utile (es: "Piccante"), scrivila tra parentesi accanto al prezzo.
8. A CAPO: Ogni prodotto deve stare su una riga dedicata. Non affiancare mai due prodotti.
9. CATEGORIE: Salta le categorie senza prodotti.

STILE:
- Risposte asciutte, cordiali, elenchi puntati, usa emoji 🍕.
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
