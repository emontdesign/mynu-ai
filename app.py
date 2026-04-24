from flask import Flask, request, jsonify
from flask_cors import CORS
import g4f
import sys

app = Flask(__name__)
CORS(app)

@app.route('/ask', methods=['POST'])
def chat():
    try:
        # Recupero i dati inviati dal bridge PHP
        data = request.get_json(force=True)
        
        query = data.get("query", "")
        nome = data.get("nome", "il ristorante")
        menu = data.get("menu", "Non disponibile")
        opening = data.get("opening", "Non disponibile")
        services = data.get("services", "Non disponibile")

        if not query:
            return jsonify({"success": False, "error": "Messaggio vuoto"})

        # Costruiamo il prompt di sistema in modo ultra-stretto per evitare risposte generiche
        # Usiamo una variabile pulita per evitare errori di indentazione
        prompt_di_sistema = (
            f"Sei Maya, l'assistente virtuale ufficiale di {nome}. "
            f"ISTRUZIONI RIGIDE: Rispondi usando ESCLUSIVAMENTE i dati forniti qui sotto. "
            f"Se l'informazione non è presente, ammetti di non saperlo e non inventare nulla. "
            f"--- DATI DEL LOCALE --- "
            f"MENU: {menu} | "
            f"ORARI: {opening} | "
            f"SERVIZI: {services} | "
            f"--- FINE DATI --- "
            f"Rispondi in italiano in modo cordiale, usa i prezzi con € e cita ingredienti o orari precisi."
        )

        # Chiamata alla AI
        response = g4f.ChatCompletion.create(
            model=g4f.models.gpt_4,
            messages=[
                {"role": "system", "content": prompt_di_sistema},
                {"role": "user", "content": query}
            ],
        )

        # Verifichiamo che la risposta sia valida
        if not response:
            raise Exception("L'AI ha restituito una risposta vuota")

        return jsonify({
            "success": True, 
            "reply": str(response)
        })

    except Exception as e:
        # Log dell'errore visibile nella dashboard di Render
        print(f"ERRORE CRITICO MAYA: {str(e)}", file=sys.stderr)
        return jsonify({
            "success": False, 
            "error": str(e)
        })

# Punto di ingresso per Render
if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000)
