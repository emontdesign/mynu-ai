from flask import Flask, request, jsonify
from flask_cors import CORS
import g4f
import sys

app = Flask(__name__)
CORS(app)

@app.route('/ask', methods=['POST'])
def chat():
    try:
        # Recupero dati
        data = request.get_json(force=True)
        
        query = data.get("query", "")
        nome = data.get("nome", "il ristorante")
        menu = data.get("menu", "")
        opening = data.get("opening", "")
        services = data.get("services", "")

        if not query:
            return jsonify({"success": False, "error": "Messaggio vuoto"})

        # Prompt di sistema
        system_instruction = f"Sei Maya, assistente di {nome}. Rispondi in italiano. Menu: {menu}. Orari: {opening}. Servizi: {services}."

        # CHIAMATA AI: Rimossa la specifica del provider per evitare crash
        # g4f sceglierà automaticamente il migliore disponibile al momento
        response = g4f.ChatCompletion.create(
            model=g4f.models.gpt_4,
            messages=[
                {"role": "system", "content": system_instruction},
                {"role": "user", "content": query}
            ],
        )

        # Risposta pulita
        return jsonify({
            "success": True, 
            "reply": str(response)
        })

    except Exception as e:
        # Questo stampa l'errore esatto nei log di Render se succede ancora qualcosa
        print(f"ERRORE RILEVATO: {str(e)}", file=sys.stderr)
        return jsonify({
            "success": False, 
            "error": str(e)
        })

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000)
