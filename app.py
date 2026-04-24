from flask import Flask, request, jsonify
from flask_cors import CORS
import g4f
import sys

app = Flask(__name__)
CORS(app)

@app.route('/ask', methods=['POST'])
def chat():
    try:
        # Recupero sicuro dei dati
        data = request.get_json(force=True)
        
        query = data.get("query", "")
        nome = data.get("nome", "Ristorante")
        menu = data.get("menu", "Dati non disponibili")
        opening = data.get("opening", "Dati non disponibili")
        services = data.get("services", "Dati non disponibili")

        if not query:
            return jsonify({"success": False, "error": "Domanda vuota"})

        # Prompt di sistema su una riga per evitare errori di indentazione
        system_instruction = f"Sei Maya, assistente di {nome}. Rispondi in italiano. Menu: {menu}. Orari: {opening}. Servizi: {services}. Se chiedono Wi-Fi dai la password se presente. Sii cordiale."

        # Chiamata AI senza specificare provider (auto-select)
        response = g4f.ChatCompletion.create(
            model=g4f.models.gpt_4,
            messages=[
                {"role": "system", "content": system_instruction},
                {"role": "user", "content": query}
            ],
        )

        return jsonify({
            "success": True, 
            "reply": str(response)
        })

    except Exception as e:
        # Stampa l'errore nei log di Render
        print(f"CRASH: {str(e)}", file=sys.stderr)
        return jsonify({
            "success": False, 
            "error": str(e)
        })

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000)
