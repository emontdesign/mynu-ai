from flask import Flask, request, jsonify
from flask_cors import CORS
import g4f

app = Flask(__name__)
CORS(app)

@app.route('/ask', methods=['POST'])
def chat():
    try:
        data = request.json
        menu_data = data.get("menu", "")
        query = data.get("query", "")
        nome_rist = data.get("nome", "il ristorante")

        prompt = f"Sei l'assistente di {nome_rist}. Rispondi in italiano basandoti SOLO su questo menu JSON: {menu_data}. Domanda: {query}"

        response = g4f.ChatCompletion.create(
            model=g4f.models.gpt_4,
            messages=[{"role": "user", "content": prompt}],
        )
        return jsonify({"success": True, "reply": response})
    except Exception as e:
        return jsonify({"success": false, "error": str(e)})

if __name__ == "__main__":
    app.run()
