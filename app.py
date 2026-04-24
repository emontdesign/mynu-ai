from flask import Flask, request, jsonify
from flask_cors import CORS
import g4f

app = Flask(__name__)
CORS(app)

@app.route('/ask', methods=['POST'])
def chat():
    try:
        data = request.json
        hours_data = data.get("hours", "")
        menu_data = data.get("menu", "")
        query = data.get("query", "")
        nome_rist = data.get("nome", "il ristorante")
        giorno_oggi = data.get("giorno_settimana", "oggi")

        prompt = f"Il tuo nome è Maya e sei l'assistente di {nome_rist}. Oggi è {giorno_oggi}. Rispondi in italiano basandoti SOLO su questo menu JSON: {menu_data}. Se ti chiedono per gli orari attieni ESCLUSIVAMENTE al JSON: {hours_data}. REGOLE PER GLI ORARI: L'array 'schedule' segue rigorosamente questo ordine: Lunedì, Martedì, Mercoledì, Giovedì, Venerdì, Sabato e Domenica. Se l'utente chiede se siete aperti oggi, guarda il campo 'is_open' nel JSON. Se 'is_closed' è "1", dì chiaramente che quel giorno il locale è CHIUSO tutto il giorno. Domanda: {query}"

        response = g4f.ChatCompletion.create(
            model=g4f.models.gpt_4,
            messages=[{"role": "user", "content": prompt}],
        )
        return jsonify({"success": True, "reply": response})
    except Exception as e:
        return jsonify({"success": false, "error": str(e)})

if __name__ == "__main__":
    app.run()
