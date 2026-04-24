from flask import Flask, request, jsonify
from flask_cors import CORS
import g4f
import sys

app = Flask(__name__)
CORS(app)

@app.route('/ask', methods=['POST'])
def chat():
    try:
        data = request.get_json(force=True)
        
        # Recupero dati dal bridge PHP
        query = data.get("query", "")
        nome_rist = data.get("nome", "il ristorante")
        menu_data = data.get("menu", "Menu non disponibile")
        hours_data = data.get("hours", "Orari non disponibili")
        # Recuperiamo anche il nome del giorno che abbiamo aggiunto nel PHP
        giorno_oggi = data.get("giorno_settimana", "oggi") 

        if not query:
            return jsonify({"success": False, "error": "Domanda vuota"})

        # ISTRUZIONI RIGIDE PER L'AI
        system_prompt = f"""
        Sei Maya, l'assistente virtuale di {nome_rist}. Rispondi in italiano.
        Oggi è {giorno_oggi}.

        CONTESTO OBBLIGATORIO:
        1. MENU (JSON): {menu_data}
        2. ORARI (JSON): {hours_data}

        REGOLE DI RISPOSTA:
        - Per gli orari, l'array 'schedule' ha indici da 0 a 6: 0=Lunedì, 1=Martedì, 2=Mercoledì, 3=Giovedì, 4=Venerdì, 5=Sabato, 6=Domenica.
        - Se chiedono 'oggi', guarda il campo 'is_open' e commenta gli orari del giorno corrente.
        - Usa SOLO le informazioni fornite. Se non trovi un piatto o un orario, dì che non lo sai.
        - Sii cordiale e usa le emoji 🍕.
        """

        # Chiamata a g4f (lasciamo che scelga il provider migliore automaticamente)
        response = g4f.ChatCompletion.create(
            model=g4f.models.gpt_4,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": query}
            ],
        )

        # Gestione risposta: g4f a volte restituisce oggetti strani, lo forziamo a stringa
        final_reply = str(response)

        return jsonify({
            "success": True, 
            "reply": final_reply
        })

    except Exception as e:
        print(f"ERRORE PYTHON: {str(e)}", file=sys.stderr)
        return jsonify({
            "success": False, 
            "error": str(e)
        })

if __name__ == "__main__":
    # La porta 5000 è quella standard per Flask
    app.run(host='0.0.0.0', port=5000)
