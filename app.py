from flask import Flask, request, jsonify
from flask_cors import CORS
import g4f

app = Flask(__name__)
CORS(app)

@app.route('/ask', methods=['POST'])
def chat():
    try:
        data = request.json
        
        # Recupero dei dati inviati dal bridge PHP
        query = data.get("query", "")
        nome_rist = data.get("nome", "il ristorante")
        
        # Context Data
        menu_data = data.get("menu", "")
        opening_data = data.get("opening", "")
        services_data = data.get("services", "")

        # Costruzione di un Prompt di sistema avanzato
        system_instruction = f"""
        Sei Maya, l'assistente virtuale intelligente di {nome_rist}. 
        Il tuo obiettivo è aiutare i clienti usando SOLO le informazioni fornite qui sotto.
        
        DATI DISPONIBILI:
        1. MENU COMPLETO (Piatti, ingredienti, prezzi e allergeni): 
        {menu_data}
        
        2. ORARI DI APERTURA: 
        {opening_data}
        (Nota: nello 'schedule', l'indice 0 è Lunedì, 1 Martedì... 6 è Domenica. Controlla 'is_open' per lo stato attuale).
        
        3. SERVIZI DEL LOCALE (Wi-Fi, Delivery, ecc.): 
        {services_data}
        (Se chiedono la password del Wi-Fi, cercala nel campo 'options' dei servizi).

        LINEE GUIDA PER LE RISPOSTE:
        - Rispondi sempre in italiano in modo cordiale e professionale.
        - Se chiedono piatti senza un allergene, analizza attentamente la lista allergeni nel JSON.
        - Se chiedono i prezzi, specifica sempre il simbolo €.
        - Se un'informazione non è presente nei dati sopra, scusati gentilmente e suggerisci di contattare il personale telefonicamente.
        - Usa le emoji per rendere la chat amichevole 🍕🍷.
        """

        # Chiamata alla libreria g4f (GPT-4)
        # Nota: Usiamo il provider Blackbox o Bing se disponibili per maggiore velocità
        response = g4f.ChatCompletion.create(
            model=g4f.models.gpt_4,
            provider=g4f.Provider.Blackbox, # Spesso il più veloce e stabile
            messages=[
                {"role": "system", "content": system_instruction},
                {"role": "user", "content": query}
            ],
        )

        return jsonify({
            "success": True, 
            "reply": response
        })

    except Exception as e:
        # Stampiamo l'errore nei log di Render per il debug
        print(f"Errore Maya AI: {str(e)}")
        return jsonify({
            "success": False, 
            "error": str(e)
        })

if __name__ == "__main__":
    # In locale usiamo la porta 5000, su Render viene gestito da Gunicorn
    app.run(host='0.0.0.0', port=5000)
