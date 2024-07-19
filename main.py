from flask import Flask, request, jsonify, render_template
import requests

app = Flask(__name__)


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/generate', methods=['POST'])
def generate():
    data = request.json
    prompt = data['prompt']

    # Parameters for the poem generation
    max_length = data.get('max_length', 50)
    num_return_sequences = data.get('num_return_sequences', 1)
    temperature = data.get('temperature', 0.8)
    top_k = data.get('top_k', 50)
    top_p = data.get('top_p', 0.95)
    num_beams = data.get('num_beams', 5)
    repetition_penalty = data.get('repetition_penalty', 1.8)

    # Hugging Face model endpoint
    api_url = 'https://api-inference.huggingface.co/models/Anu99/gpt2-poem-generator-new'
    headers = {
        "Authorization": f"Bearer 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"
    }

    payload = {
        "inputs": prompt,
        "parameters": {
            "max_length": max_length,
            "temperature": temperature,
            "top_k": top_k,
            "top_p": top_p,
            "num_return_sequences": num_return_sequences,
            "num_beams": num_beams,
            "repetition_penalty": repetition_penalty,
            "no_repeat_ngram_size": 2,
            "early_stopping": True,
            "pad_token_id": None
        }
    }

    response = requests.post(api_url, headers=headers, json=payload)

    if response.status_code == 200:
        poem = response.json()[0]['generated_text']
    else:
        poem = f"Sorry, there was an error generating the poem. Error: {response.text}"

    return jsonify({"poem": poem})


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
