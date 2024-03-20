from flask import Flask, request, jsonify
from model import train_model

app = Flask(__name__)

# Train the model and store it as a global variable
trained_model, label_encoder = train_model()

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        text = request.json.get('text')
        
        if text is None:
            return jsonify({'error': 'Text data is missing'}), 400

        # Preprocess the text (e.g., vectorize)
        # For simplicity, let's assume the vectorization process is the same as during training
        vectorized_text = trained_model.vectorizer.transform([text]).toarray()

        # Make prediction
        prediction = trained_model.predict(vectorized_text)

        # Decode the predicted label
        predicted_label = label_encoder.inverse_transform(prediction)[0]

        return jsonify({'predicted_label': predicted_label})
    else:
        return jsonify({'error': 'Method Not Allowed'}), 405

if __name__ == '__main__':
    app.run(debug=True)

