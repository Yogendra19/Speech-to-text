function predict() {
    const form = document.getElementById('upload-form');
    const resultDiv = document.getElementById('result');
    const fileInput = document.getElementById('audio-file');

    // Check if a file is selected
    if (!fileInput.files || fileInput.files.length === 0) {
        resultDiv.textContent = 'Please select a .wav file.';
        return;
    }

    const file = fileInput.files[0];
    const formData = new FormData();
    formData.append('file', file);

    fetch('http://localhost:3000/predict', {
        method: 'POST',
        body: formData
    })
    .then(response => response.json())
    .then(data => {
        if (data.error) {
            resultDiv.textContent = `Error: ${data.error}`;
        } else {
            // Display the predictions here
            resultDiv.textContent = `Predictions: ${JSON.stringify(data.predictions)}`;
        }
    })
    .catch(error => {
        resultDiv.textContent = `Error: ${error.message}`;
    });
}
