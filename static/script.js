/* script.js */
document.getElementById('submit-button').addEventListener('click', function () {
    const fileInput = document.getElementById('file-upload');
    const selectedModel = document.getElementById('model-select').value;

    if (fileInput.files.length === 0) {
        alert('Please upload an image.');
        return;
    }

    const formData = new FormData();
    formData.append('file', fileInput.files[0]);
    formData.append('model', selectedModel);

    // Fetch request to backend for processing
    fetch('/process', {
        method: 'POST',
        body: formData,
    })
        .then(response => {
            if (!response.ok) {
                throw new Error('Failed to process the image.');
            }
            return response.json();
        })
        .then(data => {
            // Show the results section
            document.getElementById('results').style.display = 'block';

            // Set object details]
            document.getElementById('class-ids').textContent = data.class_ids ? data.class_ids.join(', ') : 'N/A';
            document.getElementById('class-names').textContent = data.class_names ? data.class_names.join(', ') : 'N/A';
            document.getElementById('mask-areas').textContent = data.mask_areas ? data.mask_areas.join(', ') : 'N/A';

            // Display the processed image
            if (data.image_url) {
                document.getElementById('generated-image').src = data.image_url;
                document.getElementById('generated-image').alt = 'Processed image from AI';
            } else {
                document.getElementById('generated-image').alt = 'No image available.';
            }
        })
        .catch(error => {
            console.error('Error:', error);
            alert('There was an issue processing your request. Please try again.');
        });
});
