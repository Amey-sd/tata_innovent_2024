const liveVideo = document.getElementById('live-video');
const processedVideoFeed = document.getElementById('processed-video-feed');
const submitVideoButton = document.getElementById('submit-video-button');
const stopLiveButton = document.getElementById('stop-live-button');
const modelSelect = document.getElementById('model-select');
let mediaStream;
let intervalId;

submitVideoButton.addEventListener('click', async () => {
    const selectedModel = modelSelect.value;

    try {
        // Start the webcam feed
        mediaStream = await navigator.mediaDevices.getUserMedia({ video: { facingMode: "environment" } });
        liveVideo.srcObject = mediaStream;
        liveVideo.style.display = 'none';

        // Show the processed video feed
        processedVideoFeed.style.display = 'block';

        // Start sending frames to the server
        intervalId = setInterval(() => {
            sendFrame(selectedModel);
        }, 100); // 10 FPS
    } catch (error) {
        console.error("Error accessing webcam: ", error);
    }
});

stopLiveButton.addEventListener('click', () => {
    if (mediaStream) {
        mediaStream.getTracks().forEach(track => track.stop());
        liveVideo.srcObject = null;
        liveVideo.style.display = 'none';
    }
    if (intervalId) clearInterval(intervalId);
    processedVideoFeed.style.display = 'none';
    processedVideoFeed.src = '';
    var resultsDiv = document.getElementById('results');
    resultsDiv.style.display = 'block'; // Show the results div
});

async function sendFrame(model) {
    const canvas = document.createElement('canvas');
    const context = canvas.getContext('2d');

    canvas.width = liveVideo.videoWidth;
    canvas.height = liveVideo.videoHeight;
    context.drawImage(liveVideo, 0, 0);

    const dataURL = canvas.toDataURL('image/jpeg');
    try {
        const response = await fetch('/process_frame', {
            method: 'POST',
            body: JSON.stringify({ image: dataURL, model: model }),
            headers: { 'Content-Type': 'application/json' }
        });
        const data = await response.json();

        if (data.image) {
            processedVideoFeed.src = data.image; // Set the processed frame
        } else if (data.error) {
            console.error('Server Error:', data.error);
        }
    } catch (error) {
        console.error('Error sending frame:', error);
    }
}