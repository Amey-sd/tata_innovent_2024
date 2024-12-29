document.getElementById("video-button").addEventListener("click", async function () {
    // Select elements
    const fileInput = document.getElementById("file-upload");
    const modelSelect = document.getElementById("model-select");
    const aiResponse = document.getElementById("ai-response");

    // Get the selected file and model
    const file = fileInput.files[0];
    const model = modelSelect.value;

    // Validate file input
    if (!file) {
        alert("Please select a video file before submitting.");
        return;
    }

    // Prepare the form data
    const formData = new FormData();
    formData.append("video", file); // Ensure the key matches Flask backend
    formData.append("model", model);

    try {
        // Make the request to the backend
        const response = await fetch("/process-video", {
            method: "POST",
            body: formData,
        });

        // Handle response
        if (response.ok) {
            const result = await response.json();
            console.log("Server Response:", result);

            if (result.processed_video_url) {

                // Provide a link for the user to download the processed video
                aiResponse.innerHTML = `
                    <p>Video processed successfully!</p>
                    <p><a href="${result.processed_video_url}" target="_blank">Click here to download the processed video.</a></p>
                `;
            } else {
                // If processed video URL is not present
                aiResponse.innerText = "Error: Processed video URL not received from the server.";
            }
        } else {
            // Handle server errors
            const errorData = await response.json();
            aiResponse.innerText = `Error: ${errorData.error || "An unexpected error occurred."}`;
        }
    } catch (error) {
        // Handle client-side errors
        console.error("An error occurred:", error);
        aiResponse.innerText = "An error occurred while processing the video. Please try again.";
    }
});

document.getElementById("upload-video-button").addEventListener("click", async function () {
    const videoInput = document.getElementById("video-upload");
    const videoContainer = document.getElementById("video-container");
    const videoPlayer = document.getElementById("uploaded-video");
    const videoSource = document.getElementById("uploaded-video-source");

    const file = videoInput.files[0];
    if (!file) {
        alert("Please select a video file to upload.");
        return;
    }

    const formData = new FormData();
    formData.append("video", file);

    try {
        const response = await fetch("/upload-video", {
            method: "POST",
            body: formData,
        });

        if (response.ok) {
            const result = await response.json();
            videoSource.src = result.video_url; // Set video source URL
            videoPlayer.load();
            videoContainer.style.display = "block"; // Show video container
        } else {
            alert("Error uploading video. Please try again.");
        }
    } catch (error) {
        console.error("An error occurred:", error);
        alert("An error occurred while uploading the video.");
    }
});

