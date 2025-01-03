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
                // Alert the user that the video has been processed successfully
                alert("Video processed successfully!");

                // Show the download link
                document.getElementById("results").style.display = 'block';
                aiResponse.innerHTML = `
                    <p>Video processed successfully!</p>
                    <p><a href="${result.processed_video_url}" target="_blank">Click here to download the processed video.</a></p>
                `;
            } else {
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
