document.getElementById('submit-video-button').addEventListener('click', function() {
    var model = document.getElementById('model-select').value;
    var videoFeed = document.getElementById('video-feed');
    videoFeed.src = '/video_feed?model=' + model; // Set the source of the image to the video feed URL
    videoFeed.style.display = 'block'; // Show the image element
    });
    document.getElementById('Stop-live').addEventListener('click', function() {
    var videoFeed = document.getElementById('video-feed');
    videoFeed.src = ''; // Stop the feed by clearing the source URL
    videoFeed.style.display = 'none'; // Optionally hide the image
});