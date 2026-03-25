const uploadInput = document.getElementById('audioUpload');
const processBtn = document.getElementById('processBtn');
const resultsArea = document.getElementById('results');
const loader = document.getElementById('loader');

// Update filename display when a file is selected
uploadInput.onchange = () => {
    const fileName = uploadInput.files[0]?.name || "No file selected";
    document.getElementById('fileNameDisplay').innerText = fileName;
};

processBtn.onclick = async () => {
    if (!uploadInput.files[0]) return alert("Please select a file first!");

    const formData = new FormData();
    formData.append("file", uploadInput.files[0]);
    // Always enable diarization now that it's fast
    formData.append("enable_diarization", "true");

    loader.classList.remove('hidden');
    resultsArea.classList.add('hidden');

    try {
        const response = await fetch('/transcribe', { method: 'POST', body: formData });
        const data = await response.json();

        document.getElementById('diarizedView').innerText = data.diarized_text;
        document.getElementById('redactedView').innerText = data.redacted_text;
        document.getElementById('intelView').innerHTML = `
            <p><strong>Summary:</strong> ${data.analysis.Summary}</p>
            <p><strong>Sentiment:</strong> ${data.analysis.Sentiment}</p>
            <p><strong>Agent Actions:</strong></p>
            <ul>${(data.analysis.AgentActions || []).map(a => `<li>${a}</li>`).join('')}</ul>
            <p><strong>Next Steps:</strong></p>
            <ul>${(data.analysis.ActionItems || []).map(a => `<li>${a}</li>`).join('')}</ul>
        `;

        loader.classList.add('hidden');
        resultsArea.classList.remove('hidden');
    } catch (err) {
        alert("Extraction failed!");
        loader.classList.add('hidden');
    }
};
