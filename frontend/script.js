const submitBtn = document.getElementById('submitBtn');
const queryInput = document.getElementById('query');
const output = document.getElementById('output');

// Change this to your backend endpoint
const API_URL = 'http://localhost:8000/query';  

submitBtn.addEventListener('click', async () => {
    const query = queryInput.value.trim();
    if (!query) {
        alert('Please enter a query');
        return;
    }

    output.textContent = 'Loading...';

    try {
        const response = await fetch(API_URL, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ query })
        });

        if (!response.ok) throw new Error(`Server error: ${response.status}`);

        const data = await response.json();
        output.textContent = JSON.stringify(data, null, 2);

    } catch (err) {
        output.textContent = `Error: ${err.message}`;
    }
});
