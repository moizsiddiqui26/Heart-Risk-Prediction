document.addEventListener('DOMContentLoaded', () => {
    const form = document.getElementById('predictionForm');
    const heightInput = document.getElementById('Height');
    const weightInput = document.getElementById('Weight');
    const bmiInput = document.getElementById('BMI');
    const modal = document.getElementById('resultModal');
    const closeModalBtn = document.querySelector('.close-modal');

    // Auto-calculate BMI
    function calculateBMI() {
        const heightCm = parseFloat(heightInput.value);
        const weightKg = parseFloat(weightInput.value);

        if (heightCm > 0 && weightKg > 0) {
            const heightM = heightCm / 100;
            const bmi = (weightKg / (heightM * heightM)).toFixed(2);
            bmiInput.value = bmi;
        } else {
            bmiInput.value = '';
        }
    }

    heightInput.addEventListener('input', calculateBMI);
    weightInput.addEventListener('input', calculateBMI);

    // Form Submission
    form.addEventListener('submit', async (e) => {
        e.preventDefault();

        const submitBtn = form.querySelector('button[type="submit"]');
        const originalBtnText = submitBtn.innerHTML;
        submitBtn.innerHTML = '<i class="fa-solid fa-spinner fa-spin"></i> Analyzing...';
        submitBtn.disabled = true;

        // Collect data
        const formData = new FormData(form);
        const data = {};

        // Convert FormData to JSON object
        formData.forEach((value, key) => {
            // Handle checkboxes
            if (form.querySelector(`input[name="${key}"][type="checkbox"]`)) {
                data[key] = 1; // Checkbox checked = 1
            } else {
                data[key] = value;
            }
        });

        // Handle unchecked checkboxes (they don't appear in FormData)
        const checkboxes = form.querySelectorAll('input[type="checkbox"]');
        checkboxes.forEach(cb => {
            if (!data.hasOwnProperty(cb.name)) {
                data[cb.name] = 0;
            }
        });

        console.log("Sending data:", data);

        try {
            const response = await fetch('/predict', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify(data)
            });

            if (!response.ok) {
                throw new Error('Network response was not ok');
            }

            const result = await response.json();
            showResult(result);

        } catch (error) {
            console.error('Error:', error);
            alert('An error occurred during analysis. Please try again.');
        } finally {
            submitBtn.innerHTML = originalBtnText;
            submitBtn.disabled = false;
        }
    });

    function showResult(result) {
        const riskLevel = document.getElementById('riskLevel');
        const riskIcon = document.getElementById('riskIcon');
        const probFill = document.getElementById('probFill');
        const probValue = document.getElementById('probValue'); // Wait, I didn't add this ID in HTML, let me check.
        // Ah, I added <span id="probValue">85%</span> in HTML. Good.

        const probability = result.probability;
        const percentage = (probability * 100).toFixed(1) + '%';

        if (probability > 0.5) {
            riskLevel.textContent = 'High Risk';
            riskLevel.className = 'risk-high';
            riskIcon.innerHTML = '<i class="fa-solid fa-heart-crack risk-high"></i>';
            probFill.style.background = 'linear-gradient(90deg, #F87171, #EF4444)'; // Red 400 -> 500
        } else {
            riskLevel.textContent = 'Low Risk';
            riskLevel.className = 'risk-low';
            riskIcon.innerHTML = '<i class="fa-solid fa-heart-circle-check risk-low"></i>';
            probFill.style.background = 'linear-gradient(90deg, #34D399, #10B981)'; // Emerald 400 -> 500
        }

        probFill.style.width = percentage;
        document.querySelector('.probability-meter span').textContent = percentage; // Use selector if ID fails

        modal.classList.add('visible');
    }

    // Modal closing
    function closeModal() {
        modal.classList.remove('visible');
    }

    closeModalBtn.addEventListener('click', closeModal);
    window.onclick = function (event) {
        if (event.target == modal) {
            closeModal();
        }
    }
    // Expose to window for button onclick
    window.closeModal = closeModal;
});
