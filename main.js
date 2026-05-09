// ================= IMAGE PREVIEW =================
function previewImage(event) {
    const reader = new FileReader();

    reader.onload = function () {
        const img = document.getElementById("preview");
        img.src = reader.result;
    };

    reader.readAsDataURL(event.target.files[0]);
}


// ================= PROGRESS BAR =================
window.onload = function () {

    const bar = document.getElementById("progressBar");

    // Get confidence from HTML (Flask value)
    let confidence = parseFloat(document.getElementById("confidence")?.innerText) || 75;

    let width = 0;
    let target = confidence;  // ✅ FIXED

    let interval = setInterval(() => {
        if (width >= target) {
            clearInterval(interval);
        } else {
            width++;
            bar.style.width = width + "%";
            bar.innerText = width + "%";
        }
    }, 20);
};

new Chart(ctx, {
    type: 'pie',
    data: {
        labels: ['Clean', 'Stego'],
        datasets: [{
            data: [clean, stego],
            backgroundColor: ['green', 'red']
        }]
    },
    options: {
        responsive: true,
        maintainAspectRatio: false
    }
});