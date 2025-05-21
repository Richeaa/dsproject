document.addEventListener('DOMContentLoaded', function () {
    const form = document.getElementById('gradeForm');
    const status = document.getElementById('status');
    const status2 = document.getElementById('status2');
    const chart1 = document.getElementById('chart1');
    const chart2 = document.getElementById('chart2');

    let myChart = null;
    let myChart2 = null;

    form.addEventListener('submit', function (e) {
        e.preventDefault();

        const formData = new FormData(form);
        const grades = {
            course1: formData.get('course1'),
            course2: formData.get('course2'),
            course3: formData.get('course3'),
            course4: formData.get('course4'),
            course5: formData.get('course5'),
            model_name: formData.get('model_name') 
        };

        fetch('/predict-grades/', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json', 
            },
            body: JSON.stringify(grades),
        })
        .then(response => response.json())
        .then(data => {
            if (data.prediction !== undefined && data.avg_grade !== undefined) {
                status.textContent = `Average Grade: ${data.avg_grade}`;
                status2.textContent = `Advice: ${data.advice}`;

                if (myChart) myChart.destroy();
                if (myChart2) myChart2.destroy();
        
                myChart = new Chart(chart1, {
                    type: "line",
                    data: {
                        labels: ["Underperform", "Good"],
                        datasets: [{
                            label: "Prediction",
                            data: data.probability,
                            backgroundColor: [
                                "rgba(173, 36, 36, 0.2)",
                                "rgba(49, 255, 76, 0.2)",
                            ],
                            borderColor: [
                                "rgb(255, 148, 148)",
                                "rgb(157, 255, 165)",
                            ],
                            borderWidth: 1,
                        }],
                    },
                    options: {
                        responsive: true,
                        scales: {
                            x: { grid: { color: "rgba(0, 198, 247, 0.2)" } },
                            y: { grid: { color: "rgba(0, 198, 247, 0.2)" }, beginAtZero: true },
                        },
                    },
                });


                myChart2 = new Chart(chart2, {
                    type: "bar",
                    data: {
                        labels: ["Underperform", "Good"],
                        datasets: [{
                            label: "Probability",
                            data: data.probability,
                            backgroundColor: [
                                "rgba(255, 99, 132, 0.6)",
                                "rgba(75, 192, 192, 0.6)"
                            ],
                            borderColor: [
                                "rgba(255, 99, 132, 1)",
                                "rgba(75, 192, 192, 1)"
                            ],
                            borderWidth: 1
                        }]
                    },
                    options: {
                        responsive: true
                    }
                });
            } else {
                status.textContent = "Error: Invalid response format.";
            }
        })
        .catch(error => {
            status.textContent = "Error: " + error.message;
        });
    });
});