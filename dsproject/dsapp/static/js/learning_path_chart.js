document.addEventListener("DOMContentLoaded", function () {
    const statusBox = document.getElementById("statusBox");
    const rulesContainer = document.getElementById("rulesContainer");

    function showTab(tab) {
        document.querySelectorAll(".tab-content").forEach(div => div.style.display = "none");
        document.getElementById("tab-" + tab).style.display = "block";
    }
    window.showTab = showTab;

    // Handle existing student form
    document.getElementById("existingStudentForm").addEventListener("submit", function (e) {
        e.preventDefault();
        const student_id = e.target.student_id.value;
        const model = e.target.model.value;
        fetch('/get-learning-path-data/', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({ student_id: parseInt(student_id), model })
        })
        .then(res => res.json())
        .then(updateUI)
        .catch(err => statusBox.textContent = "Error: " + err.message);
    });

    // Handle manual input form
    document.getElementById("predictionForm").addEventListener("submit", function (e) {
        e.preventDefault();
        const formData = new FormData(e.target);
        const data = Object.fromEntries(formData.entries());

        // Convert numeric fields
        ["age", "time_Quiz", "time_IndividualAssignment", "time_GroupAssignment", "time_Forum"]
            .forEach(key => data[key] = parseFloat(data[key]));

        // Make sure to signal that this is a new prediction
        data.new_prediction = true;
        data.model = formData.get("model");

        fetch('/get-learning-path-data/', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify(data)
        })
        .then(res => res.json())
        .then(updateUI)
        .catch(err => statusBox.textContent = "Error: " + err.message);
    });

    function updateUI(data) {
        if (data.error) {
            statusBox.textContent = "Error: " + data.error;
            return;
        }

        const actual = data.actual_grade != null ? data.actual_grade.toFixed(2) : 'N/A';
        const predicted = data.predicted_grade != null ? data.predicted_grade.toFixed(2) : 'N/A';

        statusBox.textContent = `Actual Average Grade: ${actual} | Predicted Average Grade: ${predicted}`;

        updateChart("radarChart", createRadarConfig(data.activity_predictions || {}));
        updateChart("timeDistributionChart", createBarConfig(data.activity_times || {}, 'Time Spent (min)'));
        updateChart("gradeChart", createBarConfig(
            { Actual: data.actual_grade || 0, Predicted: data.predicted_grade || 0 },
            'Grade',
            [50, 100]
        ));

        // Rules
        rulesContainer.innerHTML = "";
        if (!data.association_rules || data.association_rules.length === 0) {
            rulesContainer.innerHTML = "<p>No recommended rules found.</p>";
            return;
        }

        data.association_rules.forEach(rule => {
            const div = document.createElement("div");
            div.className = "rule-item";
            div.innerHTML = `<strong>If completed:</strong> ${rule.antecedent} → <strong>Try:</strong> ${rule.consequent}<br>
                <small>Confidence: ${(rule.confidence * 100).toFixed(1)}%, 
                Support: ${(rule.support * 100).toFixed(1)}%, 
                Lift: ${rule.lift.toFixed(2)}</small>`;
            rulesContainer.appendChild(div);
        });
    }

    function updateChart(id, config) {
        const ctx = document.getElementById(id);
        if (window[id] instanceof Chart) {
            window[id].destroy();
        }
        window[id] = new Chart(ctx, config);
    }

    function createRadarConfig(data) {
        return {
            type: 'radar',
            data: {
                labels: Object.keys(data),
                datasets: [{
                    label: 'Predicted Grade',
                    data: Object.values(data),
                    backgroundColor: 'rgba(54, 162, 235, 0.2)',
                    borderColor: 'rgb(54, 162, 235)',
                }]
            },
            options: {
                scales: {
                    r: {
                        beginAtZero: true,
                        min: 50,
                        max: 100,
                        ticks: { stepSize: 10 }
                    }
                }
            }
        };
    }

    function createBarConfig(data, label, bounds = [0, null]) {
        return {
            type: 'bar',
            data: {
                labels: Object.keys(data),
                datasets: [{
                    label: label,
                    data: Object.values(data),
                    backgroundColor: 'rgba(75, 192, 192, 0.5)',
                    borderColor: 'rgba(75, 192, 192, 1)',
                    borderWidth: 1
                }]
            },
            options: {
                scales: {
                    y: {
                        beginAtZero: bounds[0] === 0,
                        min: bounds[0],
                        max: bounds[1]
                    }
                }
            }
        };
    }

    // Default to showing existing student tab
    showTab('existing');
});