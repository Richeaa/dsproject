document.addEventListener('DOMContentLoaded', function () {
    const form = document.getElementById('scheduleForm');
    const status = document.getElementById('status');
    const status2 = document.getElementById('status2');
    const status3 = document.getElementById('status3');
    const status4 = document.getElementById('status4');
    const status5 = document.getElementById('status5');
    const status6 = document.getElementById('status6');
    const status7 = document.getElementById('status7');
    const chart1 = document.getElementById('chart1');
    const chart2 = document.getElementById('chart2');
    
    form.addEventListener('submit', function (e) {
        e.preventDefault();

        const activity = document.getElementById('activity').value;
        const day = document.getElementById('dayOfWeek').value;
        

        fetch('/predict-schedule/', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                type_name: activity,
                day: day,
              
            }),
        })
        .then(response => response.json())
        .then(data => {
            if (data.predicted_hour !== undefined && data.engagement_score !== undefined && data.all_hour_scores) {
                
                const hours = [...Array(24).keys()];
                const bestHour = data.predicted_hour;
                const bestScore = data.engagement_score;
 
        
                status.textContent = `📌 Recommended Time: ${bestHour}:00 — Engagement Score: ${bestScore.toFixed(2)}`;
                status2.textContent = `🧠 Suggestion: Schedule the ${activity} on ${day} at ${bestHour}:00 for optimal engagement.`;
                status3.textContent = `Another Recommended Time`;


                if (Array.isArray(data.top3_score)) {
                    if (data.top3_score.length > 0) {
                        status4.textContent = `⏰ ${data.top3_score[0].hour}:00 — Engagement Score: ${data.top3_score[0].score.toFixed(2)}`;
                    } else {
                        status4.textContent = '';
                    }

                    if (data.top3_score.length > 0) {
                        status5.textContent = `⏰ ${data.top3_score[1].hour}:00 — Engagement Score: ${data.top3_score[1].score.toFixed(2)}`;
                    } else {
                        status5.textContent = '';
                    }

                    if (data.top3_score.length > 0) {
                        status6.textContent = `⏰ ${data.top3_score[2].hour}:00 — Engagement Score: ${data.top3_score[2].score.toFixed(2)}`;
                    } else {
                        status6.textContent = '';
                    }

                    if (data.top3_score.length > 0) {
                        status7.textContent = `⏰ ${data.top3_score[3].hour}:00 — Engagement Score: ${data.top3_score[3].score.toFixed(2)}`;
                    } else {
                        status7.textContent = '';
                    }
                } else {
                    status4.textContent = '';
                    status5.textContent = '';
                    status6.textContent = '';
                    status7.textContent = '';
                }


                const existingChart1 = Chart.getChart(chart1);
                if (existingChart1) existingChart1.destroy();

                const existingChart2 = Chart.getChart(chart2);
                if (existingChart2) existingChart2.destroy();

                new Chart(chart1, {
                    type: 'line',
                    data: {
                        labels: hours.map(h => `${h}:00`),
                        datasets: [{
                            label: 'Engagement Score',
                            data: data.all_hour_scores,
                            borderColor: 'rgba(54, 162, 235, 1)',
                            backgroundColor: 'rgba(54, 162, 235, 0.2)',
                            tension: 0.3,
                            fill: true,
                        }]
                    },
                    options: {
                        responsive: true,
                        scales: {
                            y: {
                                beginAtZero: true,
                                max: 1,
                                title: { display: true, text: 'Score (0 - 1)' }
                            },
                            x: {
                                title: { display: true, text: 'Hour of Day' }
                            }
                        }
                    }
                });

                new Chart(chart2, {
                    type: 'bar',
                    data: {
                        labels: hours.map(h => `${h}:00`),
                        datasets: [{
                            label: 'Engagement Score',
                            data: data.all_hour_scores,
                            backgroundColor: 'rgba(75, 192, 192, 0.6)',
                            borderColor: 'rgba(75, 192, 192, 1)',
                            borderWidth: 1
                        }]
                    },
                    options: {
                        responsive: true,
                        scales: {
                            y: {
                                beginAtZero: true,
                                max: 1
                            }
                        }
                    }
                });
            } else {
                status.textContent = "⚠️ Error: Invalid response from server.";
                status2.textContent = "";
            }
        })
        .catch(error => {
            status.textContent = "❌ Error: " + error.message;
        });
    });
});
