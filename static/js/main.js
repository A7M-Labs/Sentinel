// Main JavaScript file for Sentinel application

document.addEventListener('DOMContentLoaded', function() {
    // Initialize tooltips
    const tooltipTriggerList = [].slice.call(document.querySelectorAll('[data-bs-toggle="tooltip"]'));
    tooltipTriggerList.map(function (tooltipTriggerEl) {
        return new bootstrap.Tooltip(tooltipTriggerEl);
    });

    // Initialize real-time data updaters
    setupEventListeners();
    initializeCameraFeeds();
    initializeCharts();
    
    // Demo: Simulate new alerts (for presentation purposes)
    if (document.querySelector('.alerts-container')) {
        simulateNewAlerts();
    }
});

// Set up event listeners
function setupEventListeners() {
    // Event dismissal
    document.querySelectorAll('.dismiss-alert').forEach(button => {
        button.addEventListener('click', function(e) {
            e.preventDefault();
            const alertItem = this.closest('.alert-item');
            alertItem.style.opacity = '0';
            setTimeout(() => {
                alertItem.remove();
            }, 300);
        });
    });

    // Camera feed selection
    document.querySelectorAll('.camera-select').forEach(select => {
        select.addEventListener('change', function() {
            const cameraId = this.value;
            const feedContainer = this.closest('.camera-card').querySelector('.camera-feed');
            feedContainer.querySelector('img').src = `/static/img/demo/camera-${cameraId}.jpg`;
            feedContainer.querySelector('.camera-name').textContent = this.options[this.selectedIndex].text;
        });
    });
}

// Initialize camera feeds
function initializeCameraFeeds() {
    const cameraFeeds = document.querySelectorAll('.camera-feed-live');
    
    if (cameraFeeds.length === 0) return;
    
    // For demo purposes, we'll just rotate some static images
    // In a real application, this would connect to live video streams
    cameraFeeds.forEach((feed, index) => {
        let counter = 1;
        setInterval(() => {
            // In a real app, this would be replaced with actual video feed logic
            counter = counter >= 3 ? 1 : counter + 1;
            const timestamp = new Date().toLocaleTimeString();
            feed.querySelector('.timestamp').textContent = timestamp;
        }, 2000 + (index * 500)); // Stagger the updates
    });
}

// Initialize dashboard charts
function initializeCharts() {
    const eventChartElement = document.getElementById('eventChart');
    
    if (!eventChartElement) return;
    
    // Sample data for the events chart
    const eventChart = new Chart(eventChartElement, {
        type: 'line',
        data: {
            labels: ['00:00', '04:00', '08:00', '12:00', '16:00', '20:00'],
            datasets: [{
                label: 'Security Events',
                data: [2, 4, 12, 8, 15, 5],
                borderColor: '#0d6efd',
                backgroundColor: 'rgba(13, 110, 253, 0.1)',
                tension: 0.3,
                fill: true
            }]
        },
        options: {
            plugins: {
                legend: {
                    display: false
                }
            },
            scales: {
                y: {
                    beginAtZero: true,
                    grid: {
                        color: 'rgba(255, 255, 255, 0.1)'
                    },
                    ticks: {
                        color: '#e9ecef'
                    }
                },
                x: {
                    grid: {
                        color: 'rgba(255, 255, 255, 0.1)'
                    },
                    ticks: {
                        color: '#e9ecef'
                    }
                }
            }
        }
    });
    
    // Add more charts as needed
}

// Simulate new alerts coming in (for demo purposes)
function simulateNewAlerts() {
    const alertsContainer = document.querySelector('.alerts-container');
    if (!alertsContainer) return;
    
    const alertTypes = [
        { type: 'Unauthorized Entry', priority: 'high-priority', icon: 'bi-person-x-fill' },
        { type: 'Unattended Object', priority: 'medium-priority', icon: 'bi-box-fill' },
        { type: 'Suspicious Behavior', priority: 'high-priority', icon: 'bi-exclamation-diamond-fill' },
        { type: 'Motion Detected', priority: 'low-priority', icon: 'bi-camera-video-fill' }
    ];
    
    const locations = ['Front Door', 'Lobby', 'Parking Lot', 'Back Entrance', 'Storage Room'];
    
    // Every 20-30 seconds, add a new alert
    setInterval(() => {
        const randomAlert = alertTypes[Math.floor(Math.random() * alertTypes.length)];
        const randomLocation = locations[Math.floor(Math.random() * locations.length)];
        const confidence = (Math.random() * 0.3 + 0.7).toFixed(2); // 0.70-0.99
        const timestamp = new Date().toLocaleTimeString();
        
        const alertHTML = `
            <div class="alert-item ${randomAlert.priority}" style="opacity: 0;">
                <div class="d-flex justify-content-between">
                    <div>
                        <h5 class="mb-1">
                            <i class="bi ${randomAlert.icon} me-2"></i>
                            ${randomAlert.type}
                        </h5>
                        <p class="mb-1">${randomLocation} - Confidence: ${confidence}</p>
                        <small class="text-muted">${timestamp}</small>
                    </div>
                    <div>
                        <button class="btn btn-sm btn-outline-secondary dismiss-alert">
                            <i class="bi bi-x-lg"></i>
                        </button>
                    </div>
                </div>
            </div>
        `;
        
        alertsContainer.insertAdjacentHTML('afterbegin', alertHTML);
        const newAlert = alertsContainer.querySelector('.alert-item');
        
        // Add event listener to the new dismiss button
        newAlert.querySelector('.dismiss-alert').addEventListener('click', function(e) {
            e.preventDefault();
            const alertItem = this.closest('.alert-item');
            alertItem.style.opacity = '0';
            setTimeout(() => {
                alertItem.remove();
            }, 300);
        });
        
        // Animate it in
        setTimeout(() => {
            newAlert.style.opacity = '1';
        }, 10);
        
        // Limit to 20 alerts in the container
        const alerts = alertsContainer.querySelectorAll('.alert-item');
        if (alerts.length > 20) {
            alerts[alerts.length - 1].remove();
        }
    }, Math.random() * 10000 + 20000); // 20-30 seconds
} 