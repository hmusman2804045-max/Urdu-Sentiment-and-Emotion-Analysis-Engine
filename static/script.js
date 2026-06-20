document.addEventListener('DOMContentLoaded', () => {
    // Theme toggling
    const themeToggle = document.getElementById('themeToggle');
    const body = document.body;
    
    themeToggle.addEventListener('click', () => {
        body.classList.toggle('light-theme');
        // Update charts on theme change
        updateChartColors();
    });

    // Auto-resize textarea
    const textInput = document.getElementById('textInput');
    textInput.addEventListener('input', function() {
        this.style.height = 'auto';
        this.style.height = (this.scrollHeight) + 'px';
        
        // Auto-detect direction (simple heuristic)
        const urduRegex = /[\u0600-\u06FF]/;
        if (urduRegex.test(this.value)) {
            this.style.direction = 'rtl';
            this.classList.add('urdu-text');
        } else {
            this.style.direction = 'ltr';
            this.classList.remove('urdu-text');
        }
    });

    // Handle Analysis
    const analyzeBtn = document.getElementById('analyzeBtn');
    const resultsCard = document.getElementById('resultsCard');
    
    analyzeBtn.addEventListener('click', async () => {
        const text = textInput.value.trim();
        if (!text) return;
        
        // UI Loading state
        const originalBtnContent = analyzeBtn.innerHTML;
        analyzeBtn.innerHTML = `<span>Analyzing...</span><svg class="spinner" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" style="width:20px;height:20px;"><path stroke-linecap="round" stroke-linejoin="round" d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15"></path></svg>`;
        analyzeBtn.disabled = true;

        try {
            // Attempt to hit real backend
            let data;
            try {
                const response = await fetch('/analyze', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ text })
                });
                if (response.ok) {
                    data = await response.json();
                } else {
                    throw new Error('Backend not ready');
                }
            } catch (err) {
                console.log("Mocking response since backend isn't available:", err);
                // Mock fallback for UI demonstration
                await new Promise(r => setTimeout(r, 1200)); // Simulate delay
                data = generateMockResult(text);
            }
            
            displayResults(data);
            updateAnalyticsMock();
        } catch (error) {
            console.error('Analysis error:', error);
            alert('Error analyzing text. Please try again.');
        } finally {
            analyzeBtn.innerHTML = originalBtnContent;
            analyzeBtn.disabled = false;
        }
    });

    // Charts Initialization
    initCharts();
    
    // Start Live Feed Simulation
    startLiveFeed();
    
    // Init Keywords
    initKeywords();
});

// Mock Data Generator
function generateMockResult(text) {
    const isUrdu = /[\u0600-\u06FF]/.test(text);
    const lang = isUrdu ? 'Urdu' : 'Roman Urdu / English';
    
    // Randomize for demo
    const sentiments = ['Positive', 'Negative', 'Neutral'];
    const emotions = ['Joy', 'Anger', 'Fear', 'Sadness'];
    
    const sentiment = sentiments[Math.floor(Math.random() * sentiments.length)];
    const emotion = emotions[Math.floor(Math.random() * emotions.length)];
    const confS = (Math.random() * 20 + 75).toFixed(1); // 75-95%
    const confE = (Math.random() * 30 + 60).toFixed(1); // 60-90%

    // Attention map mock
    const words = text.split(/\s+/);
    const attentionWords = words.map(word => {
        return {
            word: word,
            weight: Math.random() // 0 to 1
        };
    });

    return {
        language: lang,
        sentiment: { label: sentiment, confidence: parseFloat(confS) },
        emotion: { label: emotion, confidence: parseFloat(confE) },
        attention: attentionWords
    };
}

// Display Results
function displayResults(data) {
    document.getElementById('detectedLang').textContent = data.language;
    document.getElementById('resultsCard').style.display = 'block';

    // Update Sentiment
    const sBox = document.getElementById('sentimentResult');
    const sConfBar = document.getElementById('sentimentConfidence');
    const sConfText = document.getElementById('sentimentConfText');
    
    let sEmoji = '😐', sColor = 'var(--neutral)';
    if (data.sentiment.label === 'Positive') { sEmoji = '😊'; sColor = 'var(--positive)'; }
    if (data.sentiment.label === 'Negative') { sEmoji = '😠'; sColor = 'var(--negative)'; }
    
    sBox.innerHTML = `<span class="emoji">${sEmoji}</span><span class="label" style="color: ${sColor}">${data.sentiment.label}</span>`;
    sConfBar.style.width = '0%'; // Reset for animation
    sConfBar.style.backgroundColor = sColor;
    setTimeout(() => {
        sConfBar.style.width = `${data.sentiment.confidence}%`;
    }, 100);
    sConfText.textContent = `${data.sentiment.confidence}% Confidence`;

    // Update Emotion
    const eBox = document.getElementById('emotionResult');
    const eConfBar = document.getElementById('emotionConfidence');
    const eConfText = document.getElementById('emotionConfText');
    
    let eEmoji = '🤔', eColor = 'var(--primary)';
    if (data.emotion.label === 'Joy') { eEmoji = '😄'; eColor = 'var(--joy)'; }
    if (data.emotion.label === 'Anger') { eEmoji = '😡'; eColor = 'var(--anger)'; }
    if (data.emotion.label === 'Fear') { eEmoji = '😨'; eColor = 'var(--fear)'; }
    if (data.emotion.label === 'Sadness') { eEmoji = '😢'; eColor = 'var(--sadness)'; }

    eBox.innerHTML = `<span class="emoji">${eEmoji}</span><span class="label" style="color: ${eColor}">${data.emotion.label}</span>`;
    eConfBar.style.width = '0%'; // Reset for animation
    eConfBar.style.backgroundColor = eColor;
    setTimeout(() => {
        eConfBar.style.width = `${data.emotion.confidence}%`;
    }, 100);
    eConfText.textContent = `${data.emotion.confidence}% Confidence`;

    // Render Attention Map
    const attContainer = document.getElementById('attentionResult');
    attContainer.innerHTML = '';
    
    const isRtl = /[\u0600-\u06FF]/.test(data.attention.map(w=>w.word).join(' '));
    attContainer.style.direction = isRtl ? 'rtl' : 'ltr';

    data.attention.forEach(item => {
        const span = document.createElement('span');
        span.textContent = item.word + ' ';
        span.className = 'attention-word';
        
        span.style.backgroundColor = `rgba(99, 102, 241, ${item.weight * 0.5})`;
        span.title = `Attention Weight: ${(item.weight).toFixed(2)}`;
        
        attContainer.appendChild(span);
    });
    
    // Scroll to results
    setTimeout(() => {
        document.getElementById('resultsCard').scrollIntoView({ behavior: 'smooth', block: 'nearest' });
    }, 100);
}

// Charts
let sentimentChart, emotionChart;

function initCharts() {
    const textColor = getComputedStyle(document.body).getPropertyValue('--text-primary').trim() || '#f9fafb';
    
    // Sentiment Donut
    const ctxS = document.getElementById('sentimentChart').getContext('2d');
    sentimentChart = new Chart(ctxS, {
        type: 'doughnut',
        data: {
            labels: ['Positive', 'Negative', 'Neutral'],
            datasets: [{
                data: [45, 30, 25], // initial mock data
                backgroundColor: ['#10b981', '#ef4444', '#6b7280'],
                borderWidth: 0,
                hoverOffset: 4
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: { position: 'bottom', labels: { color: textColor } },
                title: { display: true, text: 'Sentiment Distribution', color: textColor }
            },
            cutout: '70%'
        }
    });

    // Emotion Bar
    const ctxE = document.getElementById('emotionChart').getContext('2d');
    emotionChart = new Chart(ctxE, {
        type: 'bar',
        data: {
            labels: ['Joy', 'Anger', 'Fear', 'Sadness'],
            datasets: [{
                label: 'Emotion Count',
                data: [40, 25, 15, 20], // initial mock data
                backgroundColor: ['#f59e0b', '#ef4444', '#8b5cf6', '#3b82f6'],
                borderRadius: 4
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: { display: false },
                title: { display: true, text: 'Emotion Frequency', color: textColor }
            },
            scales: {
                y: { beginAtZero: true, grid: { color: 'rgba(156, 163, 175, 0.1)' }, ticks: { color: textColor } },
                x: { grid: { display: false }, ticks: { color: textColor } }
            }
        }
    });
}

function updateChartColors() {
    const textColor = getComputedStyle(document.body).getPropertyValue('--text-primary').trim() || '#111827';
    if (sentimentChart) {
        sentimentChart.options.plugins.legend.labels.color = textColor;
        sentimentChart.options.plugins.title.color = textColor;
        sentimentChart.update();
    }
    if (emotionChart) {
        emotionChart.options.plugins.title.color = textColor;
        emotionChart.options.scales.x.ticks.color = textColor;
        emotionChart.options.scales.y.ticks.color = textColor;
        emotionChart.update();
    }
}

function updateAnalyticsMock() {
    if (sentimentChart && emotionChart) {
        // Slightly jitter data for demo
        const sData = sentimentChart.data.datasets[0].data;
        sData[0] += Math.floor(Math.random() * 3);
        sData[1] += Math.floor(Math.random() * 3);
        sData[2] += Math.floor(Math.random() * 3);
        sentimentChart.update();

        const eData = emotionChart.data.datasets[0].data;
        eData[Math.floor(Math.random() * 4)] += 1;
        emotionChart.update();
    }
}

// Live Feed Simulation
const mockTweets = [
    { text: "آج کا دن بہت اچھا گزر رہا ہے! 🌞", user: "@ahmed_pk", pos: true, neu: false, neg: false },
    { text: "traffic ne bohat tang kiya hua hai aaj 😡", user: "@sana_tweets", pos: false, neu: false, neg: true },
    { text: "Weather update: It might rain tomorrow in Lahore.", user: "@news_update", pos: false, neu: true, neg: false },
    { text: "مجھے ڈر ہے کہ کل کا امتحان کیسا ہوگا 😰", user: "@student_life", pos: false, neu: false, neg: true },
    { text: "ye movie bohat zbardast thi, highly recommended! 🎬", user: "@cinephile", pos: true, neu: false, neg: false }
];

function startLiveFeed() {
    const container = document.getElementById('liveFeed');
    
    // Initial tweets
    for (let i = 0; i < 3; i++) {
        addTweetToFeed(container, mockTweets[i]);
    }

    // Add new tweet periodically
    setInterval(() => {
        const tweet = mockTweets[Math.floor(Math.random() * mockTweets.length)];
        addTweetToFeed(container, tweet);
    }, 8000);
}

function addTweetToFeed(container, data) {
    const el = document.createElement('div');
    el.className = 'tweet';
    
    let tagsHtml = '';
    if (data.pos) tagsHtml += '<span class="tag pos">Positive</span>';
    if (data.neg) tagsHtml += '<span class="tag neg">Negative</span>';
    if (data.neu) tagsHtml += '<span class="tag neu">Neutral</span>';
    
    el.innerHTML = `
        <div class="tweet-header">
            <span class="tweet-user">${data.user}</span>
            <span class="tweet-time">just now</span>
        </div>
        <div class="tweet-text ${/[\u0600-\u06FF]/.test(data.text) ? 'urdu-text' : ''}">
            ${data.text}
        </div>
        <div class="tweet-tags">
            ${tagsHtml}
        </div>
    `;
    
    container.prepend(el);
    if (container.children.length > 10) {
        container.removeChild(container.lastChild);
    }
}

function initKeywords() {
    const keywords = ['اچھا (Good)', 'برا (Bad)', 'خوش (Happy)', 'traffic', 'movie', 'امتحان', 'khushi', 'zbardast', 'غصہ', 'barish'];
    const container = document.getElementById('trendingKeywords');
    
    keywords.forEach(kw => {
        const span = document.createElement('span');
        span.className = 'keyword';
        span.textContent = kw;
        span.onclick = () => {
            document.getElementById('textInput').value = kw.split(' ')[0];
            document.getElementById('analyzeBtn').click();
        };
        container.appendChild(span);
    });
}
