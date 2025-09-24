from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
import sqlite3
import polars as pl
import json
import os
from datetime import datetime
from typing import List, Dict, Any
from statistical_analysis import RegressionDiscontinuityAnalyzer, analyze_supplement_effectiveness
from openai_insights import generate_health_insights, predict_biomarker_changes, analyze_biomarker_correlations

app = FastAPI(title="Health Data Analytics", version="1.0.0")

# Database initialization
def init_database():
    conn = sqlite3.connect('health_data.db')
    cursor = conn.cursor()
    
    # Biomarkers table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS biomarkers (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            value REAL NOT NULL,
            unit TEXT,
            reference_range_min REAL,
            reference_range_max REAL,
            test_date DATE NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    # Supplements table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS supplements (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            start_date DATE NOT NULL,
            dosage TEXT,
            expected_biomarkers TEXT,
            notes TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    # Health metrics table (from wearables/CSV)
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS health_metrics (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            metric_name TEXT NOT NULL,
            value REAL NOT NULL,
            unit TEXT,
            measurement_date DATE NOT NULL,
            source TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    conn.commit()
    conn.close()

@app.on_event("startup")
async def startup_event():
    init_database()

@app.get("/", response_class=HTMLResponse)
async def dashboard():
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Health Data Analytics</title>
        <meta charset="utf-8">
        <meta name="viewport" content="width=device-width, initial-scale=1">
        <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
        <style>
            body { font-family: -apple-system, BlinkMacSystemFont, sans-serif; margin: 0; padding: 20px; background: #f8f9fa; }
            .container { max-width: 1200px; margin: 0 auto; }
            .header { background: white; padding: 20px; border-radius: 8px; margin-bottom: 20px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
            .card { background: white; padding: 20px; border-radius: 8px; margin-bottom: 20px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
            .upload-zone { border: 2px dashed #ddd; padding: 40px; text-align: center; border-radius: 8px; cursor: pointer; }
            .upload-zone:hover { border-color: #007bff; background: #f8f9ff; }
            .btn { background: #007bff; color: white; border: none; padding: 12px 24px; border-radius: 6px; cursor: pointer; font-size: 14px; }
            .btn:hover { background: #0056b3; }
            .grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 20px; }
            .metric-card { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 20px; border-radius: 8px; }
            .metric-value { font-size: 2em; font-weight: bold; margin-bottom: 5px; }
            .metric-label { opacity: 0.9; font-size: 0.9em; }
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1>🧬 Health Data Analytics</h1>
                <p>Track biomarkers, analyze supplement effects, and predict health trends</p>
            </div>
            
            <div class="grid">
                <div class="card">
                    <h3>📊 Upload Health Data</h3>
                    <div class="upload-zone" onclick="document.getElementById('csvFile').click()">
                        <div>📁 Click to upload CSV file</div>
                        <small>Expected columns: date, metric_name, value, unit</small>
                    </div>
                    <input type="file" id="csvFile" accept=".csv" style="display: none;" onchange="uploadCSV(this)">
                </div>
                
                <div class="card">
                    <h3>🩸 Add Blood Test Results</h3>
                    <form onsubmit="addBiomarker(event)">
                        <input type="text" id="biomarkerName" placeholder="Biomarker name" required style="width: 100%; padding: 8px; margin: 5px 0; border: 1px solid #ddd; border-radius: 4px;">
                        <input type="number" id="biomarkerValue" placeholder="Value" step="0.01" required style="width: 100%; padding: 8px; margin: 5px 0; border: 1px solid #ddd; border-radius: 4px;">
                        <input type="text" id="biomarkerUnit" placeholder="Unit (e.g., mg/dL)" style="width: 100%; padding: 8px; margin: 5px 0; border: 1px solid #ddd; border-radius: 4px;">
                        <input type="date" id="testDate" required style="width: 100%; padding: 8px; margin: 5px 0; border: 1px solid #ddd; border-radius: 4px;">
                        <button type="submit" class="btn" style="width: 100%; margin-top: 10px;">Add Biomarker</button>
                    </form>
                </div>
                
                <div class="card">
                    <h3>💊 Track Supplements</h3>
                    <form onsubmit="addSupplement(event)">
                        <input type="text" id="supplementName" placeholder="Supplement name" required style="width: 100%; padding: 8px; margin: 5px 0; border: 1px solid #ddd; border-radius: 4px;">
                        <input type="date" id="supplementStartDate" required style="width: 100%; padding: 8px; margin: 5px 0; border: 1px solid #ddd; border-radius: 4px;">
                        <input type="text" id="supplementDosage" placeholder="Dosage (e.g., 500mg daily)" style="width: 100%; padding: 8px; margin: 5px 0; border: 1px solid #ddd; border-radius: 4px;">
                        <input type="text" id="expectedBiomarkers" placeholder="Target biomarkers" style="width: 100%; padding: 8px; margin: 5px 0; border: 1px solid #ddd; border-radius: 4px;">
                        <button type="submit" class="btn" style="width: 100%; margin-top: 10px;">Track Supplement</button>
                    </form>
                </div>
            </div>
            
            <div class="card">
                <h3>💊 Supplement Timeline</h3>
                <div id="supplementChart" style="height: 400px;"></div>
            </div>
            
            <div class="card">
                <h3>📈 Biomarker Trends</h3>
                <div id="biomarkerChart" style="height: 400px;"></div>
            </div>
            
            <div class="card">
                <h3>🔬 Regression Discontinuity Analysis</h3>
                <div style="margin-bottom: 20px;">
                    <label for="rddMetricSelect" style="display: block; margin-bottom: 5px; font-weight: bold;">Select Health Metric:</label>
                    <select id="rddMetricSelect" style="width: 100%; padding: 8px; border: 1px solid #ddd; border-radius: 4px; font-size: 14px;">
                        <option value="">Choose a metric...</option>
                        <option value="resting_heart_rate">Resting Heart Rate</option>
                        <option value="stress_level">Stress Level</option>
                        <option value="sleep_score">Sleep Score</option>
                        <option value="steps">Steps</option>
                        <option value="intensity_minutes">Intensity Minutes</option>
                    </select>
                    <button onclick="analyzeMetricRDD()" class="btn" style="margin-top: 10px;">Analyze Supplement Effect</button>
                </div>
                <div id="rddResults" style="min-height: 200px; padding: 20px; background: #f8f9fa; border-radius: 8px;">
                    <div style="text-align: center; color: #666;">
                        <p>Select a health metric above to analyze supplement effectiveness using Regression Discontinuity Design</p>
                        <small>This analysis measures whether there was a statistically significant change since supplementation started</small>
                    </div>
                </div>
            </div>
            
            <div class="grid">
                <div class="metric-card">
                    <div class="metric-value">🟢</div>
                    <div class="metric-label">Overall Health Score</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value" id="weeklyChange">--</div>
                    <div class="metric-label">Weekly Change</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value" id="nextPrediction">--</div>
                    <div class="metric-label">Next Test Prediction</div>
                </div>
            </div>
            
            <div class="card">
                <h3>🤖 AI Insights</h3>
                <div id="aiInsights" style="min-height: 200px; padding: 20px; background: #f8f9fa; border-radius: 8px;">
                    <div style="text-align: center; color: #666;">
                        <button onclick="generateInsights()" class="btn">Generate AI Analysis</button>
                        <p style="margin-top: 15px;">Get personalized insights about your health trends and supplement effectiveness</p>
                    </div>
                </div>
            </div>
        </div>
        
        <script>
            async function uploadCSV(input) {
                const file = input.files[0];
                if (!file) return;
                
                const formData = new FormData();
                formData.append('file', file);
                
                try {
                    const response = await fetch('/upload-csv', {
                        method: 'POST',
                        body: formData
                    });
                    const result = await response.json();
                    if (response.ok) {
                        alert('CSV uploaded successfully!');
                        loadDashboardData();
                    } else {
                        alert('Error: ' + result.detail);
                    }
                } catch (error) {
                    alert('Upload failed: ' + error.message);
                }
            }
            
            async function addBiomarker(event) {
                event.preventDefault();
                const data = {
                    name: document.getElementById('biomarkerName').value,
                    value: parseFloat(document.getElementById('biomarkerValue').value),
                    unit: document.getElementById('biomarkerUnit').value,
                    test_date: document.getElementById('testDate').value
                };
                
                try {
                    const response = await fetch('/biomarkers', {
                        method: 'POST',
                        headers: {'Content-Type': 'application/json'},
                        body: JSON.stringify(data)
                    });
                    
                    if (response.ok) {
                        alert('Biomarker added successfully!');
                        event.target.reset();
                        loadDashboardData();
                    } else {
                        const error = await response.json();
                        alert('Error: ' + error.detail);
                    }
                } catch (error) {
                    alert('Failed to add biomarker: ' + error.message);
                }
            }
            
            async function addSupplement(event) {
                event.preventDefault();
                const data = {
                    name: document.getElementById('supplementName').value,
                    start_date: document.getElementById('supplementStartDate').value,
                    dosage: document.getElementById('supplementDosage').value,
                    expected_biomarkers: document.getElementById('expectedBiomarkers').value
                };
                
                try {
                    const response = await fetch('/supplements', {
                        method: 'POST',
                        headers: {'Content-Type': 'application/json'},
                        body: JSON.stringify(data)
                    });
                    
                    if (response.ok) {
                        alert('Supplement added successfully!');
                        event.target.reset();
                        loadDashboardData();
                    } else {
                        const error = await response.json();
                        alert('Error: ' + error.detail);
                    }
                } catch (error) {
                    alert('Failed to add supplement: ' + error.message);
                }
            }
            
            async function analyzeMetricRDD() {
                const selectedMetric = document.getElementById('rddMetricSelect').value;
                if (!selectedMetric) {
                    alert('Please select a health metric first');
                    return;
                }
                
                const resultsDiv = document.getElementById('rddResults');
                resultsDiv.innerHTML = '<div style="text-align: center; color: #666;">Analyzing supplement effect on ' + selectedMetric + '...</div>';
                
                try {
                    const response = await fetch('/analyze-supplement-effect', {
                        method: 'POST',
                        headers: {'Content-Type': 'application/json'},
                        body: JSON.stringify({
                            'supplement_name': 'vitamin d',  // Default to vitamin d
                            'target_metrics': [selectedMetric]
                        })
                    });
                    
                    const result = await response.json();
                    
                    if (response.ok) {
                        displayRDDResults(result, selectedMetric);
                    } else {
                        resultsDiv.innerHTML = '<p style="color: red;">Analysis failed: ' + result.detail + '</p>';
                    }
                } catch (error) {
                    resultsDiv.innerHTML = '<p style="color: red;">Error: ' + error.message + '</p>';
                }
            }
            
            function displayRDDResults(result, metric) {
                const effectData = result.metric_effects[metric];
                const resultsDiv = document.getElementById('rddResults');
                
                if (effectData.error) {
                    resultsDiv.innerHTML = '<p style="color: red;">Analysis Error: ' + effectData.error + '</p>';
                    return;
                }
                
                const isSignificant = effectData.statistically_significant;
                const pValue = effectData.treatment_p_value;
                const effect = effectData.treatment_effect;
                const observations = effectData.n_observations;
                
                const significanceColor = isSignificant ? '#28a745' : '#dc3545';
                const significanceIcon = isSignificant ? '✅' : '❌';
                const effectDirection = effect > 0 ? 'increase' : 'decrease';
                
                let html = '<div style="border-left: 4px solid ' + significanceColor + '; padding-left: 15px;">';
                html += '<h4 style="margin-top: 0; color: ' + significanceColor + ';">' + significanceIcon + ' RDD Analysis Results</h4>';
                html += '<div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 15px; margin: 15px 0;">';
                html += '<div><strong>Effect Size:</strong><br>' + Math.abs(effect).toFixed(2) + ' point ' + effectDirection + '</div>';
                html += '<div><strong>P-value:</strong><br>' + (pValue ? pValue.toFixed(4) : 'N/A') + '</div>';
                html += '<div><strong>Significance:</strong><br>' + (isSignificant ? 'Significant' : 'Not Significant') + '</div>';
                html += '<div><strong>Sample Size:</strong><br>' + observations + ' observations</div>';
                html += '</div>';
                
                if (isSignificant) {
                    html += '<div style="background: #d4edda; border: 1px solid #c3e6cb; padding: 10px; border-radius: 4px; margin: 10px 0;">';
                    html += '<strong>🎯 Significant Result:</strong> The supplement had a statistically significant effect on ' + metric.replace('_', ' ') + '. ';
                    html += 'There was a ' + Math.abs(effect).toFixed(1) + ' point ' + effectDirection + ' after supplementation started.';
                    html += '</div>';
                } else {
                    html += '<div style="background: #f8d7da; border: 1px solid #f5c6cb; padding: 10px; border-radius: 4px; margin: 10px 0;">';
                    html += '<strong>📊 No Significant Effect:</strong> The analysis did not detect a statistically significant change in ' + metric.replace('_', ' ') + ' after supplementation (p > 0.05).';
                    html += '</div>';
                }
                
                html += '<small style="color: #666; display: block; margin-top: 10px;">Analysis used Regression Discontinuity Design with 30-day bandwidth around supplement start date.</small>';
                html += '</div>';
                
                resultsDiv.innerHTML = html;
            }
            
            async function generateInsights() {
                const button = event.target;
                const originalText = button.textContent;
                button.textContent = 'Analyzing...';
                button.disabled = true;
                
                try {
                    const response = await fetch('/generate-insights', {
                        method: 'POST',
                        headers: {'Content-Type': 'application/json'}
                    });
                    
                    const result = await response.json();
                    
                    if (response.ok) {
                        displayInsights(result);
                    } else {
                        document.getElementById('aiInsights').innerHTML = 
                            '<p style="color: red;">Failed to generate insights: ' + result.detail + '</p>';
                    }
                } catch (error) {
                    document.getElementById('aiInsights').innerHTML = 
                        '<p style="color: red;">Error: ' + error.message + '</p>';
                } finally {
                    button.textContent = originalText;
                    button.disabled = false;
                }
            }
            
            function displayInsights(result) {
                const insights = result.insights;
                const predictions = result.predictions;
                
                let html = '<h4>📊 Analysis Results</h4>';
                
                if (insights && !insights.error) {
                    html += '<div style="margin-bottom: 20px;">';
                    html += '<h5>🎯 Key Insights</h5>';
                    html += '<ul>';
                    if (insights.key_insights) {
                        insights.key_insights.forEach(insight => {
                            html += '<li>' + insight + '</li>';
                        });
                    }
                    html += '</ul>';
                    
                    if (insights.supplement_effectiveness) {
                        html += '<p><strong>Supplement Effectiveness:</strong> ' + insights.supplement_effectiveness + '</p>';
                    }
                    
                    if (insights.recommendations) {
                        html += '<h5>💡 Recommendations</h5>';
                        html += '<ul>';
                        insights.recommendations.forEach(rec => {
                            html += '<li>' + rec + '</li>';
                        });
                        html += '</ul>';
                    }
                    html += '</div>';
                }
                
                if (predictions && !predictions.error && predictions.predictions) {
                    html += '<div>';
                    html += '<h5>🔮 Biomarker Predictions</h5>';
                    predictions.predictions.forEach(pred => {
                        const changeColor = pred.predicted_change > 0 ? 'green' : 'red';
                        html += '<div style="margin: 10px 0; padding: 10px; background: #f0f0f0; border-radius: 4px;">';
                        html += '<strong>' + pred.biomarker + '</strong><br>';
                        html += 'Current: ' + pred.current_value + ' → ';
                        html += 'Predicted: ' + pred.predicted_value + ' ';
                        html += '<span style="color: ' + changeColor + ';">(' + 
                               (pred.predicted_change > 0 ? '+' : '') + pred.predicted_change.toFixed(2) + ')</span><br>';
                        html += '<small>' + pred.reasoning + '</small>';
                        html += '</div>';
                    });
                    html += '</div>';
                }
                
                if ((!insights || insights.error) && (!predictions || predictions.error)) {
                    html += '<p>No insights available. Make sure you have uploaded health data and added biomarkers.</p>';
                }
                
                document.getElementById('aiInsights').innerHTML = html;
            }
            
            async function loadDashboardData() {
                try {
                    // Load biomarker trends
                    const biomarkerResponse = await fetch('/biomarkers');
                    const biomarkers = await biomarkerResponse.json();
                    
                    if (biomarkers.length > 0) {
                        updateBiomarkerChart(biomarkers);
                    }
                    
                    // Load health metrics
                    const metricsResponse = await fetch('/health-metrics');
                    const metrics = await metricsResponse.json();
                    
                    if (metrics.length > 0) {
                        updateSupplementChart(metrics);
                    }
                    
                    // Load weekly changes for key metrics
                    await updateWeeklyChange();
                    
                    // Load next test predictions
                    await updateNextPrediction();
                    
                } catch (error) {
                    console.error('Failed to load dashboard data:', error);
                }
            }
            
            async function updateWeeklyChange() {
                try {
                    // Get weekly analysis for stress level (most important metric)
                    const response = await fetch('/weekly-analysis/stress_level');
                    const data = await response.json();
                    
                    if (data.latest_change && data.latest_change.weekly_pct_change !== null) {
                        const change = data.latest_change.weekly_pct_change * 100;
                        const changeText = change > 0 ? `+${change.toFixed(1)}%` : `${change.toFixed(1)}%`;
                        const changeColor = change < 0 ? '#28a745' : '#dc3545';  // Green for reduction in stress
                        
                        document.getElementById('weeklyChange').innerHTML = 
                            `<span style="color: ${changeColor}">${changeText}</span>`;
                    } else {
                        document.getElementById('weeklyChange').textContent = 'No data';
                    }
                } catch (error) {
                    document.getElementById('weeklyChange').textContent = 'Error';
                    console.error('Failed to load weekly change:', error);
                }
            }
            
            async function updateNextPrediction() {
                try {
                    // Get biomarkers for prediction
                    const biomarkerResponse = await fetch('/biomarkers');
                    const biomarkers = await biomarkerResponse.json();
                    
                    if (biomarkers.length > 0) {
                        // Show next test date (30 days from latest biomarker test)
                        const latestTest = biomarkers.reduce((latest, current) => {
                            return new Date(current.test_date) > new Date(latest.test_date) ? current : latest;
                        });
                        
                        const nextTestDate = new Date(latestTest.test_date);
                        nextTestDate.setDate(nextTestDate.getDate() + 30);  // 30 days from last test
                        
                        const today = new Date();
                        const daysUntil = Math.ceil((nextTestDate - today) / (1000 * 60 * 60 * 24));
                        
                        if (daysUntil > 0) {
                            document.getElementById('nextPrediction').innerHTML = `${daysUntil} days`;
                        } else {
                            document.getElementById('nextPrediction').innerHTML = '<span style="color: #dc3545;">Due now</span>';
                        }
                    } else {
                        document.getElementById('nextPrediction').textContent = 'No tests';
                    }
                } catch (error) {
                    document.getElementById('nextPrediction').textContent = 'Error';
                    console.error('Failed to load next prediction:', error);
                }
            }
            
            function updateBiomarkerChart(data) {
                // Show vertical bar chart of latest biomarker values
                if (data.length === 0) return;
                
                // Get latest value for each biomarker
                const latest = {};
                data.forEach(item => {
                    const biomarker = item.name;
                    const date = new Date(item.test_date);
                    if (!latest[biomarker] || new Date(latest[biomarker].test_date) < date) {
                        latest[biomarker] = item;
                    }
                });
                
                const names = Object.keys(latest);
                const values = names.map(name => latest[name].value);
                const units = names.map(name => latest[name].unit || '');
                
                const trace = {
                    x: names,
                    y: values,
                    type: 'bar',
                    marker: {
                        color: values.map((_, i) => `hsl(${i * 137.5 % 360}, 70%, 50%)`)
                    },
                    text: values.map((v, i) => `${v} ${units[i]}`),
                    textposition: 'auto'
                };
                
                Plotly.newPlot('biomarkerChart', [trace], {
                    title: 'Latest Biomarker Values',
                    xaxis: { title: 'Biomarker', tickangle: -45 },
                    yaxis: { title: 'Value' },
                    responsive: true,
                    margin: { b: 100 }
                });
            }
            
            function updateSupplementChart(data) {
                // Show supplement timeline with health metrics overlay
                const trace = {
                    x: data.map(d => d.measurement_date),
                    y: data.map(d => d.value),
                    type: 'scatter',
                    mode: 'lines+markers',
                    name: 'Health Metrics',
                    line: { color: '#007bff', width: 2 }
                };
                
                Plotly.newPlot('supplementChart', [trace], {
                    title: 'Supplement Timeline & Health Metrics',
                    xaxis: { title: 'Date' },
                    yaxis: { title: 'Value' },
                    responsive: true,
                    showlegend: true
                });
            }
            
            // Load initial data
            loadDashboardData();
        </script>
    </body>
    </html>
    """

@app.post("/upload-csv")
async def upload_csv(file: UploadFile = File(...)):
    if not file.filename or not file.filename.endswith('.csv'):
        raise HTTPException(status_code=400, detail="File must be a CSV")
    
    try:
        # Read CSV with Polars
        content = await file.read()
        df = pl.read_csv(content)
        
        # Store in database
        conn = sqlite3.connect('health_data.db')
        cursor = conn.cursor()
        
        # Validate CSV has required columns: date, metric_name, value, unit
        required_columns = ['date', 'metric_name', 'value', 'unit']
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            raise HTTPException(status_code=400, detail=f"CSV missing required columns: {missing_columns}. Expected: {required_columns}")
        
        # Process valid CSV data
        for row in df.iter_rows(named=True):
            cursor.execute('''
                INSERT INTO health_metrics (metric_name, value, unit, measurement_date, source)
                VALUES (?, ?, ?, ?, ?)
            ''', (
                row['metric_name'],
                float(row['value']),
                row.get('unit', ''),
                row['date'],
                file.filename
            ))
        
        conn.commit()
        conn.close()
        
        return {"message": f"Successfully uploaded {len(df)} health metrics"}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to process CSV: {str(e)}")

@app.post("/biomarkers")
async def add_biomarker(biomarker: Dict[str, Any]):
    try:
        conn = sqlite3.connect('health_data.db')
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO biomarkers (name, value, unit, test_date)
            VALUES (?, ?, ?, ?)
        ''', (
            biomarker['name'],
            biomarker['value'],
            biomarker.get('unit', ''),
            biomarker['test_date']
        ))
        
        conn.commit()
        conn.close()
        
        return {"message": "Biomarker added successfully"}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to add biomarker: {str(e)}")

@app.get("/biomarkers")
async def get_biomarkers():
    try:
        conn = sqlite3.connect('health_data.db')
        cursor = conn.cursor()
        
        cursor.execute('SELECT * FROM biomarkers ORDER BY test_date')
        rows = cursor.fetchall()
        
        columns = ['id', 'name', 'value', 'unit', 'reference_range_min', 'reference_range_max', 'test_date', 'created_at']
        biomarkers = [dict(zip(columns, row)) for row in rows]
        
        conn.close()
        return biomarkers
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch biomarkers: {str(e)}")

@app.get("/health-metrics")
async def get_health_metrics():
    try:
        conn = sqlite3.connect('health_data.db')
        cursor = conn.cursor()
        
        cursor.execute('SELECT * FROM health_metrics ORDER BY measurement_date')
        rows = cursor.fetchall()
        
        columns = ['id', 'metric_name', 'value', 'unit', 'measurement_date', 'source', 'created_at']
        metrics = [dict(zip(columns, row)) for row in rows]
        
        conn.close()
        return metrics
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch health metrics: {str(e)}")

@app.post("/supplements")
async def add_supplement(supplement: Dict[str, Any]):
    try:
        conn = sqlite3.connect('health_data.db')
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO supplements (name, start_date, dosage, expected_biomarkers, notes)
            VALUES (?, ?, ?, ?, ?)
        ''', (
            supplement['name'],
            supplement['start_date'],
            supplement.get('dosage', ''),
            supplement.get('expected_biomarkers', ''),
            supplement.get('notes', '')
        ))
        
        conn.commit()
        conn.close()
        
        return {"message": "Supplement added successfully"}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to add supplement: {str(e)}")

@app.get("/supplements")
async def get_supplements():
    try:
        conn = sqlite3.connect('health_data.db')
        cursor = conn.cursor()
        
        cursor.execute('SELECT * FROM supplements ORDER BY start_date')
        rows = cursor.fetchall()
        
        columns = ['id', 'name', 'start_date', 'dosage', 'expected_biomarkers', 'notes', 'created_at']
        supplements = [dict(zip(columns, row)) for row in rows]
        
        conn.close()
        return supplements
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch supplements: {str(e)}")

@app.post("/analyze-supplement-effect")
async def analyze_supplement_effect(request: Dict[str, Any]):
    try:
        supplement_name = request['supplement_name']
        target_metrics = request.get('target_metrics', ['resting_heart_rate', 'sleep_hours'])
        
        analyzer = RegressionDiscontinuityAnalyzer()
        results = analyze_supplement_effectiveness(supplement_name, target_metrics)
        
        return results
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")

@app.post("/generate-insights")
async def generate_insights():
    try:
        # Get current data
        biomarkers = await get_biomarkers()
        health_metrics = await get_health_metrics()
        supplements = await get_supplements()
        
        # Calculate weekly changes
        analyzer = RegressionDiscontinuityAnalyzer()
        weekly_changes = {}
        
        if health_metrics:
            unique_metrics = list(set([m['metric_name'] for m in health_metrics]))
            for metric in unique_metrics[:3]:  # Limit to prevent API overuse
                changes = analyzer.calculate_weekly_changes(metric)
                if len(changes) > 0:
                    weekly_changes[metric] = changes.to_dicts()
        
        # Generate AI insights
        supplement_effects = {}
        if supplements and health_metrics:
            latest_supplement = supplements[-1] if supplements else None
            if latest_supplement:
                effects = analyze_supplement_effectiveness(
                    latest_supplement['name'], 
                    ['resting_heart_rate', 'sleep_hours']
                )
                supplement_effects = effects
        
        insights = generate_health_insights(biomarkers, supplement_effects, weekly_changes)
        predictions = predict_biomarker_changes(biomarkers, supplements, weekly_changes)
        
        return {
            "insights": insights,
            "predictions": predictions,
            "data_summary": {
                "biomarkers": len(biomarkers),
                "health_metrics": len(health_metrics),
                "supplements": len(supplements)
            }
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to generate insights: {str(e)}")

@app.get("/weekly-analysis/{metric_name}")
async def get_weekly_analysis(metric_name: str):
    try:
        analyzer = RegressionDiscontinuityAnalyzer()
        weekly_data = analyzer.calculate_weekly_changes(metric_name)
        
        if len(weekly_data) == 0:
            return {"error": f"No data found for metric: {metric_name}"}
        
        return {
            "metric": metric_name,
            "weekly_data": weekly_data.to_dicts(),
            "latest_change": weekly_data.tail(1).to_dicts()[0] if len(weekly_data) > 0 else None
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Weekly analysis failed: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=5000)