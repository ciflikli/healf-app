from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
import sqlite3
import polars as pl
import json
import os
from datetime import datetime
from typing import List, Dict, Any

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
                        <small>Wearable data, daily metrics, health tracking</small>
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
            </div>
            
            <div class="card">
                <h3>💊 Supplement Timeline</h3>
                <div id="supplementChart" style="height: 400px;"></div>
            </div>
            
            <div class="card">
                <h3>📈 Biomarker Trends</h3>
                <div id="biomarkerChart" style="height: 400px;"></div>
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
                } catch (error) {
                    console.error('Failed to load dashboard data:', error);
                }
            }
            
            function updateBiomarkerChart(data) {
                const biomarkerGroups = {};
                data.forEach(item => {
                    if (!biomarkerGroups[item.name]) {
                        biomarkerGroups[item.name] = { x: [], y: [] };
                    }
                    biomarkerGroups[item.name].x.push(item.test_date);
                    biomarkerGroups[item.name].y.push(item.value);
                });
                
                const traces = Object.keys(biomarkerGroups).map(name => ({
                    x: biomarkerGroups[name].x,
                    y: biomarkerGroups[name].y,
                    type: 'scatter',
                    mode: 'lines+markers',
                    name: name,
                    line: { width: 3 }
                }));
                
                Plotly.newPlot('biomarkerChart', traces, {
                    title: 'Biomarker Trends Over Time',
                    xaxis: { title: 'Date' },
                    yaxis: { title: 'Value' },
                    responsive: true
                });
            }
            
            function updateSupplementChart(data) {
                // Placeholder for supplement timeline visualization
                const trace = {
                    x: data.map(d => d.measurement_date),
                    y: data.map(d => d.value),
                    type: 'scatter',
                    mode: 'lines+markers',
                    name: 'Health Metrics'
                };
                
                Plotly.newPlot('supplementChart', [trace], {
                    title: 'Health Metrics Timeline',
                    xaxis: { title: 'Date' },
                    yaxis: { title: 'Value' },
                    responsive: true
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
        
        # Assume CSV has columns: date, metric_name, value, unit
        for row in df.iter_rows(named=True):
            cursor.execute('''
                INSERT INTO health_metrics (metric_name, value, unit, measurement_date, source)
                VALUES (?, ?, ?, ?, ?)
            ''', (
                row.get('metric_name', 'unknown'),
                float(row.get('value', 0)),
                row.get('unit', ''),
                row.get('date', datetime.now().strftime('%Y-%m-%d')),
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

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=5000)