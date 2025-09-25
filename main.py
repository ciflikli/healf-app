from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
import sqlite3
import polars as pl
import json
import os
import time
from datetime import datetime
from typing import List, Dict, Any
from statistical_analysis import RegressionDiscontinuityAnalyzer, analyze_supplement_effectiveness, build_rdd_plot_series, decompose_metric_arima
from openai_insights import generate_health_insights, predict_biomarker_changes, analyze_biomarker_correlations, generate_biomarker_delta_predictions

app = FastAPI(title="Health Data Analytics", version="1.0.0")

# Cache for biomarker predictions to avoid slow repeated API calls
biomarker_prediction_cache = {}
CACHE_EXPIRY_HOURS = 1  # Cache results for 1 hour

# Function to pre-generate predictions for all biomarkers
async def pregenerate_all_biomarker_predictions():
    """Generate predictions for all biomarkers at 3, 6, and 12 months and store in database"""
    try:
        print("Starting pre-generation using OpenAI GPT-4o with time-specific predictions...")
        
        # Get all biomarkers and supplements
        biomarkers = await get_biomarkers()
        supplements = await get_supplements()
        
        # Get unique biomarker names
        unique_biomarkers = list(set([b['name'] for b in biomarkers]))
        months_list = [3, 6, 12]
        
        conn = sqlite3.connect('health_data.db')
        cursor = conn.cursor()
        
        predictions_generated = 0
        
        for biomarker_name in unique_biomarkers:
            for months_ahead in months_list:
                print(f"Generating prediction for {biomarker_name} at {months_ahead} months...")
                
                try:
                    # Generate the prediction using OpenAI GPT-4o
                    prediction = generate_biomarker_delta_predictions(
                        biomarker_name, biomarkers, supplements, months_ahead
                    )
                    
                    # Store in database (replace if exists)
                    cursor.execute('''
                        INSERT OR REPLACE INTO biomarker_predictions 
                        (biomarker_name, months_ahead, predicted_delta, predicted_value, current_value,
                         is_improvement, confidence, reasoning, recommendation, timeframe, 
                         analysis_method, direction_context)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ''', (
                        biomarker_name,
                        months_ahead,
                        prediction.get('predicted_delta', 0),
                        prediction.get('predicted_value', 0),
                        prediction.get('current_value', 0),
                        prediction.get('is_improvement', False),
                        prediction.get('confidence', 0),
                        prediction.get('reasoning', ''),
                        prediction.get('recommendation', ''),
                        prediction.get('timeframe', ''),
                        prediction.get('analysis_method', ''),
                        prediction.get('direction_context', '')
                    ))
                    
                    predictions_generated += 1
                    print(f"✅ Stored prediction for {biomarker_name} at {months_ahead} months")
                    
                except Exception as e:
                    print(f"Failed to generate prediction for {biomarker_name} at {months_ahead} months: {e}")
                    continue
        
        conn.commit()
        conn.close()
        
        print(f"Pre-generation completed: {predictions_generated} predictions stored")
        return {"message": f"Generated {predictions_generated} predictions", "total": predictions_generated}
        
    except Exception as e:
        print(f"Pre-generation failed: {e}")
        return {"error": f"Pre-generation failed: {str(e)}"}

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
    
    # Biomarker predictions table for pre-computed predictions
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS biomarker_predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            biomarker_name TEXT NOT NULL,
            months_ahead INTEGER NOT NULL,
            predicted_delta REAL NOT NULL,
            predicted_value REAL NOT NULL,
            current_value REAL NOT NULL,
            is_improvement BOOLEAN,
            confidence REAL NOT NULL,
            reasoning TEXT,
            recommendation TEXT,
            timeframe TEXT,
            analysis_method TEXT,
            direction_context TEXT,
            generated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(biomarker_name, months_ahead)
        )
    ''')
    
    conn.commit()
    conn.close()

@app.on_event("startup")
async def startup_event():
    init_database()
    
    # Schedule pre-generation of biomarker predictions in background (non-blocking)
    import asyncio
    import threading
    
    def run_pregeneration():
        """Run pre-generation in a separate thread to avoid blocking startup"""
        import asyncio
        asyncio.run(pregenerate_all_biomarker_predictions())
    
    # Start pre-generation in background thread so it doesn't block server startup
    thread = threading.Thread(target=run_pregeneration, daemon=True)
    thread.start()
    print("Pre-generation started in background thread")

@app.get("/", response_class=HTMLResponse)
async def dashboard():
    with open('new_dashboard.html', 'r') as f:
        return f.read()

# API endpoints

@app.post("/populate-supplement-biomarkers")
async def populate_supplement_biomarkers():
    """Use LLM to populate expected biomarkers for supplements"""
    try:
        from openai_insights import generate_supplement_biomarkers
        
        # Get supplements with empty expected_biomarkers
        conn = sqlite3.connect('health_data.db')
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT id, name, dosage FROM supplements 
            WHERE expected_biomarkers IS NULL OR expected_biomarkers = '' OR expected_biomarkers = 'None'
        """)
        supplements_to_update = cursor.fetchall()
        
        if not supplements_to_update:
            conn.close()
            return {"message": "All supplements already have expected biomarkers populated"}
        
        updated_count = 0
        for supplement_id, name, dosage in supplements_to_update:
            # Generate expected biomarkers using LLM
            expected_biomarkers = await generate_supplement_biomarkers(name, dosage)
            
            if expected_biomarkers:
                # Update database
                cursor.execute("""
                    UPDATE supplements 
                    SET expected_biomarkers = ? 
                    WHERE id = ?
                """, (expected_biomarkers, supplement_id))
                updated_count += 1
        
        conn.commit()
        conn.close()
        
        return {
            "message": f"Successfully populated expected biomarkers for {updated_count} supplements",
            "updated_supplements": updated_count
        }
        
    except Exception as e:
        return {"error": f"Failed to populate supplement biomarkers: {str(e)}"}

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


@app.get("/biomarkers/dates")
async def get_biomarker_dates():
    """Get discrete test dates per biomarker for date selector"""
    try:
        conn = sqlite3.connect('health_data.db')
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT name, test_date, COUNT(*) as test_count
            FROM biomarkers 
            GROUP BY name, test_date
            ORDER BY name, test_date DESC
        ''')
        rows = cursor.fetchall()
        
        # Group by biomarker name
        biomarker_dates = {}
        for name, test_date, count in rows:
            if name not in biomarker_dates:
                biomarker_dates[name] = []
            biomarker_dates[name].append({
                "date": test_date,
                "test_count": count
            })
        
        conn.close()
        return biomarker_dates
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch biomarker dates: {str(e)}")


@app.get("/metrics/decompose/{metric_name}")
async def decompose_metric(metric_name: str, freq: str = "D", horizon: int = 30):
    """Get ARIMA time series decomposition and forecast for a health metric"""
    try:
        result = decompose_metric_arima(metric_name, freq, horizon)
        return result
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Time series decomposition failed: {str(e)}")


@app.get("/rdd/plot/{metric_name}/{intervention_date}")
async def get_rdd_plot_data(metric_name: str, intervention_date: str, bandwidth: int = 30):
    """Get RDD analysis with plot series data for before/after slopes using intervention date"""
    try:
        # Validate intervention date format
        from datetime import datetime
        try:
            datetime.strptime(intervention_date, '%Y-%m-%d')
        except ValueError:
            raise HTTPException(status_code=400, detail=f"Invalid date format. Expected YYYY-MM-DD, got: {intervention_date}")
        
        # Perform RDD analysis using the intervention date directly
        analyzer = RegressionDiscontinuityAnalyzer()
        effect_analysis = analyzer.estimate_rdd_effect(metric_name, intervention_date, bandwidth=bandwidth)
        
        # Get the RDD data for plotting
        rdd_data = analyzer.prepare_rdd_data(metric_name, intervention_date)
        
        # Build plot series with before/after slopes
        plot_data = build_rdd_plot_series(effect_analysis, rdd_data)
        
        return {
            "effect_analysis": effect_analysis,
            "plot_data": plot_data,
            "intervention_date": intervention_date,
            "metric": metric_name,
            "start_date": intervention_date,  # Keep for backwards compatibility
            "bandwidth_days": bandwidth
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"RDD plot analysis failed: {str(e)}")


@app.get("/ai/biomarker-delta/{biomarker_name}")
async def get_biomarker_delta_prediction(biomarker_name: str, months_ahead: int = 6):
    """Get AI-powered biomarker delta prediction with directional improvement logic"""
    try:
        # Create cache key from biomarker name and months ahead
        cache_key = f"{biomarker_name.lower()}_{months_ahead}"
        current_time = time.time()
        
        # Check cache first
        if cache_key in biomarker_prediction_cache:
            cached_data, cached_time = biomarker_prediction_cache[cache_key]
            # Return cached result if within expiry time (1 hour = 3600 seconds)
            if current_time - cached_time < CACHE_EXPIRY_HOURS * 3600:
                # Create a copy of cached data and add cache indicators
                result = cached_data.copy()
                result["cached"] = True
                result["cached_at"] = datetime.fromtimestamp(cached_time).isoformat()
                return result
        
        # Cache miss or expired - generate new prediction
        biomarkers = await get_biomarkers()
        supplements = await get_supplements()
        
        # Generate delta prediction
        prediction = generate_biomarker_delta_predictions(
            biomarker_name, biomarkers, supplements, months_ahead
        )
        
        # Create result with cache indicator and store clean copy in cache
        result = prediction.copy()
        result["cached"] = False
        
        # Store clean prediction (without cache indicators) in cache
        biomarker_prediction_cache[cache_key] = (prediction, current_time)
        
        return result
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Biomarker delta prediction failed: {str(e)}")


@app.post("/ai/pregenerate-predictions")
async def pregenerate_predictions():
    """Pre-generate biomarker predictions for all biomarkers at 3, 6, and 12 months"""
    result = await pregenerate_all_biomarker_predictions()
    return result


@app.get("/ai/stored-predictions")
async def get_stored_predictions():
    """Get all pre-generated biomarker predictions"""
    try:
        conn = sqlite3.connect('health_data.db')
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT biomarker_name, months_ahead, predicted_delta, predicted_value, current_value,
                   is_improvement, confidence, reasoning, recommendation, timeframe,
                   analysis_method, direction_context, generated_at
            FROM biomarker_predictions
            ORDER BY biomarker_name, months_ahead
        ''')
        
        rows = cursor.fetchall()
        conn.close()
        
        predictions = []
        for row in rows:
            predictions.append({
                "biomarker_name": row[0],
                "months_ahead": row[1], 
                "predicted_delta": row[2],
                "predicted_value": row[3],
                "current_value": row[4],
                "is_improvement": bool(row[5]),
                "confidence": row[6],
                "reasoning": row[7],
                "recommendation": row[8],
                "timeframe": row[9],
                "analysis_method": row[10],
                "direction_context": row[11],
                "generated_at": row[12]
            })
        
        return predictions
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch stored predictions: {str(e)}")


@app.get("/health-metrics/unique")
async def get_unique_health_metrics():
    """Get list of unique health metric names for dropdowns"""
    try:
        conn = sqlite3.connect('health_data.db')
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT DISTINCT metric_name, COUNT(*) as data_points
            FROM health_metrics 
            GROUP BY metric_name
            ORDER BY data_points DESC, metric_name
        ''')
        rows = cursor.fetchall()
        
        metrics = [{"name": name, "data_points": count} for name, count in rows]
        
        conn.close()
        return metrics
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch unique health metrics: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=5000)