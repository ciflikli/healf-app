import polars as pl
import polars_ols  # Enables least_squares namespace
import sqlite3
import numpy as np
from typing import Dict, List, Tuple, Optional
from datetime import datetime, timedelta
import statsmodels.api as sm
from scipy import stats


class RegressionDiscontinuityAnalyzer:
    """
    Implements Regression Discontinuity Design (RDD) for measuring supplement effects
    on health metrics and biomarkers.
    """
    
    def __init__(self, db_path: str = 'health_data.db'):
        self.db_path = db_path
    
    def get_health_data(self) -> pl.DataFrame:
        """Load all health metrics from database"""
        conn = sqlite3.connect(self.db_path)
        
        query = """
        SELECT metric_name, value, measurement_date, source
        FROM health_metrics 
        ORDER BY measurement_date
        """
        
        df = pl.read_database(query, connection=conn)
        conn.close()
        
        # Convert date column to datetime
        df = df.with_columns([
            pl.col("measurement_date").str.to_date().alias("date")
        ])
        
        return df
    
    def get_supplement_timeline(self) -> pl.DataFrame:
        """Load supplement start dates"""
        conn = sqlite3.connect(self.db_path)
        
        query = """
        SELECT name, start_date, expected_biomarkers
        FROM supplements 
        ORDER BY start_date
        """
        
        df = pl.read_database(query, connection=conn)
        conn.close()
        
        if len(df) > 0:
            df = df.with_columns([
                pl.col("start_date").str.to_date().alias("start_date")
            ])
        
        return df
    
    def prepare_rdd_data(self, metric_name: str, supplement_start_date: str) -> pl.DataFrame:
        """
        Prepare data for regression discontinuity analysis
        
        Args:
            metric_name: Name of the health metric to analyze
            supplement_start_date: Date when supplementation started (YYYY-MM-DD)
        
        Returns:
            DataFrame with running variable and treatment indicator
        """
        health_data = self.get_health_data()
        supplement_date = datetime.strptime(supplement_start_date, '%Y-%m-%d').date()
        
        # Filter for specific metric
        metric_data = health_data.filter(pl.col("metric_name") == metric_name)
        
        if len(metric_data) == 0:
            raise ValueError(f"No data found for metric: {metric_name}")
        
        # Create running variable (days from supplement start)
        # and treatment indicator
        rdd_data = metric_data.with_columns([
            (pl.col("date") - pl.lit(supplement_date)).dt.total_days().alias("days_from_supplement"),
            (pl.col("date") >= pl.lit(supplement_date)).alias("post_supplement")
        ]).sort("days_from_supplement")
        
        return rdd_data
    
    def estimate_rdd_effect(self, metric_name: str, supplement_start_date: str, 
                           bandwidth: int = 30) -> Dict:
        """
        Estimate the effect of supplementation using RDD
        
        Args:
            metric_name: Health metric to analyze
            supplement_start_date: Supplement start date
            bandwidth: Days around cutoff to include in analysis
        
        Returns:
            Dictionary with RDD results
        """
        try:
            rdd_data = self.prepare_rdd_data(metric_name, supplement_start_date)
            
            # Apply bandwidth constraint
            analysis_data = rdd_data.filter(
                (pl.col("days_from_supplement") >= -bandwidth) &
                (pl.col("days_from_supplement") <= bandwidth)
            )
            
            if len(analysis_data) < 10:
                return {"error": "Insufficient data points for analysis"}
            
            # Prepare data for statsmodels analysis (fallback for now)
            analysis_pandas = analysis_data.select([
                pl.col("days_from_supplement").alias("time"),
                pl.col("post_supplement").cast(pl.Float64).alias("treatment"),
                (pl.col("days_from_supplement") * pl.col("post_supplement").cast(pl.Float64)).alias("time_treatment"),
                pl.col("value").alias("outcome")
            ]).to_pandas()
            
            # Use statsmodels for RDD estimation
            X = analysis_pandas[["time", "treatment", "time_treatment"]]
            X = sm.add_constant(X)
            y = analysis_pandas["outcome"]
            
            model = sm.OLS(y, X).fit()
            coeffs = model.params.values
            preds = model.fittedvalues.values
            
            # Calculate R-squared and other statistics
            actual = analysis_pandas["outcome"].values
            ss_res = np.sum((actual - preds) ** 2)
            ss_tot = np.sum((actual - np.mean(actual)) ** 2)
            r_squared = 1 - (ss_res / ss_tot)
            
            # Treatment effect is the coefficient on the treatment indicator (β₂)
            treatment_effect = coeffs[1] if len(coeffs) > 1 else 0
            
            return {
                "metric": metric_name,
                "treatment_effect": treatment_effect,
                "pre_trend": coeffs[0] if len(coeffs) > 0 else 0,
                "slope_change": coeffs[2] if len(coeffs) > 2 else 0,
                "r_squared": r_squared,
                "n_observations": len(analysis_data),
                "bandwidth_days": bandwidth,
                "supplement_start": supplement_start_date
            }
            
        except Exception as e:
            return {"error": f"RDD analysis failed: {str(e)}"}
    
    def calculate_weekly_changes(self, metric_name: str) -> pl.DataFrame:
        """
        Calculate week-over-week percentage changes for a health metric
        """
        health_data = self.get_health_data()
        metric_data = health_data.filter(pl.col("metric_name") == metric_name)
        
        if len(metric_data) == 0:
            return pl.DataFrame()
        
        # Calculate weekly averages
        weekly_data = metric_data.with_columns([
            pl.col("date").dt.strftime("%Y-W%U").alias("week")
        ]).group_by("week").agg([
            pl.col("value").mean().alias("avg_value"),
            pl.col("date").min().alias("week_start")
        ]).sort("week_start")
        
        # Calculate week-over-week changes
        weekly_changes = weekly_data.with_columns([
            pl.col("avg_value").pct_change().alias("weekly_pct_change"),
            (pl.col("avg_value") - pl.col("avg_value").shift(1)).alias("weekly_abs_change")
        ])
        
        return weekly_changes
    
    def predict_biomarker_delta(self, biomarker_name: str, 
                               supplement_effects: Dict, 
                               months_ahead: int = 6) -> Dict:
        """
        Predict biomarker changes based on supplement effects and current trends
        
        Args:
            biomarker_name: Name of biomarker to predict
            supplement_effects: Dictionary of supplement effects from RDD
            months_ahead: Months to predict ahead
        
        Returns:
            Prediction dictionary with confidence intervals
        """
        try:
            # Get historical biomarker data
            conn = sqlite3.connect(self.db_path)
            query = """
            SELECT value, test_date
            FROM biomarkers 
            WHERE name = ?
            ORDER BY test_date
            """
            
            biomarker_data = pl.read_database(
                query, 
                connection=conn,
                execute_options={"parameters": [biomarker_name]}
            )
            conn.close()
            
            if len(biomarker_data) < 2:
                return {"error": "Insufficient biomarker history for prediction"}
            
            # Convert dates and calculate trend
            biomarker_data = biomarker_data.with_columns([
                pl.col("test_date").str.to_date().alias("date")
            ])
            
            # Calculate natural trend (without supplement effect)
            values = biomarker_data.select("value").to_numpy().flatten()
            dates_numeric = np.arange(len(values))
            
            # Simple linear regression for baseline trend
            slope, intercept, r_value, p_value, std_err = stats.linregress(dates_numeric, values)
            
            # Project natural trend forward
            future_periods = months_ahead * 30 / np.mean(np.diff(dates_numeric)) if len(dates_numeric) > 1 else months_ahead
            natural_projection = intercept + slope * (len(values) + future_periods)
            
            # Add expected supplement effect
            supplement_effect = supplement_effects.get("treatment_effect", 0)
            expected_delta = supplement_effect * months_ahead * 0.5  # Assume gradual accumulation
            
            predicted_value = natural_projection + expected_delta
            current_value = values[-1]
            
            # Calculate confidence interval (simplified)
            prediction_std = std_err * np.sqrt(future_periods)
            confidence_interval = 1.96 * prediction_std
            
            return {
                "biomarker": biomarker_name,
                "current_value": float(current_value),
                "predicted_value": float(predicted_value),
                "predicted_delta": float(predicted_value - current_value),
                "confidence_interval": float(confidence_interval),
                "natural_trend": float(natural_projection - current_value),
                "supplement_effect": float(expected_delta),
                "months_ahead": months_ahead,
                "prediction_quality": abs(r_value) if not np.isnan(r_value) else 0
            }
            
        except Exception as e:
            return {"error": f"Biomarker prediction failed: {str(e)}"}


def analyze_supplement_effectiveness(supplement_name: str, 
                                   target_metrics: List[str]) -> Dict:
    """
    Comprehensive analysis of supplement effectiveness across multiple metrics
    """
    analyzer = RegressionDiscontinuityAnalyzer()
    
    # Get supplement start date
    supplements = analyzer.get_supplement_timeline()
    supplement_info = supplements.filter(pl.col("name") == supplement_name)
    
    if len(supplement_info) == 0:
        return {"error": f"Supplement '{supplement_name}' not found"}
    
    start_date = supplement_info.select("start_date").to_numpy()[0][0].strftime('%Y-%m-%d')
    
    results = {
        "supplement": supplement_name,
        "start_date": start_date,
        "metric_effects": {},
        "overall_effectiveness": 0
    }
    
    significant_effects = 0
    total_effects = 0
    
    for metric in target_metrics:
        effect_analysis = analyzer.estimate_rdd_effect(metric, start_date)
        results["metric_effects"][metric] = effect_analysis
        
        # Simple significance check (could be improved with proper p-values)
        if "treatment_effect" in effect_analysis:
            total_effects += 1
            if abs(effect_analysis["treatment_effect"]) > 0.1:  # Threshold for meaningful effect
                significant_effects += 1
    
    if total_effects > 0:
        results["overall_effectiveness"] = significant_effects / total_effects
    
    return results