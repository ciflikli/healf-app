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
        
        # Convert date column to datetime with robust parsing
        df = df.with_columns([
            pl.col("measurement_date").str.to_date(strict=False).alias("date")
        ]).filter(pl.col("date").is_not_null())  # Remove rows with invalid dates
        
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
                pl.col("start_date").str.to_date(strict=False).alias("start_date")
            ]).filter(pl.col("start_date").is_not_null())
        
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
            
            # Correct coefficient mapping for X = [const, time, treatment, time_treatment]
            # β₀ = intercept, β₁ = pre_slope, β₂ = treatment_effect (jump), β₃ = slope_change
            intercept = coeffs[0] if len(coeffs) > 0 else 0
            pre_slope = coeffs[1] if len(coeffs) > 1 else 0  
            treatment_effect = coeffs[2] if len(coeffs) > 2 else 0  # Actual treatment jump
            slope_change = coeffs[3] if len(coeffs) > 3 else 0     # Change in slope
            
            # Get p-values and standard errors for proper significance testing
            p_values = model.pvalues.values if hasattr(model, 'pvalues') else [None] * len(coeffs)
            std_errors = model.bse.values if hasattr(model, 'bse') else [None] * len(coeffs)
            
            return {
                "metric": metric_name,
                "intercept": float(intercept),
                "pre_slope": float(pre_slope), 
                "treatment_effect": float(treatment_effect),  # Corrected: actual jump at cutoff
                "slope_change": float(slope_change),          # Corrected: change in trend
                "treatment_p_value": float(p_values[2]) if len(p_values) > 2 and p_values[2] is not None else None,
                "treatment_std_error": float(std_errors[2]) if len(std_errors) > 2 and std_errors[2] is not None else None,
                "r_squared": float(r_squared),
                "n_observations": len(analysis_data),
                "bandwidth_days": bandwidth,
                "supplement_start": supplement_start_date,
                "statistically_significant": bool(p_values[2] < 0.05) if len(p_values) > 2 and p_values[2] is not None else False
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
    
    start_date_obj = supplement_info.select("start_date").to_numpy()[0][0]
    if hasattr(start_date_obj, 'strftime'):
        start_date = start_date_obj.strftime('%Y-%m-%d')
    else:
        start_date = str(start_date_obj)[:10]  # Convert numpy datetime to string
    
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
        
        # Proper significance check using p-values
        if "treatment_effect" in effect_analysis and "statistically_significant" in effect_analysis:
            total_effects += 1
            if effect_analysis["statistically_significant"]:
                significant_effects += 1
    
    if total_effects > 0:
        results["overall_effectiveness"] = significant_effects / total_effects
    
    return results


def build_rdd_plot_series(effect_data: Dict, rdd_data: pl.DataFrame) -> Dict:
    """
    Build plot series data for RDD visualization with before/after slopes
    """
    try:
        if 'treatment_effect' not in effect_data or len(rdd_data) == 0:
            return {"error": "Insufficient data for plot series"}
        
        # Get the data points
        x_values = rdd_data.select('running_var').to_numpy().flatten()
        y_values = rdd_data.select('value').to_numpy().flatten()
        treatment = rdd_data.select('treatment').to_numpy().flatten()
        
        # Split into before and after treatment
        before_mask = treatment == 0
        after_mask = treatment == 1
        
        x_before = x_values[before_mask] 
        y_before = y_values[before_mask]
        x_after = x_values[after_mask]
        y_after = y_values[after_mask]
        
        plot_data = {
            "scatter": {
                "x_all": x_values.tolist(),
                "y_all": y_values.tolist(),
                "treatment": treatment.tolist()
            },
            "cutoff": 0,
            "treatment_effect": effect_data.get('treatment_effect', 0)
        }
        
        # Fit regression lines for before and after
        if len(x_before) >= 3:
            # Before treatment line
            X_before = sm.add_constant(x_before)
            model_before = sm.OLS(y_before, X_before).fit()
            x_line_before = np.linspace(min(x_before), 0, 50)
            y_line_before = model_before.predict(sm.add_constant(x_line_before))
            
            plot_data["before_line"] = {
                "x": x_line_before.tolist(),
                "y": y_line_before.tolist()
            }
        
        if len(x_after) >= 3:
            # After treatment line  
            X_after = sm.add_constant(x_after)
            model_after = sm.OLS(y_after, X_after).fit()
            x_line_after = np.linspace(0, max(x_after), 50)
            y_line_after = model_after.predict(sm.add_constant(x_line_after))
            
            plot_data["after_line"] = {
                "x": x_line_after.tolist(),
                "y": y_line_after.tolist()
            }
        
        return plot_data
        
    except Exception as e:
        return {"error": f"Failed to build plot series: {str(e)}"}


def decompose_metric_arima(metric_name: str, freq: str = 'D', horizon: int = 30) -> Dict:
    """
    Decompose time series using ARIMA and generate forecast
    """
    try:
        from statsmodels.tsa.seasonal import seasonal_decompose
        from statsmodels.tsa.arima.model import ARIMA
        from statsmodels.tsa.statespace.tools import diff
        import warnings
        warnings.filterwarnings('ignore')
        
        # Load health data
        analyzer = RegressionDiscontinuityAnalyzer()
        health_data = analyzer.get_health_data()
        
        # Filter for specific metric
        metric_data = health_data.filter(pl.col("metric_name") == metric_name)
        
        if len(metric_data) < 14:  # Need at least 2 weeks of data
            return {"error": f"Insufficient data for {metric_name} (need at least 14 points)"}
        
        # Sort by date and create time series
        ts_data = metric_data.sort("date").with_columns([
            pl.col("date").cast(pl.Date),
            pl.col("value").cast(pl.Float64)
        ])
        
        # Handle duplicate dates by averaging
        ts_data = ts_data.group_by("date").agg(pl.col("value").mean())
        ts_data = ts_data.sort("date")
        
        dates = ts_data.select("date").to_pandas().squeeze()
        values = ts_data.select("value").to_pandas().squeeze()
        
        # Create proper time series index
        import pandas as pd
        ts = pd.Series(values.values, index=pd.to_datetime(dates.values))
        ts = ts.asfreq(freq)  # Set frequency
        
        result = {
            "metric": metric_name,
            "original": {
                "dates": [d.strftime('%Y-%m-%d') for d in ts.index],
                "values": ts.values.tolist()
            }
        }
        
        # Perform seasonal decomposition if we have enough data
        if len(ts) >= 24:  # Need at least 24 observations for decomposition
            try:
                decomposition = seasonal_decompose(ts, model='additive', period=7)
                
                result["decomposition"] = {
                    "trend": {
                        "dates": [d.strftime('%Y-%m-%d') for d in decomposition.trend.index],
                        "values": [v if not pd.isna(v) else None for v in decomposition.trend.values]
                    },
                    "seasonal": {
                        "dates": [d.strftime('%Y-%m-%d') for d in decomposition.seasonal.index],
                        "values": decomposition.seasonal.values.tolist()
                    },
                    "residual": {
                        "dates": [d.strftime('%Y-%m-%d') for d in decomposition.resid.index],
                        "values": [v if not pd.isna(v) else None for v in decomposition.resid.values]
                    }
                }
            except Exception as e:
                result["decomposition"] = {"error": f"Decomposition failed: {str(e)}"}
        
        # ARIMA forecast
        try:
            # Simple ARIMA(1,1,1) model as default
            model = ARIMA(ts, order=(1,1,1))
            fitted_model = model.fit()
            
            forecast = fitted_model.forecast(steps=horizon)
            conf_int = fitted_model.get_forecast(steps=horizon).conf_int()
            
            # Generate future dates
            future_dates = pd.date_range(start=ts.index[-1] + pd.Timedelta(days=1), 
                                       periods=horizon, freq=freq)
            
            result["forecast"] = {
                "dates": [d.strftime('%Y-%m-%d') for d in future_dates],
                "predicted": forecast.tolist(),
                "lower_ci": conf_int.iloc[:, 0].tolist(),
                "upper_ci": conf_int.iloc[:, 1].tolist(),
                "model_info": str(fitted_model.summary()).split('\n')[:10]  # First 10 lines only
            }
            
        except Exception as e:
            # Fallback to simple linear trend
            from scipy.stats import linregress
            
            x = np.arange(len(ts))
            slope, intercept, r_value, p_value, std_err = linregress(x, ts.values)
            
            future_x = np.arange(len(ts), len(ts) + horizon)
            future_values = slope * future_x + intercept
            future_dates = pd.date_range(start=ts.index[-1] + pd.Timedelta(days=1), 
                                       periods=horizon, freq=freq)
            
            result["forecast"] = {
                "dates": [d.strftime('%Y-%m-%d') for d in future_dates],
                "predicted": future_values.tolist(),
                "model_info": [f"Linear trend fallback: slope={slope:.3f}, R²={r_value**2:.3f}"],
                "method": "linear_fallback"
            }
        
        return result
        
    except Exception as e:
        return {"error": f"Time series analysis failed for {metric_name}: {str(e)}"}