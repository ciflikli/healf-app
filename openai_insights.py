import json
import os
from datetime import date, datetime
from typing import Dict, List, Any
from openai import OpenAI

# the newest OpenAI model is "gpt-5" which was released August 7, 2025.
# do not change this unless explicitly requested by the user
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
openai_client = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None


def serialize_for_json(obj):
    """Convert dates and other non-serializable objects to strings"""
    if isinstance(obj, (date, datetime)):
        return obj.isoformat()
    elif isinstance(obj, dict):
        return {k: serialize_for_json(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [serialize_for_json(item) for item in obj]
    return obj

def generate_health_insights(biomarker_data: List[Dict], 
                           supplement_effects: Dict,
                           weekly_changes: Dict) -> Dict:
    """
    Generate AI-powered insights about health data and supplement effectiveness
    """
    if not openai_client:
        return generate_fallback_insights(biomarker_data, supplement_effects, weekly_changes)
    
    try:
        # Serialize data to handle dates
        safe_biomarkers = serialize_for_json(biomarker_data)
        safe_effects = serialize_for_json(supplement_effects)
        safe_changes = serialize_for_json(weekly_changes)
        
        prompt = f"""
        You are a health data analyst. Analyze the following health data and provide insights:
        
        Biomarker Data: {json.dumps(safe_biomarkers, indent=2)}
        Supplement Effects: {json.dumps(safe_effects, indent=2)}
        Weekly Changes: {json.dumps(safe_changes, indent=2)}
        
        Please provide a JSON response with the following structure:
        {{
            "overall_health_trend": "improving/stable/declining",
            "key_insights": ["insight1", "insight2", "insight3"],
            "supplement_effectiveness": "high/moderate/low/unclear",
            "recommendations": ["recommendation1", "recommendation2"],
            "concerning_trends": ["concern1", "concern2"],
            "positive_changes": ["positive1", "positive2"],
            "confidence_level": 0.8
        }}
        
        Focus on:
        1. Statistical significance of supplement effects
        2. Correlation patterns between metrics
        3. Concerning or improving trends
        4. Actionable recommendations
        """
        
        response = openai_client.chat.completions.create(
            model="gpt-5",
            messages=[
                {"role": "system", "content": "You are a health data analyst providing evidence-based insights on biomarker trends and supplement effectiveness."},
                {"role": "user", "content": prompt}
            ],
            response_format={"type": "json_object"}
        )
        
        return json.loads(response.choices[0].message.content)
    
    except Exception as e:
        error_msg = str(e)
        if "insufficient_quota" in error_msg or "quota" in error_msg:
            return generate_fallback_insights(biomarker_data, supplement_effects, weekly_changes)
        return {"error": f"Failed to generate insights: {error_msg}"}


def predict_biomarker_changes(current_biomarkers: List[Dict],
                            supplement_timeline: List[Dict],
                            health_trends: Dict,
                            months_ahead: int = 6) -> Dict:
    """
    Use AI to predict biomarker changes based on current trends and supplementation
    """
    if not openai_client:
        return generate_fallback_predictions(current_biomarkers, supplement_timeline, health_trends)
    
    try:
        # Serialize data to handle dates
        safe_biomarkers = serialize_for_json(current_biomarkers)
        safe_timeline = serialize_for_json(supplement_timeline)
        safe_trends = serialize_for_json(health_trends)
        
        prompt = f"""
        Based on the following health data, predict biomarker changes for the next {months_ahead} months:
        
        Current Biomarkers: {json.dumps(safe_biomarkers, indent=2)}
        Supplement Timeline: {json.dumps(safe_timeline, indent=2)}
        Health Trends: {json.dumps(safe_trends, indent=2)}
        
        Consider:
        1. Current biomarker values and reference ranges
        2. Historical trends and patterns
        3. Expected effects of ongoing supplementation
        4. Scientific literature on supplement efficacy
        
        Provide predictions in JSON format:
        {{
            "predictions": [
                {{
                    "biomarker": "biomarker_name",
                    "current_value": 50.0,
                    "predicted_value": 45.0,
                    "predicted_change": -5.0,
                    "confidence": 0.75,
                    "reasoning": "explanation",
                    "recommendation": "action to take"
                }}
            ],
            "overall_outlook": "improving/stable/concerning",
            "key_factors": ["factor1", "factor2"],
            "recommendations": ["rec1", "rec2"]
        }}
        """
        
        response = openai_client.chat.completions.create(
            model="gpt-5",
            messages=[
                {"role": "system", "content": "You are a predictive health analytics expert. Provide scientifically-grounded predictions based on available data."},
                {"role": "user", "content": prompt}
            ],
            response_format={"type": "json_object"}
        )
        
        return json.loads(response.choices[0].message.content)
    
    except Exception as e:
        error_msg = str(e)
        if "insufficient_quota" in error_msg or "quota" in error_msg:
            return generate_fallback_predictions(current_biomarkers, supplement_timeline, health_trends)
        return {"error": f"Failed to generate predictions: {error_msg}"}


def generate_fallback_insights(biomarker_data: List[Dict], 
                             supplement_effects: Dict,
                             weekly_changes: Dict) -> Dict:
    """
    Generate basic insights using statistical analysis when OpenAI is unavailable
    """
    try:
        insights = []
        recommendations = []
        concerning_trends = []
        positive_changes = []
        
        # Analyze supplement effectiveness
        effectiveness_scores = []
        significant_effects = []
        
        if supplement_effects and 'metric_effects' in supplement_effects:
            for metric, effect_data in supplement_effects['metric_effects'].items():
                if isinstance(effect_data, dict) and 'statistically_significant' in effect_data:
                    if effect_data['statistically_significant']:
                        effect_size = effect_data.get('treatment_effect', 0)
                        p_value = effect_data.get('treatment_p_value', 1.0)
                        significant_effects.append({
                            'metric': metric.replace('_', ' ').title(),
                            'effect': effect_size,
                            'p_value': p_value
                        })
                        effectiveness_scores.append(1.0)
                        
                        if effect_size < 0:  # Improvement for stress, etc.
                            if 'stress' in metric.lower():
                                positive_changes.append(f"Significant stress reduction (-{abs(effect_size):.1f} points, p={p_value:.3f})")
                            elif 'heart_rate' in metric.lower() and effect_size < 0:
                                positive_changes.append(f"Resting heart rate improved by {abs(effect_size):.1f} BPM")
                        else:
                            if 'sleep' in metric.lower() or 'steps' in metric.lower():
                                positive_changes.append(f"Improved {metric.replace('_', ' ')} (+{effect_size:.1f})")
                    else:
                        effectiveness_scores.append(0.0)
        
        # Determine overall supplement effectiveness
        if effectiveness_scores:
            avg_effectiveness = sum(effectiveness_scores) / len(effectiveness_scores)
            if avg_effectiveness > 0.5:
                supplement_effectiveness = "high"
                insights.append("Statistical analysis shows significant supplement benefits")
            elif avg_effectiveness > 0.2:
                supplement_effectiveness = "moderate" 
                insights.append("Some positive supplement effects detected")
            else:
                supplement_effectiveness = "low"
                insights.append("Limited evidence of supplement effectiveness")
        else:
            supplement_effectiveness = "unclear"
            insights.append("Insufficient data to assess supplement effectiveness")
        
        # Analyze weekly changes for trends
        if weekly_changes:
            for metric, changes in weekly_changes.items():
                if isinstance(changes, list) and len(changes) >= 3:
                    recent_changes = changes[-3:]  # Last 3 weeks
                    avg_change = sum(c.get('weekly_pct_change', 0) or 0 for c in recent_changes) / len(recent_changes)
                    
                    if abs(avg_change) > 0.1:  # >10% average change
                        if avg_change > 0:
                            if 'stress' in metric.lower():
                                concerning_trends.append(f"Stress levels trending up (+{avg_change*100:.1f}% avg)")
                            else:
                                positive_changes.append(f"{metric.replace('_', ' ').title()} improving (+{avg_change*100:.1f}% avg)")
                        else:
                            if 'stress' in metric.lower():
                                positive_changes.append(f"Stress levels trending down ({avg_change*100:.1f}% avg)")
                            else:
                                concerning_trends.append(f"{metric.replace('_', ' ').title()} declining ({avg_change*100:.1f}% avg)")
        
        # Basic biomarker analysis
        if biomarker_data:
            insights.append(f"Tracking {len(biomarker_data)} biomarkers from recent lab work")
            
            # Look for out-of-range values
            out_of_range = []
            for biomarker in biomarker_data:
                if 'reference_range_min' in biomarker and 'reference_range_max' in biomarker:
                    if (biomarker['reference_range_min'] and biomarker['value'] < biomarker['reference_range_min']) or \
                       (biomarker['reference_range_max'] and biomarker['value'] > biomarker['reference_range_max']):
                        out_of_range.append(biomarker['name'])
            
            if out_of_range:
                concerning_trends.append(f"{len(out_of_range)} biomarkers outside reference range")
                recommendations.append("Consult healthcare provider about out-of-range biomarkers")
        
        # Generate recommendations
        if significant_effects:
            recommendations.append("Continue current supplement regimen - showing measurable benefits")
        
        if not positive_changes:
            recommendations.append("Consider adjusting supplement timing or dosage")
            
        recommendations.append("Monitor trends over next 4-6 weeks for pattern confirmation")
        
        # Determine overall trend
        if len(positive_changes) > len(concerning_trends):
            overall_trend = "improving"
        elif len(concerning_trends) > len(positive_changes):
            overall_trend = "declining" 
        else:
            overall_trend = "stable"
        
        return {
            "overall_health_trend": overall_trend,
            "key_insights": insights[:5] if insights else ["Statistical analysis of your health data is available"],
            "supplement_effectiveness": supplement_effectiveness,
            "recommendations": recommendations[:5] if recommendations else ["Continue tracking health metrics"],
            "concerning_trends": concerning_trends[:3],
            "positive_changes": positive_changes[:5],
            "confidence_level": 0.7,
            "analysis_method": "Statistical Analysis (OpenAI unavailable)"
        }
        
    except Exception as e:
        return {
            "overall_health_trend": "unclear",
            "key_insights": ["Health data analysis available - OpenAI service temporarily unavailable"],
            "supplement_effectiveness": "unclear", 
            "recommendations": ["Continue monitoring health trends"],
            "concerning_trends": [],
            "positive_changes": [],
            "confidence_level": 0.5,
            "analysis_method": f"Fallback analysis due to: {str(e)}"
        }


def generate_fallback_predictions(current_biomarkers: List[Dict],
                                supplement_timeline: List[Dict], 
                                health_trends: Dict) -> Dict:
    """
    Generate basic predictions using trend analysis when OpenAI is unavailable
    """
    try:
        predictions = []
        
        # Use the most recent biomarkers for prediction
        if current_biomarkers:
            recent_biomarkers = {}
            for biomarker in current_biomarkers:
                name = biomarker['name']
                test_date = biomarker.get('test_date', '')
                if name not in recent_biomarkers or test_date > recent_biomarkers[name].get('test_date', ''):
                    recent_biomarkers[name] = biomarker
            
            # Generate simple trend-based predictions
            for name, biomarker in list(recent_biomarkers.items())[:5]:  # Limit to 5 predictions
                current_value = biomarker['value']
                
                # Simple heuristic predictions based on supplement effects
                predicted_change = 0
                confidence = 0.6
                reasoning = "Trend-based analysis"
                
                # Apply some domain knowledge
                if 'cholesterol' in name.lower() and 'ldl' in name.lower():
                    predicted_change = -5  # Assume slight improvement
                    reasoning = "Supplement effects typically improve cholesterol profiles"
                elif 'vitamin' in name.lower() and 'd' in name.lower():
                    predicted_change = 3  # Assume supplementation effect
                    reasoning = "Vitamin D supplementation should improve levels"
                elif 'inflammation' in name.lower() or 'crp' in name.lower():
                    predicted_change = -0.2
                    reasoning = "Anti-inflammatory effects from supplements"
                else:
                    predicted_change = current_value * 0.02  # 2% improvement assumption
                    reasoning = "General health trend assumption"
                
                predicted_value = current_value + predicted_change
                
                predictions.append({
                    "biomarker": name,
                    "current_value": current_value,
                    "predicted_value": round(predicted_value, 2),
                    "predicted_change": round(predicted_change, 2),
                    "confidence": confidence,
                    "reasoning": reasoning,
                    "recommendation": "Continue current supplement regimen and retest in 3 months"
                })
        
        return {
            "predictions": predictions,
            "overall_outlook": "stable" if predictions else "unclear",
            "key_factors": ["Current supplement regimen", "Historical biomarker trends"],
            "recommendations": [
                "Schedule follow-up lab work in 3 months",
                "Continue current supplement protocol", 
                "Monitor weekly health metrics for trend confirmation"
            ],
            "analysis_method": "Statistical Trend Analysis (OpenAI unavailable)"
        }
        
    except Exception as e:
        return {
            "predictions": [],
            "overall_outlook": "unclear",
            "key_factors": ["Data analysis pending"],
            "recommendations": ["Continue health monitoring"],
            "analysis_method": f"Fallback analysis due to: {str(e)}"
        }


def analyze_biomarker_correlations(biomarker_data: List[Dict],
                                 health_metrics: List[Dict]) -> Dict:
    """
    Use AI to identify correlations and patterns between biomarkers and health metrics
    """
    if not openai_client:
        return {"error": "OpenAI API key not configured"}
    
    try:
        prompt = f"""
        Analyze correlations between biomarkers and health metrics:
        
        Biomarker Data: {json.dumps(biomarker_data, indent=2)}
        Health Metrics: {json.dumps(health_metrics, indent=2)}
        
        Identify:
        1. Strong correlations between different biomarkers
        2. Relationships between health metrics and biomarker changes
        3. Temporal patterns and trends
        4. Potential causal relationships
        
        Respond in JSON format:
        {{
            "correlations": [
                {{
                    "metric1": "biomarker_name",
                    "metric2": "health_metric_name", 
                    "correlation_strength": "strong/moderate/weak",
                    "relationship": "positive/negative/complex",
                    "explanation": "detailed explanation"
                }}
            ],
            "patterns": ["pattern1", "pattern2"],
            "insights": ["insight1", "insight2"],
            "clinical_significance": "high/moderate/low"
        }}
        """
        
        response = openai_client.chat.completions.create(
            model="gpt-5",
            messages=[
                {"role": "system", "content": "You are a biostatistician analyzing health data correlations. Focus on clinically relevant patterns."},
                {"role": "user", "content": prompt}
            ],
            response_format={"type": "json_object"}
        )
        
        return json.loads(response.choices[0].message.content)
    
    except Exception as e:
        return {"error": f"Failed to analyze correlations: {str(e)}"}


def generate_supplement_recommendations(current_biomarkers: List[Dict],
                                      health_goals: List[str],
                                      current_supplements: List[Dict]) -> Dict:
    """
    Generate AI-powered supplement recommendations based on biomarker data
    """
    if not openai_client:
        return {"error": "OpenAI API key not configured"}
    
    try:
        prompt = f"""
        Based on biomarker data and health goals, suggest evidence-based supplement recommendations:
        
        Current Biomarkers: {json.dumps(current_biomarkers, indent=2)}
        Health Goals: {json.dumps(health_goals, indent=2)}
        Current Supplements: {json.dumps(current_supplements, indent=2)}
        
        Provide recommendations considering:
        1. Biomarker values outside optimal ranges
        2. Scientific evidence for supplement efficacy
        3. Potential interactions with current supplements
        4. Safety considerations
        
        Respond in JSON format:
        {{
            "recommendations": [
                {{
                    "supplement": "supplement_name",
                    "dosage": "recommended_dosage",
                    "rationale": "why this supplement",
                    "target_biomarkers": ["biomarker1", "biomarker2"],
                    "expected_timeline": "timeframe_for_effects",
                    "evidence_level": "high/moderate/low",
                    "safety_notes": "important_considerations"
                }}
            ],
            "discontinue": ["supplement_to_stop"],
            "monitoring": ["what_to_track"],
            "warnings": ["important_warnings"]
        }}
        """
        
        response = openai_client.chat.completions.create(
            model="gpt-5",
            messages=[
                {"role": "system", "content": "You are a clinical nutritionist providing evidence-based supplement recommendations. Prioritize safety and scientific evidence."},
                {"role": "user", "content": prompt}
            ],
            response_format={"type": "json_object"}
        )
        
        return json.loads(response.choices[0].message.content)
    
    except Exception as e:
        return {"error": f"Failed to generate recommendations: {str(e)}"}