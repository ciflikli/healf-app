import json
import os
from typing import Dict, List, Any
from openai import OpenAI

# the newest OpenAI model is "gpt-5" which was released August 7, 2025.
# do not change this unless explicitly requested by the user
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
openai_client = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None


def generate_health_insights(biomarker_data: List[Dict], 
                           supplement_effects: Dict,
                           weekly_changes: Dict) -> Dict:
    """
    Generate AI-powered insights about health data and supplement effectiveness
    """
    if not openai_client:
        return {"error": "OpenAI API key not configured"}
    
    try:
        # Prepare context for the AI model
        context = {
            "biomarkers": biomarker_data,
            "supplement_effects": supplement_effects,
            "weekly_changes": weekly_changes
        }
        
        prompt = f"""
        You are a health data analyst. Analyze the following health data and provide insights:
        
        Biomarker Data: {json.dumps(biomarker_data, indent=2)}
        Supplement Effects: {json.dumps(supplement_effects, indent=2)}
        Weekly Changes: {json.dumps(weekly_changes, indent=2)}
        
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
        return {"error": f"Failed to generate insights: {str(e)}"}


def predict_biomarker_changes(current_biomarkers: List[Dict],
                            supplement_timeline: List[Dict],
                            health_trends: Dict,
                            months_ahead: int = 6) -> Dict:
    """
    Use AI to predict biomarker changes based on current trends and supplementation
    """
    if not openai_client:
        return {"error": "OpenAI API key not configured"}
    
    try:
        prompt = f"""
        Based on the following health data, predict biomarker changes for the next {months_ahead} months:
        
        Current Biomarkers: {json.dumps(current_biomarkers, indent=2)}
        Supplement Timeline: {json.dumps(supplement_timeline, indent=2)}
        Health Trends: {json.dumps(health_trends, indent=2)}
        
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
        return {"error": f"Failed to generate predictions: {str(e)}"}


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