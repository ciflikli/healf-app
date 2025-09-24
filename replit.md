# Health Data Analytics Platform

## Overview

A FastAPI-based health analytics platform that applies advanced statistical methods to analyze the effectiveness of supplements on biomarkers and health metrics. The system uses Regression Discontinuity Design (RDD) to measure causal effects of interventions on health outcomes, integrates with OpenAI for intelligent insights generation, and provides data visualization capabilities for health trends analysis.

## User Preferences

Preferred communication style: Simple, everyday language.

## System Architecture

### Backend Framework
- **FastAPI**: Chosen for its automatic API documentation, type hints support, and high performance for data-heavy operations
- **Python-based analytics stack**: Leverages scientific computing libraries for statistical analysis

### Data Processing Layer
- **Polars**: Primary data manipulation library, selected for superior performance with large datasets compared to pandas
- **Polars-OLS**: Specialized extension for statistical modeling and regression analysis
- **Statsmodels + SciPy**: Provides advanced statistical methods for regression discontinuity analysis

### Statistical Analysis Engine
- **Regression Discontinuity Design (RDD)**: Core methodology for measuring causal effects of supplement interventions
- **Custom RegressionDiscontinuityAnalyzer**: Implements sophisticated statistical methods to isolate supplement effects from natural health variations
- **Multi-metric correlation analysis**: Analyzes relationships between different biomarkers and health metrics

### Database Architecture
- **SQLite**: Lightweight relational database for structured health data storage
- **Three-table schema**:
  - `biomarkers`: Lab test results with reference ranges and temporal tracking
  - `supplements`: Intervention timeline with dosage and expected effects
  - `health_metrics`: Continuous monitoring data from wearables and manual entries

### AI Integration Layer
- **OpenAI GPT-5**: Latest model for generating contextual health insights and recommendations
- **Structured prompt engineering**: Ensures consistent, actionable AI responses in JSON format
- **Multi-dimensional analysis**: Combines statistical results with domain knowledge for comprehensive insights

### API Design
- **RESTful endpoints**: Standard HTTP methods for data upload, analysis, and retrieval
- **File upload support**: Handles CSV imports for bulk health data ingestion
- **JSON responses**: Standardized data format for frontend consumption

## External Dependencies

### AI Services
- **OpenAI API**: Powers intelligent health insights, biomarker prediction, and correlation analysis using GPT-5 model

### Python Scientific Stack
- **Polars**: High-performance DataFrame library for data manipulation
- **Polars-OLS**: Statistical modeling extension for regression analysis
- **Statsmodels**: Advanced statistical analysis and econometric modeling
- **SciPy**: Scientific computing functions for statistical tests
- **NumPy**: Numerical computing foundation

### Web Framework
- **FastAPI**: Modern Python web framework with automatic API documentation
- **SQLite3**: Built-in Python database interface

### Development Environment
- **Environment variables**: OpenAI API key configuration for external service authentication