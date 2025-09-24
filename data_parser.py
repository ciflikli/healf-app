import polars as pl
import sqlite3
from datetime import datetime, timedelta
import re
from typing import List, Dict, Any
import os


class RealDataParser:
    """Parser for real health data files with various formats"""
    
    def __init__(self, db_path: str = 'health_data.db'):
        self.db_path = db_path
    
    def parse_date_flexible(self, date_str: str) -> str:
        """Parse various date formats and return YYYY-MM-DD"""
        date_str = str(date_str).strip()
        
        # Handle DD/MM/YYYY format (supplements)
        if re.match(r'\d{1,2}/\d{1,2}/\d{4}', date_str):
            try:
                parts = date_str.split('/')
                if len(parts[0]) <= 2 and int(parts[0]) <= 12:  # Likely MM/DD/YYYY
                    return datetime.strptime(date_str, '%m/%d/%Y').strftime('%Y-%m-%d')
                else:  # DD/MM/YYYY
                    return datetime.strptime(date_str, '%d/%m/%Y').strftime('%Y-%m-%d')
            except:
                return date_str
        
        # Handle YYYY-MM-DD format (already correct)
        if re.match(r'\d{4}-\d{1,2}-\d{1,2}', date_str):
            return date_str
        
        # Handle weekly ranges like "Sep 16-22" 
        if re.match(r'[A-Za-z]{3} \d{1,2}-\d{1,2}', date_str):
            return self.parse_weekly_date(date_str)
        
        # Handle ranges like "Aug 26 - Sep 1"
        if ' - ' in date_str:
            return self.parse_date_range(date_str)
        
        return date_str
    
    def parse_weekly_date(self, date_str: str) -> str:
        """Parse weekly date ranges like 'Sep 16-22' to the start date"""
        try:
            # Extract month and start day
            parts = date_str.split()
            month_str = parts[0]
            day_range = parts[1]
            start_day = int(day_range.split('-')[0])
            
            # Assume current year context (2025 for recent data)
            year = 2025 if month_str in ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep'] else 2024
            
            month_map = {
                'Jan': 1, 'Feb': 2, 'Mar': 3, 'Apr': 4, 'May': 5, 'Jun': 6,
                'Jul': 7, 'Aug': 8, 'Sep': 9, 'Oct': 10, 'Nov': 11, 'Dec': 12
            }
            
            month = month_map.get(month_str, 1)
            return datetime(year, month, start_day).strftime('%Y-%m-%d')
        except:
            return '2025-01-01'  # Default fallback
    
    def parse_date_range(self, date_str: str) -> str:
        """Parse date ranges like 'Aug 26 - Sep 1' to start date"""
        try:
            start_part = date_str.split(' - ')[0].strip()
            
            # Handle formats like "Aug 26" or "Dec 31, 2024"
            if ',' in start_part:
                return datetime.strptime(start_part, '%b %d, %Y').strftime('%Y-%m-%d')
            else:
                # Assume 2025 for recent months
                year = 2025
                return datetime.strptime(f"{start_part}, {year}", '%b %d, %Y').strftime('%Y-%m-%d')
        except:
            return '2025-01-01'  # Default fallback
    
    def load_supplements_data(self, file_path: str) -> int:
        """Load supplements data from CSV"""
        try:
            df = pl.read_csv(file_path)
            
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            count = 0
            for row in df.iter_rows(named=True):
                if row['name'] and row['start_date']:
                    start_date = self.parse_date_flexible(row['start_date'])
                    dosage = f"{row.get('daily_dose', '')} {row.get('dose_unit', '')}"
                    
                    cursor.execute('''
                        INSERT OR REPLACE INTO supplements (name, start_date, dosage, expected_biomarkers, notes)
                        VALUES (?, ?, ?, ?, ?)
                    ''', (
                        row['name'],
                        start_date,
                        dosage.strip(),
                        '',  # Will be filled in manually
                        f"Loaded from {file_path}"
                    ))
                    count += 1
            
            conn.commit()
            conn.close()
            return count
        except Exception as e:
            print(f"Error loading supplements: {e}")
            return 0
    
    def extract_biomarkers_from_text(self, text_content: str) -> List[Dict[str, Any]]:
        """Extract biomarker data from lab report text"""
        biomarkers = []
        test_date = "2025-07-30"  # From the PDF
        
        # Define biomarker patterns and their parsing
        biomarker_patterns = {
            # Lipids
            'HDL Cholesterol': r'HDL Cholesterol:\s+(\d+\.?\d*)\s+mmol/L',
            'Non-HDL Cholesterol': r'Non-HDL Cholesterol:\s+(\d+\.?\d*)\s+mmol/L',
            'Total Cholesterol': r'Total Cholesterol:\s+(\d+\.?\d*)\s+mmol/L',
            'LDL': r'LDL:\s+(\d+\.?\d*)\s+mmol/L',
            'Triglycerides': r'Triglycerides:\s+(\d+\.?\d*)\s+mmol/L',
            'Cholesterol HDL Ratio': r'Cholesterol HDL Ratio:\s+(\d+\.?\d*)\s+Ratio',
            
            # Liver Function
            'Albumin': r'Albumin:\s+(\d+\.?\d*)\s+g/L',
            'Globulin': r'Globulin:\s+(\d+\.?\d*)\s+g/L',
            'Total Protein': r'Total Protein:\s+(\d+\.?\d*)\s+g/L',
            'ALT': r'ALT:\s+(\d+\.?\d*)\s+IU/L',
            'ALP': r'ALP:\s+(\d+\.?\d*)\s+U/L',
            'Total Bilirubin': r'Total Bilirubin:\s+(\d+\.?\d*)\s+umol/L',
            'GGT': r'GGT:\s+(\d+\.?\d*)\s+U/L',
            
            # Kidney Function
            'Urea': r'Urea:\s+(\d+\.?\d*)\s+mmol/L',
            'Creatinine': r'Creatinine:\s+(\d+\.?\d*)\s+umol/L',
            'Uric Acid': r'Uric Acid:\s+(\d+\.?\d*)\s+umol/L',
            
            # Iron Profile
            'Ferritin': r'Ferritin:\s+(\d+\.?\d*)\s+ug/L',
            'Iron': r'Iron:\s+(\d+\.?\d*)\s+umol/L',
            'TIBC': r'TIBC:\s+(\d+\.?\d*)\s+umol/L',
            'UIBC': r'UIBC:\s+(\d+\.?\d*)\s+umol/L',
            'Transferrin Saturation': r'Transferrin Saturation:\s+(\d+\.?\d*)\s+%',
            
            # Thyroid
            'Free T4': r'Free T4:\s+(\d+\.?\d*)\s+pmol/L',
            'Free T3': r'Free T3:\s+(\d+\.?\d*)\s+pmol/L',
            'TSH': r'Thyroid Stimulating Hormone:\s+(\d+\.?\d*)\s+uIU/mL',
            
            # Vitamins
            'Vitamin B12 Active': r'Vitamin B12 \(Active\):\s+(\d+\.?\d*)\s+pmol/L',
            'Vitamin D': r'Vitamin D:\s+(\d+\.?\d*)\s+nmol/L',
            'Folate': r'Folate:\s+(\d+\.?\d*)\s+ug/L',
            
            # Hormones
            'Free Testosterone': r'Free Testosterone:\s+(\d+\.?\d*)\s+nmol/L',
            'SHBG': r'SHBG:\s+(\d+\.?\d*)\s+nmol/L',
            'Testosterone': r'Testosterone:\s+(\d+\.?\d*)\s+nmol/L',
            
            # General Chemistry
            'HbA1c': r'HbA1c:\s+(\d+\.?\d*)\s+mmol/mol',
            'HSCRP': r'HSCRP:\s+(\d+\.?\d*)\s+mg/L',
            'Creatinine Kinase': r'Creatinine Kinase:\s+(\d+\.?\d*)\s+U/L',
            'Magnesium': r'Magnesium:\s+(\d+\.?\d*)\s+mmol/L',
            'Copper': r'Copper:\s+(\d+\.?\d*)\s+umol/L',
            'Zinc': r'Zinc:\s+(\d+\.?\d*)\s+umol/L'
        }
        
        for biomarker_name, pattern in biomarker_patterns.items():
            match = re.search(pattern, text_content)
            if match:
                value = float(match.group(1))
                biomarkers.append({
                    'name': biomarker_name,
                    'value': value,
                    'test_date': test_date,
                    'unit': self.get_unit_for_biomarker(biomarker_name)
                })
        
        return biomarkers
    
    def get_unit_for_biomarker(self, name: str) -> str:
        """Get appropriate unit for biomarker"""
        unit_map = {
            'HDL Cholesterol': 'mmol/L', 'Non-HDL Cholesterol': 'mmol/L', 'Total Cholesterol': 'mmol/L',
            'LDL': 'mmol/L', 'Triglycerides': 'mmol/L', 'Cholesterol HDL Ratio': 'ratio',
            'Albumin': 'g/L', 'Globulin': 'g/L', 'Total Protein': 'g/L',
            'ALT': 'IU/L', 'ALP': 'U/L', 'Total Bilirubin': 'umol/L', 'GGT': 'U/L',
            'Urea': 'mmol/L', 'Creatinine': 'umol/L', 'Uric Acid': 'umol/L',
            'Ferritin': 'ug/L', 'Iron': 'umol/L', 'TIBC': 'umol/L', 'UIBC': 'umol/L',
            'Transferrin Saturation': '%', 'Free T4': 'pmol/L', 'Free T3': 'pmol/L',
            'TSH': 'uIU/mL', 'Vitamin B12 Active': 'pmol/L', 'Vitamin D': 'nmol/L',
            'Folate': 'ug/L', 'Free Testosterone': 'nmol/L', 'SHBG': 'nmol/L',
            'Testosterone': 'nmol/L', 'HbA1c': 'mmol/mol', 'HSCRP': 'mg/L',
            'Creatinine Kinase': 'U/L', 'Magnesium': 'mmol/L', 'Copper': 'umol/L', 'Zinc': 'umol/L'
        }
        return unit_map.get(name, '')
    
    def load_biomarkers_from_pdf_text(self, pdf_content: str) -> int:
        """Load biomarkers from PDF text content"""
        biomarkers = self.extract_biomarkers_from_text(pdf_content)
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        count = 0
        for biomarker in biomarkers:
            cursor.execute('''
                INSERT OR REPLACE INTO biomarkers (name, value, unit, test_date)
                VALUES (?, ?, ?, ?)
            ''', (
                biomarker['name'],
                biomarker['value'],
                biomarker['unit'],
                biomarker['test_date']
            ))
            count += 1
        
        conn.commit()
        conn.close()
        return count
    
    def load_health_metrics_csv(self, file_path: str, metric_mapping: Dict[str, str]) -> int:
        """Load health metrics from CSV with flexible column mapping"""
        try:
            df = pl.read_csv(file_path, truncate_ragged_lines=True)
            
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            count = 0
            for row in df.iter_rows(named=True):
                # Get date from first column (usually unnamed or date column)
                date_value = None
                for col_name, col_value in row.items():
                    if col_name in ['Date', '', None] or 'date' in str(col_name).lower():
                        date_value = col_value
                        break
                
                if not date_value:
                    continue
                    
                measurement_date = self.parse_date_flexible(str(date_value))
                
                # Process each metric column
                for col_name, col_value in row.items():
                    if col_name in metric_mapping and col_value is not None:
                        try:
                            # Clean numeric values
                            if isinstance(col_value, str):
                                # Remove non-numeric characters except decimal points
                                cleaned_value = re.sub(r'[^\d.-]', '', col_value)
                                if cleaned_value:
                                    numeric_value = float(cleaned_value)
                                else:
                                    continue
                            else:
                                numeric_value = float(col_value)
                            
                            cursor.execute('''
                                INSERT INTO health_metrics (metric_name, value, unit, measurement_date, source)
                                VALUES (?, ?, ?, ?, ?)
                            ''', (
                                metric_mapping[col_name],
                                numeric_value,
                                '',  # Units will be inferred
                                measurement_date,
                                os.path.basename(file_path)
                            ))
                            count += 1
                        except (ValueError, TypeError):
                            continue
            
            conn.commit()
            conn.close()
            return count
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
            return 0
    
    def load_all_real_data(self) -> Dict[str, int]:
        """Load all real data files"""
        results = {}
        
        # Load supplements
        results['supplements'] = self.load_supplements_data('attached_assets/supplements_1758739677605.csv')
        
        # Load biomarkers from PDF text (manually extracted)
        pdf_text = """
        HDL Cholesterol: 1.68 mmol/L
        Non-HDL Cholesterol: 3.92 mmol/L
        Total Cholesterol: 5.6 mmol/L
        LDL: 3.36 mmol/L
        Triglycerides: 1.23 mmol/L
        Cholesterol HDL Ratio: 3.3 Ratio
        Albumin: 45 g/L
        Globulin: 24 g/L
        Total Protein: 69 g/L
        ALT: 21.6 IU/L
        ALP: 72 U/L
        Total Bilirubin: 11.6 umol/L
        GGT: 23 U/L
        Urea: 6.1 mmol/L
        Creatinine: 79 umol/L
        Uric Acid: 289 umol/L
        Ferritin: 118 ug/L
        Iron: 13.10 umol/L
        TIBC: 58.3 umol/L
        UIBC: 45.2 umol/L
        Transferrin Saturation: 22.5 %
        Free T4: 15.1 pmol/L
        Free T3: 4.20 pmol/L
        Thyroid Stimulating Hormone: 2.03 uIU/mL
        Vitamin B12 (Active): 86.5 pmol/L
        Vitamin D: 64.0 nmol/L
        Folate: 2.6 ug/L
        Free Testosterone: 0.29 nmol/L
        SHBG: 35.9 nmol/L
        Testosterone: 14.900 nmol/L
        HbA1c: 32 mmol/mol
        HSCRP: 4.65 mg/L
        Creatinine Kinase: 262 U/L
        Magnesium: 0.89 mmol/L
        Copper: 16.3 umol/L
        Zinc: 14.7 umol/L
        """
        results['biomarkers'] = self.load_biomarkers_from_pdf_text(pdf_text)
        
        # Load health metrics CSVs
        health_metrics_files = [
            ('attached_assets/calories_data_1758739677607.csv', {
                'Active Calories': 'active_calories',
                'Resting Calories': 'resting_calories',
                'Total': 'total_calories'
            }),
            ('attached_assets/floors_climbed_data_1758739677607.csv', {
                'Climbed Floors': 'floors_climbed',
                'Descended Floors': 'floors_descended'
            }),
            ('attached_assets/resting_heart_rate_data_1758739677608.csv', {
                'Resting Heart Rate': 'resting_heart_rate'
            }),
            ('attached_assets/steps_data_1758739677609.csv', {
                'Actual': 'steps'
            }),
            ('attached_assets/stress_data_1758739677609.csv', {
                'Stress': 'stress_level'
            }),
            ('attached_assets/intensity_minutes_data_1758739677607.csv', {
                'Actual': 'intensity_minutes'
            })
        ]
        
        for file_path, mapping in health_metrics_files:
            file_name = os.path.basename(file_path)
            results[file_name] = self.load_health_metrics_csv(file_path, mapping)
        
        # Handle sleep data separately (has complex format)
        results['sleep_data'] = self.load_sleep_data('attached_assets/sleep_data_1758739677608.csv')
        
        return results
    
    def load_sleep_data(self, file_path: str) -> int:
        """Load sleep data with special handling for duration and score"""
        try:
            df = pl.read_csv(file_path, truncate_ragged_lines=True)
            
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            count = 0
            for row in df.iter_rows(named=True):
                date_str = row.get('Date', '')
                if not date_str or date_str == '--':
                    continue
                
                measurement_date = self.parse_date_flexible(date_str)
                
                # Extract sleep score
                if row.get('Avg Score') and row['Avg Score'] != '--':
                    try:
                        score = float(row['Avg Score'])
                        cursor.execute('''
                            INSERT INTO health_metrics (metric_name, value, unit, measurement_date, source)
                            VALUES (?, ?, ?, ?, ?)
                        ''', ('sleep_score', score, 'score', measurement_date, os.path.basename(file_path)))
                        count += 1
                    except ValueError:
                        pass
                
                # Extract sleep duration (convert "7h 30min" to hours)
                if row.get('Avg Duration') and row['Avg Duration'] != '--':
                    duration_str = row['Avg Duration']
                    try:
                        # Parse "7h 30min" format
                        hours = 0
                        minutes = 0
                        
                        hour_match = re.search(r'(\d+)h', duration_str)
                        if hour_match:
                            hours = int(hour_match.group(1))
                        
                        min_match = re.search(r'(\d+)min', duration_str)
                        if min_match:
                            minutes = int(min_match.group(1))
                        
                        total_hours = hours + minutes / 60.0
                        
                        cursor.execute('''
                            INSERT INTO health_metrics (metric_name, value, unit, measurement_date, source)
                            VALUES (?, ?, ?, ?, ?)
                        ''', ('sleep_duration', total_hours, 'hours', measurement_date, os.path.basename(file_path)))
                        count += 1
                    except:
                        pass
            
            conn.commit()
            conn.close()
            return count
        except Exception as e:
            print(f"Error loading sleep data: {e}")
            return 0