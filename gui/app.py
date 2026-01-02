import sys
import os
import threading
import pandas as pd
import json
from pathlib import Path
from flask import Flask, render_template, request, jsonify, redirect, url_for, send_from_directory

# Add project root to sys.path to allow imports from core/config
current_dir = Path(__file__).parent
project_root = current_dir.parent
sys.path.append(str(project_root))

# Import your existing modules
from config.settings import PATHS, ANALYSIS, initialize
from config.questions import get_opinion_questions, get_demographic_questions
from generators.survey_data_generator import SurveyDataGenerator
from core.archetypal_analyzer import ArchetypalAnalyzer
from core.encoding import SurveyEncoder
from generators.persona_generator import PersonaGenerator
from agents.survey_agent import SurveyAgent
from utils.file_io import load_dataframe, save_dataframe

initialize()

app = Flask(__name__)
app.secret_key = 'archetypes_secret_key'

# ============================================================================
# GLOBAL STATE
# ============================================================================
class SystemState:
    def __init__(self):
        self.df = None                # Current loaded survey data
        self.archetypes = None        # Extracted archetypes (matrices)
        self.proportions = None       # Weights
        self.personas = []            # Generated persona dicts
        self.is_busy = False          # For loading spinners
        self.status_message = "Ready" # Progress text
        self.analyzer = None          # The analyzer instance

state = SystemState()

# ============================================================================
# ROUTES: DASHBOARD & DATA
# ============================================================================

@app.route('/')
def index():
    """Dashboard: Load data or generate synthetic"""
    data_summary = None
    if state.df is not None:
        data_summary = {
            'rows': len(state.df),
            'cols': len(state.df.columns),
            'preview': state.df.head(5).to_html(classes='table table-sm table-striped')
        }
    
    return render_template('index.html', summary=data_summary, busy=state.is_busy)

@app.route('/generate_data', methods=['POST'])
def generate_data():
    """Trigger synthetic data generation"""
    try:
        n = int(request.form.get('n_respondents', 200))
        gen = SurveyDataGenerator()
        state.df = gen.generate(n_respondents=n)
        
        # Save to disk as per existing workflow
        save_dataframe(state.df, filename="gui_generated_survey.csv")
        return redirect(url_for('index'))
    except Exception as e:
        return f"Error: {str(e)}", 500

@app.route('/upload_data', methods=['POST'])
def upload_data():
    """Handle CSV upload"""
    if 'file' not in request.files:
        return redirect(request.url)
    file = request.files['file']
    if file.filename == '':
        return redirect(request.url)
    
    if file:
        state.df = pd.read_csv(file)
        save_dataframe(state.df, filename="gui_uploaded_survey.csv")
        return redirect(url_for('index'))

# ============================================================================
# ROUTES: ANALYSIS
# ============================================================================

@app.route('/analysis')
def analysis():
    """Archetypal Analysis Configuration"""
    if state.df is None:
        return redirect(url_for('index'))
    
    return render_template('analysis.html', 
                          n_archetypes=ANALYSIS.N_ARCHETYPES,
                          busy=state.is_busy)

@app.route('/run_analysis', methods=['POST'])
def run_analysis():
    """Run the NMF/PCHA Algorithm in background thread"""
    k = int(request.form.get('k', 4))
    
    def _background_task():
        state.is_busy = True
        state.status_message = "Encoding Data..."
        
        try:
            # 1. Encode
            encoder = SurveyEncoder()
            opinion_questions = get_opinion_questions()
            encoder.fit(opinion_questions)
            matrix = encoder.encode_dataframe(state.df, opinion_questions, use_opinion_only=True)
            
            # 2. Fit
            state.status_message = f"Extracting {k} Archetypes..."
            analyzer = ArchetypalAnalyzer(n_archetypes=k)
            state.archetypes, weights, state.proportions = analyzer.fit(matrix)
            state.analyzer = analyzer # Save for R2 scores
            
            state.status_message = "Analysis Complete"
        except Exception as e:
            state.status_message = f"Error: {e}"
        finally:
            state.is_busy = False

    thread = threading.Thread(target=_background_task)
    thread.start()
    
    return jsonify({'status': 'started'})

# ============================================================================
# ROUTES: PERSONAS
# ============================================================================

@app.route('/personas')
def personas():
    """View and Generate Personas"""
    has_archetypes = state.archetypes is not None
    return render_template('personas.html', 
                          personas=state.personas, 
                          has_archetypes=has_archetypes,
                          busy=state.is_busy)

@app.route('/generate_personas', methods=['POST'])
def generate_personas():
    """Trigger LLM generation"""
    if state.archetypes is None:
        return "No archetypes found", 400

    def _background_gen():
        state.is_busy = True
        state.status_message = "Connecting to Ollama..."
        
        try:
            pg = PersonaGenerator()
            questions = get_opinion_questions()
            
            # Use R2 scores if available, else defaults
            r2_scores = state.analyzer.get_r2_scores() if state.analyzer else [0.8]*len(state.archetypes)

            state.status_message = "Generating Personas (this takes time)..."
            state.personas = pg.generate_batch(
                archetypes=state.archetypes,
                questions=questions,
                weights=state.proportions,
                r2_scores=r2_scores
            )
            state.status_message = "Personas Generated"
        except Exception as e:
            state.status_message = f"Error: {e}"
            print(e)
        finally:
            state.is_busy = False

    thread = threading.Thread(target=_background_gen)
    thread.start()
    return jsonify({'status': 'started'})

# ============================================================================
# ROUTES: UTILS (Images & Status)
# ============================================================================

@app.route('/status')
def status():
    return jsonify({
        'busy': state.is_busy, 
        'message': state.status_message,
        'has_data': state.df is not None,
        'has_archetypes': state.archetypes is not None,
        'persona_count': len(state.personas)
    })

# Serve images from data/output/plots
@app.route('/plots/<path:filename>')
def serve_plot(filename):
    return send_from_directory(PATHS.PLOTS_DIR, filename)

if __name__ == '__main__':
    app.run(debug=True, port=5000)