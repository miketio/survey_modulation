# api/server.py - Complete rewrite with JSON configuration
import sys
import os
from pathlib import Path
from typing import List, Dict, Any
import json
import asyncio

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns

# Import existing logic
from config.settings import PATHS, ANALYSIS, DATA_GEN, initialize
from config.questions import Question, QuestionType, get_opinion_questions
from config.loader import (
    load_questions,
    save_questions,
    load_archetypes,
    save_archetypes,
    load_system_config,
    save_system_config,
    load_personas,
    save_personas,
    get_question_templates,
    get_archetype_sets
)
from generators.survey_data_generator import SurveyDataGenerator
from core.archetypal_analyzer import ArchetypalAnalyzer
from core.encoding import SurveyEncoder
from generators.persona_generator import PersonaGenerator
from agents.survey_agent import SurveyAgent
from utils.file_io import save_dataframe

# Initialize
initialize()

app = FastAPI(title="Survey Archetypes API v2 - JSON Config")

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================================================
# GLOBAL STATE
# ============================================================================

class AppState:
    def __init__(self):
        self.df = None
        self.encoder = None
        self.analyzer = None
        self.personas = []
        self.questions = []
        self.initial_archetypes = []
        
        # Load config from JSON
        system_config = load_system_config()
        self.config = {
            'demographicContext': system_config['data_generation']['demographic_context'],
            'temperature': system_config['ollama']['temperature_persona'],
            'randomSeed': system_config['analysis']['random_seed'],
            'respondents': system_config['data_generation']['n_respondents'],
            'calibrationSamples': system_config['simulation']['n_calibration_samples'],
            'simulatedPopulation': system_config['simulation']['n_simulated_respondents']
        }
        
        self.second_survey_questions = []
        self.calibration_data = []
        self.simulated_df = None

state = AppState()

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def questions_from_frontend(questions_data: List[Dict]) -> List[Question]:
    """Convert frontend question format to Question objects"""
    questions = []
    for q_data in questions_data:
        q_type = QuestionType(q_data['type'])
        
        if q_type == QuestionType.LIKERT:
            questions.append(Question(
                id=q_data['id'],
                text=q_data['text'],
                type=q_type,
                category='opinion',
                scale=(1, 5)
            ))
        elif q_type in [QuestionType.CATEGORICAL, QuestionType.ORDINAL]:
            options_data = q_data.get('options', '')
            
            if isinstance(options_data, list):
                options = [str(opt).strip() for opt in options_data if opt]
            elif isinstance(options_data, str):
                options = [opt.strip() for opt in options_data.split(',') if opt.strip()]
            else:
                options = ['Option 1', 'Option 2', 'Option 3']
            
            if not options:
                options = ['Option 1', 'Option 2', 'Option 3']
            
            questions.append(Question(
                id=q_data['id'],
                text=q_data['text'],
                type=q_type,
                category='demographic',
                options=options
            ))
    
    return questions
# ============================================================================
# BASIC ENDPOINTS
# ============================================================================

@app.get("/")
def health_check():
    return {"status": "online", "system": "Survey Archetypes v2 - JSON Config"}

# ============================================================================
# CONFIGURATION ENDPOINTS
# ============================================================================

@app.get("/api/config/system")
async def get_system_config_endpoint():
    """Get current system configuration"""
    try:
        config = load_system_config()
        return config
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.put("/api/config/system")
async def update_system_config_endpoint(config: Dict[str, Any]):
    """Update system configuration"""
    try:
        save_system_config(config)
        
        # Update state.config to match
        state.config = {
            'demographicContext': config['data_generation']['demographic_context'],
            'temperature': config['ollama']['temperature_persona'],
            'randomSeed': config['analysis']['random_seed'],
            'respondents': config['data_generation']['n_respondents'],
            'calibrationSamples': config['simulation']['n_calibration_samples'],
            'simulatedPopulation': config['simulation']['n_simulated_respondents']
        }
        
        return {"message": "Configuration updated", "config": config}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/config/questions")
async def get_available_question_templates():
    """Get list of available question templates"""
    try:
        templates = get_question_templates()
        return {"templates": templates}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/config/questions/{template_name}")
async def get_question_template(template_name: str):
    """Get questions from a specific template"""
    try:
        questions = load_questions(template_name)
        return {
            "template": template_name,
            "questions": [q.to_dict() for q in questions]
        }
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"Template '{template_name}' not found")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/config/questions/{template_name}")
async def save_question_template(template_name: str, request: Dict[str, Any]):
    """Save a custom question template"""
    try:
        questions_data = request.get('questions', [])
        questions = []
        
        for q_data in questions_data:
            q = Question(
                id=q_data['id'],
                text=q_data['text'],
                type=QuestionType(q_data['type']),
                category=q_data.get('category', 'opinion'),
                scale=tuple(q_data['scale']) if 'scale' in q_data else None,
                options=q_data.get('options')
            )
            questions.append(q)
        
        save_questions(
            questions,
            template_name,
            description=request.get('description', ''),
            version=request.get('version', '1.0')
        )
        
        return {
            "message": f"Template '{template_name}' saved",
            "count": len(questions)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/config/archetypes")
async def get_available_archetypes():
    """Get list of available archetype sets"""
    try:
        sets = get_archetype_sets(include_discovered=True)
        return {"archetype_sets": sets}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/config/archetypes/{set_name}")
async def get_archetype_set(set_name: str):
    """Get a specific archetype set"""
    try:
        archetypes = load_archetypes(set_name)
        return {
            "name": set_name,
            "archetypes": archetypes
        }
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"Archetype set '{set_name}' not found")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/config/archetypes/{set_name}")
async def save_archetype_set(set_name: str, request: Dict[str, Any]):
    """Save a custom archetype set"""
    try:
        archetypes = request.get('archetypes', [])
        
        save_archetypes(
            archetypes,
            set_name,
            description=request.get('description', ''),
            version=request.get('version', '1.0'),
            demographic_context=request.get('demographic_context', ''),
            discovered=request.get('discovered', False)
        )
        
        return {
            "message": f"Archetype set '{set_name}' saved",
            "count": len(archetypes)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ============================================================================
# DATA GENERATION
# ============================================================================

@app.post("/api/data/generate")
async def generate_data(request: Dict[str, Any]):
    """Generate synthetic survey data with custom initial archetypes"""
    try:
        config = request.get('config', {})
        state.config = config
        
        # Get archetypes from request or use default
        if 'archetypes' in request and request['archetypes']:
            state.initial_archetypes = request.get('archetypes', [])
        else:
            # Load default archetypes from JSON
            state.initial_archetypes = load_archetypes("default")
        
        # Update settings
        DATA_GEN.N_RESPONDENTS = config.get('respondents', 200)
        DATA_GEN.DEMOGRAPHIC_CONTEXT = config.get('demographicContext', 'University Students in New York')
        ANALYSIS.RANDOM_SEED = config.get('randomSeed', 42)
        
        # Parse questions
        questions_data = request.get('questions', [])
        state.questions = questions_from_frontend(questions_data)
        
        # Generate data with custom archetypes
        generator = SurveyDataGenerator(
            archetypes=state.initial_archetypes,
            seed=ANALYSIS.RANDOM_SEED
        )
        
        df = generator.generate(n_respondents=DATA_GEN.N_RESPONDENTS)
        state.df = df
        
        save_dataframe(df, filename="generated_survey.csv")
        
        return {
            "message": "Data generated successfully",
            "count": len(df),
            "archetypes": len(state.initial_archetypes),
            "config": state.config
        }
    
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

# ============================================================================
# ARCHETYPAL ANALYSIS
# ============================================================================

@app.post("/api/analysis/k-comparison")
async def run_k_comparison():
    """Run archetypal analysis for k=2 to 8"""
    try:
        if state.df is None:
            raise HTTPException(status_code=400, detail="No data available")
        
        opinion_qs = get_opinion_questions()
        
        state.encoder = SurveyEncoder()
        state.encoder.fit(opinion_qs)
        data_matrix = state.encoder.encode_dataframe(state.df, opinion_qs, use_opinion_only=True)
        
        results = []
        for k in range(2, 9):
            analyzer = ArchetypalAnalyzer(n_archetypes=k, random_state=ANALYSIS.RANDOM_SEED)
            analyzer.fit(data_matrix, method='auto', verbose=False)
            
            results.append({
                'k': k,
                'r2': float(analyzer.get_total_r2()),
                'proportions': [float(p) for p in analyzer.get_proportions()]
            })
        
        return {"message": "k-comparison complete", "results": results}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ============================================================================
# PERSONA GENERATION
# ============================================================================

@app.post("/api/personas/generate")
async def generate_personas(k: int):
    """Generate personas for selected k"""
    try:
        if state.df is None:
            raise HTTPException(status_code=400, detail="No data available")
        
        opinion_qs = get_opinion_questions()
        
        state.encoder = SurveyEncoder()
        state.encoder.fit(opinion_qs)
        data_matrix = state.encoder.encode_dataframe(state.df, opinion_qs, use_opinion_only=True)
        
        state.analyzer = ArchetypalAnalyzer(n_archetypes=k, random_state=ANALYSIS.RANDOM_SEED)
        archetypes, weights, proportions = state.analyzer.fit(data_matrix, method='auto', verbose=True)
        
        persona_gen = PersonaGenerator(demographic_context=state.config.get('demographicContext'))
        
        personas = []
        for i, archetype_pattern in enumerate(archetypes):
            persona = persona_gen.generate_persona(
                answers=archetype_pattern,
                questions=opinion_qs,
                archetype_index=i,
                weight=float(proportions[i]),
                r2=state.analyzer.get_total_r2(),
                contrast_personas=personas
            )
            personas.append(persona)
        
        state.personas = personas
        save_personas(personas, name="generated_personas")
        
        return {"message": "Personas generated", "count": len(personas), "personas": personas}
    
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/personas/update")
async def update_persona(persona: Dict[str, Any]):
    """Update a persona and regenerate its system prompt"""
    try:
        archetype_idx = persona.get('archetype_index')
        
        # Regenerate system prompt from edited fields
        from generators.persona_generator import generate_system_prompt
        persona['system_prompt'] = generate_system_prompt(persona)
        
        # Update in state
        for i, p in enumerate(state.personas):
            if p['archetype_index'] == archetype_idx:
                state.personas[i] = persona
                break
        
        save_personas(state.personas, name="generated_personas")
        return {"message": "Persona updated with regenerated system prompt", "persona": persona}
    
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

# ============================================================================
# SECOND SURVEY QUESTIONS
# ============================================================================

@app.post("/api/survey/questions")
async def save_second_survey_questions(questions: List[Dict]):
    """Save second survey questions"""
    try:
        state.second_survey_questions = questions_from_frontend(questions)
        return {"message": "Questions saved", "count": len(state.second_survey_questions)}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ============================================================================
# CALIBRATION PHASE
# ============================================================================

@app.websocket("/api/calibration/live")
async def calibration_live(websocket: WebSocket):
    """WebSocket endpoint for live calibration streaming"""
    await websocket.accept()
    
    try:
        # Receive config
        config_msg = await websocket.receive_json()
        n_samples = config_msg.get('n_samples', 10)
        
        if len(state.personas) == 0:
            await websocket.send_json({"type": "error", "message": "No personas available"})
            await asyncio.sleep(0.1)
            await websocket.close()
            return
        
        if len(state.second_survey_questions) == 0:
            await websocket.send_json({"type": "error", "message": "No questions defined"})
            await asyncio.sleep(0.1)
            await websocket.close()
            return
        
        # Create agents
        agents = [SurveyAgent(p) for p in state.personas]
        
        calibration_results = []
        total_items = len(agents) * len(state.second_survey_questions)
        processed = 0
        
        # Calibrate each agent on each question
        for agent_idx, agent in enumerate(agents):
            for q_idx, question in enumerate(state.second_survey_questions):
                # Collect n_samples answers
                samples = []
                for _ in range(n_samples):
                    answer = agent.answer_survey([question], verbose=False, n_samples=1)[0]
                    samples.append(answer)
                
                # Calculate statistics based on question type
                if question.type == QuestionType.LIKERT:
                    numeric_samples = []
                    for s in samples:
                        try:
                            numeric_samples.append(float(s))
                        except (ValueError, TypeError):
                            # If conversion fails, default to neutral (3)
                            numeric_samples.append(3.0)
                    
                    if len(numeric_samples) == 0:
                        # Fallback if no valid samples
                        numeric_samples = [3.0] * n_samples
                    
                    modal_answer = int(max(set(numeric_samples), key=numeric_samples.count))
                    mean_answer = float(np.mean(numeric_samples))
                    std_answer = float(np.std(numeric_samples))
                    
                    # Transition probabilities
                    n_lower = sum(1 for s in numeric_samples if s < modal_answer)
                    n_higher = sum(1 for s in numeric_samples if s > modal_answer)
                    n_stay = sum(1 for s in numeric_samples if s == modal_answer)
                    
                    total = len(numeric_samples)
                    p_lower = n_lower / total if total > 0 else 0.0
                    p_higher = n_higher / total if total > 0 else 0.0
                    p_stay = n_stay / total if total > 0 else 1.0
                    
                    result = {
                        'persona_index': agent_idx,
                        'persona_name': agent.get_name(),
                        'question_id': question.id,
                        'question_text': question.text,
                        'question_type': 'likert',
                        'modal_answer': modal_answer,
                        'mean_answer': mean_answer,
                        'std': std_answer,
                        'p_lower': p_lower,
                        'p_stay': p_stay,
                        'p_higher': p_higher,
                        'samples': numeric_samples
                    }
                
                else:  # Categorical or Ordinal
                    if not isinstance(samples, list):
                        samples = [samples]  
                    string_samples = [str(s) for s in samples]
                    
                    sample_counts = {}
                    for s in string_samples:
                        sample_counts[s] = sample_counts.get(s, 0) + 1
                    
                    if len(sample_counts) == 0:
                        # Fallback if no samples
                        modal_answer = question.options[0] if question.options else "Unknown"
                        p_stay = 1.0
                        p_change = 0.0
                        distribution = {modal_answer: n_samples}  
                    else:
                        modal_answer = max(sample_counts, key=sample_counts.get)
                        total_samples = len(string_samples)  
                        p_stay = sample_counts[modal_answer] / total_samples
                        p_change = 1.0 - p_stay
                        distribution = sample_counts  
                    
                    result = {
                        'persona_index': agent_idx,
                        'persona_name': agent.get_name(),
                        'question_id': question.id,
                        'question_text': question.text,
                        'question_type': question.type.value,
                        'modal_answer': modal_answer,
                        'mean_answer': None,
                        'std': None,
                        'p_lower': None,
                        'p_stay': p_stay,
                        'p_higher': None,
                        'p_change': p_change,
                        'samples': string_samples,  
                        'distribution': distribution  
                    }
                
                calibration_results.append(result)
                processed += 1
                
                # Stream update
                await websocket.send_json({
                    "type": "calibration_update",
                    "data": result,
                    "progress": (processed / total_items) * 100
                })
        
        # Save calibration data
        state.calibration_data = calibration_results
        
        # Create DataFrame - handle mixed types
        calibration_df = pd.DataFrame(calibration_results)
        save_dataframe(calibration_df, filename="calibration_data.csv")
        
        # Send completion
        await websocket.send_json({
            "type": "calibration_complete",
            "message": "Calibration complete",
            "total_items": len(calibration_results)
        })
        
        await asyncio.sleep(0.1)
        
    except WebSocketDisconnect:
        print("Client disconnected during calibration")
    except Exception as e:
        import traceback
        traceback.print_exc()
        try:
            await websocket.send_json({"type": "error", "message": str(e)})
            await asyncio.sleep(0.1)
        except:
            pass
    finally:
        try:
            await websocket.close()
        except:
            pass
        print("WebSocket connection closed")

# ============================================================================
# POPULATION SIMULATION
# ============================================================================

@app.post("/api/simulation/run")
async def run_simulation(request: Dict[str, Any]):
    """Run population simulation using calibrated probabilities"""
    try:
        if len(state.calibration_data) == 0:
            raise HTTPException(status_code=400, detail="No calibration data. Run calibration first.")
        
        n_respondents = request.get('n_respondents', 1000)
        
        # Build proportions from personas
        proportions = np.array([p['weight'] for p in state.personas])
        proportions = proportions / proportions.sum()
        
        # Assign archetypes to respondents
        archetype_assignments = np.random.choice(
            len(state.personas),
            size=n_respondents,
            p=proportions
        )
        
        # Generate responses using transition probabilities
        simulated_responses = []
        
        for resp_id in range(n_respondents):
            archetype_idx = archetype_assignments[resp_id]
            
            response = {
                'respondent_id': resp_id,
                'archetype_index': archetype_idx,
                'archetype_name': state.personas[archetype_idx]['name']
            }
            
            for q in state.second_survey_questions:
                # Find calibration data for this archetype+question
                calib_row = next((c for c in state.calibration_data 
                                 if c['persona_index'] == archetype_idx 
                                 and c['question_id'] == q.id), None)
                
                if calib_row:
                    if q.type == QuestionType.LIKERT:
                        # Use transition probabilities
                        base_answer = calib_row['modal_answer']
                        p_lower = calib_row['p_lower']
                        p_stay = calib_row['p_stay']
                        p_higher = calib_row['p_higher']
                        
                        # Sample transition
                        transition = np.random.choice([-1, 0, 1], p=[p_lower, p_stay, p_higher])
                        final_answer = int(np.clip(base_answer + transition, 1, 5))
                        
                        response[q.id] = final_answer
                    
                    else:  # Categorical or Ordinal
                        # Sample from distribution
                        distribution = calib_row.get('distribution', {})
                        if distribution:
                            options = list(distribution.keys())
                            probs = [distribution[opt] / len(calib_row['samples']) for opt in options]
                            response[q.id] = np.random.choice(options, p=probs)
                        else:
                            response[q.id] = calib_row['modal_answer']
                else:
                    # Default values
                    if q.type == QuestionType.LIKERT:
                        response[q.id] = 3
                    else:
                        response[q.id] = q.options[len(q.options) // 2] if q.options else 'Unknown'
            
            simulated_responses.append(response)
        
        # Create DataFrame
        simulated_df = pd.DataFrame(simulated_responses)
        state.simulated_df = simulated_df
        
        # Save
        save_dataframe(simulated_df, filename="simulated_survey.csv")
        
        # Calculate distribution
        archetype_dist = simulated_df['archetype_name'].value_counts().to_dict()
        
        return {
            "message": "Simulation complete",
            "total_respondents": n_respondents,
            "n_archetypes": len(state.personas),
            "n_questions": len(state.second_survey_questions),
            "total_answers": n_respondents * len(state.second_survey_questions),
            "archetype_distribution": archetype_dist
        }
    
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

# ============================================================================
# VISUALIZATION
# ============================================================================

@app.post("/api/visualization/simulated-population")
async def generate_visualization():
    """Generate simulated population visualization"""
    try:
        if state.simulated_df is None:
            raise HTTPException(status_code=400, detail="No simulation data available")
        
        # Filter Likert questions only
        likert_questions = [q for q in state.second_survey_questions if q.type == QuestionType.LIKERT]
        
        if len(likert_questions) == 0:
            raise HTTPException(status_code=400, detail="No Likert questions found")
        
        # Create visualization
        n_questions = len(likert_questions)
        n_cols = min(3, n_questions)
        n_rows = (n_questions + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 4*n_rows))
        if n_rows == 1 and n_cols == 1:
            axes = np.array([axes])
        axes = axes.flatten() if n_rows > 1 or n_cols > 1 else [axes[0]]
        
        # Color map
        archetype_names = sorted(state.simulated_df['archetype_name'].unique())
        colors = plt.cm.Set3(np.linspace(0, 1, len(archetype_names)))
        colors_map = {name: colors[i] for i, name in enumerate(archetype_names)}
        
        for i, q in enumerate(likert_questions):
            ax = axes[i]
            q_id = q.id
            
            if q_id not in state.simulated_df.columns:
                ax.axis('off')
                continue
            
            all_values = pd.to_numeric(state.simulated_df[q_id], errors='coerce').dropna()
            
            if len(all_values) == 0:
                ax.axis('off')
                continue
            
            # Stacked histogram by archetype
            bottom = np.zeros(5)
            
            for arch_name in archetype_names:
                arch_data = state.simulated_df[state.simulated_df['archetype_name'] == arch_name]
                values = pd.to_numeric(arch_data[q_id], errors='coerce').dropna()
                
                if len(values) == 0:
                    continue
                
                counts = [int((values == score).sum()) for score in range(1, 6)]
                
                ax.bar(range(1, 6), counts, bottom=bottom,
                      label=arch_name, color=colors_map[arch_name],
                      edgecolor='black', linewidth=0.5, alpha=0.8)
                
                bottom += np.array(counts)
            
            # Mean and std
            mean_val = all_values.mean()
            std_val = all_values.std()
            
            ax.axvline(mean_val, color='red', linestyle='--', linewidth=2,
                      label=f'Mean: {mean_val:.2f}', zorder=10)
            ax.axvspan(mean_val - std_val, mean_val + std_val,
                      alpha=0.15, color='red', zorder=5)
            
            # Text box
            textstr = f'μ = {mean_val:.2f}\nσ = {std_val:.2f}'
            props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
            ax.text(0.98, 0.98, textstr, transform=ax.transAxes,
                   fontsize=9, verticalalignment='top', horizontalalignment='right',
                   bbox=props)
            
            title_text = q.text[:40] + "..." if len(q.text) > 40 else q.text
            ax.set_xlabel('Response', fontsize=10)
            ax.set_ylabel('Count', fontsize=10)
            ax.set_title(f"{q.id}: {title_text}", fontsize=10, fontweight='bold')
            ax.set_xticks([1, 2, 3, 4, 5])
            ax.set_xlim(0.5, 5.5)
            ax.legend(fontsize=7, loc='upper left')
            ax.grid(True, alpha=0.3, axis='y')
        
        # Hide unused subplots
        for i in range(n_questions, len(axes)):
            axes[i].axis('off')
        
        plt.tight_layout()
        
        # Save to bytes
        from io import BytesIO
        buf = BytesIO()
        plt.savefig(buf, format='png', dpi=300, bbox_inches='tight')
        buf.seek(0)
        plt.close()
        
        return StreamingResponse(buf, media_type="image/png")
    
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

# ============================================================================
# DOWNLOAD ENDPOINTS
# ============================================================================

@app.get("/api/download/simulated-survey")
async def download_simulated_survey():
    """Download simulated survey CSV"""
    try:
        filepath = PATHS.OUTPUT_DIR / "simulated_survey.csv"
        if not filepath.exists():
            raise HTTPException(status_code=404, detail="File not found")
        
        return FileResponse(filepath, media_type="text/csv", filename="simulated_survey.csv")
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/download/calibration")
async def download_calibration():
    """Download calibration data CSV"""
    try:
        filepath = PATHS.OUTPUT_DIR / "calibration_data.csv"
        if not filepath.exists():
            raise HTTPException(status_code=404, detail="File not found")
        
        return FileResponse(filepath, media_type="text/csv", filename="calibration_data.csv")
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/download/personas")
async def download_personas():
    """Download personas JSON"""
    try:
        filepath = PATHS.PERSONAS_JSON
        if not filepath.exists():
            raise HTTPException(status_code=404, detail="File not found")
        
        return FileResponse(filepath, media_type="application/json", filename="personas.json")
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    print("\n" + "="*80)
    print("🚀 Starting Survey Archetypes API Server v2")
    print("="*80)
    print(f"\n📍 API URL: http://localhost:8000")
    print(f"📍 Docs: http://localhost:8000/docs")
    print(f"📍 Frontend: http://localhost:5173")
    print("\n" + "="*80 + "\n")
    
    uvicorn.run(app, host="0.0.0.0", port=8000)