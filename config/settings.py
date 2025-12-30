"""
config/settings.py

Central configuration for all system parameters, paths, and model settings.
"""

from pathlib import Path
from dataclasses import dataclass
from typing import Dict

# ============================================================================
# PATHS
# ============================================================================

@dataclass
class Paths:
    """All file paths used in the project"""
    
    # Base directories
    BASE_DIR: Path = Path(__file__).parent.parent
    DATA_DIR: Path = BASE_DIR / "data"
    
    # Input/Output
    INPUT_DIR: Path = DATA_DIR / "input"
    PERSONAS_DIR: Path = DATA_DIR / "personas"
    OUTPUT_DIR: Path = DATA_DIR / "output"
    PLOTS_DIR: Path = OUTPUT_DIR / "plots"
    
    # Specific files
    ORIGINAL_SURVEY: Path = INPUT_DIR / "original_survey.csv"
    PERSONAS_JSON: Path = PERSONAS_DIR / "personas.json"
    SIMULATED_SURVEY: Path = OUTPUT_DIR / "simulated_survey.csv"
    ARCHETYPE_DISTRIBUTIONS: Path = OUTPUT_DIR / "archetype_distributions.csv"
    VALIDATION_REPORT: Path = OUTPUT_DIR / "validation_report.json"
    
    def ensure_dirs(self):
        """Create all necessary directories"""
        for dir_path in [
            self.INPUT_DIR, 
            self.PERSONAS_DIR, 
            self.OUTPUT_DIR, 
            self.PLOTS_DIR
        ]:
            dir_path.mkdir(parents=True, exist_ok=True)

PATHS = Paths()

# ============================================================================
# OLLAMA SETTINGS
# ============================================================================

class OllamaConfig:
    """Configuration for Ollama LLM"""
    
    MODEL: str = "gemma3:4b"
    URL: str = "http://localhost:11434"
    TEMPERATURE_PERSONA: float = 0.7
    TEMPERATURE_AGENT: float = 0.5
    TIMEOUT: int = 120  # seconds
    MAX_RETRIES: int = 3

OLLAMA = OllamaConfig()

# ============================================================================
# ANALYSIS PARAMETERS
# ============================================================================

class AnalysisConfig:
    """Parameters for archetypal analysis"""
    
    # Archetypal Analysis
    N_ARCHETYPES: int = 4  # Default number of archetypes
    MAX_ARCHETYPES_TO_TEST: int = 8  # Maximum k to test
    TARGET_R2: float = 0.80  # Target explained variance
    MAX_ITERATIONS: int = 5000  # NMF/PCHA max iterations
    RANDOM_SEED: int = 42
    
    # Validation
    N_BOOTSTRAP_SAMPLES: int = 1000
    CONFIDENCE_LEVEL: float = 0.95
    MIN_PERSONA_DIVERSITY: float = 0.6

ANALYSIS = AnalysisConfig()

# ============================================================================
# SIMULATION PARAMETERS
# ============================================================================

class SimulationConfig:
    """Parameters for population simulation"""
    
    N_CALIBRATION_SAMPLES: int = 10  # How many times each agent answers
    N_SIMULATED_RESPONDENTS: int = 1000  # Population size
    USE_EXISTING_PERSONAS: bool = True  # Load vs generate
    STRATIFIED_SAMPLING: bool = False  # Exact archetype proportions

SIMULATION = SimulationConfig()

# ============================================================================
# DATA GENERATION PARAMETERS
# ============================================================================

class DataGenerationConfig:
    """Parameters for synthetic data generation"""
    
    N_RESPONDENTS: int = 200
    MISSING_RATE: float = 0.03  # 3% missing data
    DEMOGRAPHIC_CONTEXT: str = "University Students in New York"

DATA_GEN = DataGenerationConfig()

# ============================================================================
# VISUALIZATION SETTINGS
# ============================================================================

class VisualizationConfig:
    """Settings for plots and figures"""
    
    DPI: int = 300
    FIGURE_FORMAT: str = "png"
    STYLE: str = "seaborn-v0_8-darkgrid"
    COLOR_PALETTE: str = "husl"
    N_COLORS: int = 8

VIZ = VisualizationConfig()

# ============================================================================
# EXPORT CONFIGURATION
# ============================================================================

def get_config_dict() -> Dict:
    """Export all configuration as dictionary"""
    return {
        'ollama': {
            'model': OLLAMA.MODEL,
            'url': OLLAMA.URL,
            'temperature_persona': OLLAMA.TEMPERATURE_PERSONA,
            'temperature_agent': OLLAMA.TEMPERATURE_AGENT,
        },
        'analysis': {
            'n_archetypes': ANALYSIS.N_ARCHETYPES,
            'target_r2': ANALYSIS.TARGET_R2,
            'random_seed': ANALYSIS.RANDOM_SEED,
        },
        'simulation': {
            'n_calibration_samples': SIMULATION.N_CALIBRATION_SAMPLES,
            'n_simulated_respondents': SIMULATION.N_SIMULATED_RESPONDENTS,
        },
        'data_generation': {
            'n_respondents': DATA_GEN.N_RESPONDENTS,
            'demographic_context': DATA_GEN.DEMOGRAPHIC_CONTEXT,
        }
    }

# ============================================================================
# INITIALIZATION
# ============================================================================

def initialize():
    """Initialize configuration (create directories, etc.)"""
    PATHS.ensure_dirs()
    print("✅ Configuration initialized")
    print(f"   📁 Base directory: {PATHS.BASE_DIR}")
    print(f"   📁 Data directory: {PATHS.DATA_DIR}")
    print(f"   🤖 Ollama model: {OLLAMA.MODEL}")
    print(f"   🎯 Target R²: {ANALYSIS.TARGET_R2:.0%}")

if __name__ == "__main__":
    initialize()