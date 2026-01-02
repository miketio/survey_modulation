"""
config/settings.py

Central configuration for all system parameters, paths, and model settings.
Now loads from JSON configuration files via loader.
"""

from pathlib import Path
from dataclasses import dataclass
from typing import Dict

from config.loader import load_system_config

# ============================================================================
# PATHS
# ============================================================================

@dataclass
class Paths:
    """All file paths used in the project"""
    
    # Base directories
    BASE_DIR: Path = Path(__file__).parent.parent
    DATA_DIR: Path = BASE_DIR / "data"
    
    # Config directories (NEW)
    CONFIG_DIR: Path = DATA_DIR / "config"
    QUESTIONS_DIR: Path = CONFIG_DIR / "questions"
    ARCHETYPES_DIR: Path = CONFIG_DIR / "archetypes"
    PERSONAS_DIR: Path = CONFIG_DIR / "personas"
    SCHEMA_DIR: Path = CONFIG_DIR / "schema"
    
    # Input/Output
    INPUT_DIR: Path = DATA_DIR / "input"
    OUTPUT_DIR: Path = DATA_DIR / "output"
    PLOTS_DIR: Path = OUTPUT_DIR / "plots"
    
    # Specific files
    ORIGINAL_SURVEY: Path = INPUT_DIR / "original_survey.csv"
    PERSONAS_JSON: Path = PERSONAS_DIR / "generated_personas.json"
    SIMULATED_SURVEY: Path = OUTPUT_DIR / "simulated_survey.csv"
    ARCHETYPE_DISTRIBUTIONS: Path = OUTPUT_DIR / "archetype_distributions.csv"
    VALIDATION_REPORT: Path = OUTPUT_DIR / "validation_report.json"
    SYSTEM_CONFIG_JSON: Path = CONFIG_DIR / "system_config.json"
    
    def ensure_dirs(self):
        """Create all necessary directories"""
        for dir_path in [
            self.INPUT_DIR,
            self.CONFIG_DIR,
            self.QUESTIONS_DIR,
            self.ARCHETYPES_DIR,
            self.ARCHETYPES_DIR / "discovered",
            self.PERSONAS_DIR,
            self.SCHEMA_DIR,
            self.OUTPUT_DIR,
            self.PLOTS_DIR
        ]:
            dir_path.mkdir(parents=True, exist_ok=True)

PATHS = Paths()

# ============================================================================
# LOAD CONFIGURATION FROM JSON
# ============================================================================

# Load system config from JSON
_system_config = load_system_config()

# ============================================================================
# OLLAMA SETTINGS
# ============================================================================

class OllamaConfig:
    """Configuration for Ollama LLM - loaded from JSON"""
    
    def __init__(self, config_dict: Dict):
        self.MODEL: str = config_dict.get('model', 'gemma3:4b')
        self.URL: str = config_dict.get('url', 'http://localhost:11434')
        self.TEMPERATURE_PERSONA: float = config_dict.get('temperature_persona', 0.7)
        self.TEMPERATURE_AGENT: float = config_dict.get('temperature_agent', 0.5)
        self.TIMEOUT: int = config_dict.get('timeout', 120)
        self.MAX_RETRIES: int = config_dict.get('max_retries', 3)

OLLAMA = OllamaConfig(_system_config.get('ollama', {}))

# ============================================================================
# ANALYSIS PARAMETERS
# ============================================================================

class AnalysisConfig:
    """Parameters for archetypal analysis - loaded from JSON"""
    
    def __init__(self, config_dict: Dict):
        self.N_ARCHETYPES: int = config_dict.get('n_archetypes', 4)
        self.MAX_ARCHETYPES_TO_TEST: int = config_dict.get('max_archetypes_to_test', 8)
        self.TARGET_R2: float = config_dict.get('target_r2', 0.80)
        self.MAX_ITERATIONS: int = config_dict.get('max_iterations', 5000)
        self.RANDOM_SEED: int = config_dict.get('random_seed', 42)
        
        # Validation
        self.N_BOOTSTRAP_SAMPLES: int = config_dict.get('n_bootstrap_samples', 1000)
        self.CONFIDENCE_LEVEL: float = config_dict.get('confidence_level', 0.95)
        self.MIN_PERSONA_DIVERSITY: float = config_dict.get('min_persona_diversity', 0.6)

ANALYSIS = AnalysisConfig(_system_config.get('analysis', {}))

# ============================================================================
# SIMULATION PARAMETERS
# ============================================================================

class SimulationConfig:
    """Parameters for population simulation - loaded from JSON"""
    
    def __init__(self, config_dict: Dict):
        self.N_CALIBRATION_SAMPLES: int = config_dict.get('n_calibration_samples', 10)
        self.N_SIMULATED_RESPONDENTS: int = config_dict.get('n_simulated_respondents', 1000)
        self.USE_EXISTING_PERSONAS: bool = config_dict.get('use_existing_personas', True)
        self.STRATIFIED_SAMPLING: bool = config_dict.get('stratified_sampling', False)

SIMULATION = SimulationConfig(_system_config.get('simulation', {}))

# ============================================================================
# DATA GENERATION PARAMETERS
# ============================================================================

class DataGenerationConfig:
    """Parameters for synthetic data generation - loaded from JSON"""
    
    def __init__(self, config_dict: Dict):
        self.N_RESPONDENTS: int = config_dict.get('n_respondents', 200)
        self.MISSING_RATE: float = config_dict.get('missing_rate', 0.0)
        self.DEMOGRAPHIC_CONTEXT: str = config_dict.get(
            'demographic_context', 
            'University Students in New York'
        )

DATA_GEN = DataGenerationConfig(_system_config.get('data_generation', {}))

# ============================================================================
# VISUALIZATION SETTINGS
# ============================================================================

class VisualizationConfig:
    """Settings for plots and figures - loaded from JSON"""
    
    def __init__(self, config_dict: Dict):
        self.DPI: int = config_dict.get('dpi', 300)
        self.FIGURE_FORMAT: str = config_dict.get('figure_format', 'png')
        self.STYLE: str = config_dict.get('style', 'seaborn-v0_8-darkgrid')
        self.COLOR_PALETTE: str = config_dict.get('color_palette', 'husl')
        self.N_COLORS: int = config_dict.get('n_colors', 8)

VIZ = VisualizationConfig(_system_config.get('visualization', {}))

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
            'timeout': OLLAMA.TIMEOUT,
            'max_retries': OLLAMA.MAX_RETRIES,
        },
        'analysis': {
            'n_archetypes': ANALYSIS.N_ARCHETYPES,
            'target_r2': ANALYSIS.TARGET_R2,
            'random_seed': ANALYSIS.RANDOM_SEED,
            'max_archetypes_to_test': ANALYSIS.MAX_ARCHETYPES_TO_TEST,
            'max_iterations': ANALYSIS.MAX_ITERATIONS,
        },
        'simulation': {
            'n_calibration_samples': SIMULATION.N_CALIBRATION_SAMPLES,
            'n_simulated_respondents': SIMULATION.N_SIMULATED_RESPONDENTS,
            'use_existing_personas': SIMULATION.USE_EXISTING_PERSONAS,
            'stratified_sampling': SIMULATION.STRATIFIED_SAMPLING,
        },
        'data_generation': {
            'n_respondents': DATA_GEN.N_RESPONDENTS,
            'missing_rate': DATA_GEN.MISSING_RATE,
            'demographic_context': DATA_GEN.DEMOGRAPHIC_CONTEXT,
        },
        'visualization': {
            'dpi': VIZ.DPI,
            'figure_format': VIZ.FIGURE_FORMAT,
            'style': VIZ.STYLE,
            'color_palette': VIZ.COLOR_PALETTE,
            'n_colors': VIZ.N_COLORS,
        }
    }

# ============================================================================
# RELOAD CONFIGURATION
# ============================================================================

def reload_config():
    """
    Reload configuration from JSON files.
    Call this if you've updated system_config.json.
    """
    global OLLAMA, ANALYSIS, SIMULATION, DATA_GEN, VIZ, _system_config
    
    # Clear loader cache
    from config.loader import get_loader
    get_loader().clear_cache()
    
    # Reload config
    _system_config = load_system_config()
    
    # Reinitialize config objects
    OLLAMA = OllamaConfig(_system_config.get('ollama', {}))
    ANALYSIS = AnalysisConfig(_system_config.get('analysis', {}))
    SIMULATION = SimulationConfig(_system_config.get('simulation', {}))
    DATA_GEN = DataGenerationConfig(_system_config.get('data_generation', {}))
    VIZ = VisualizationConfig(_system_config.get('visualization', {}))
    
    print("✅ Configuration reloaded from JSON")

# ============================================================================
# INITIALIZATION
# ============================================================================

def initialize():
    """Initialize configuration (create directories, etc.)"""
    PATHS.ensure_dirs()
    print("✅ Configuration initialized")
    print(f"   📁 Base directory: {PATHS.BASE_DIR}")
    print(f"   📁 Data directory: {PATHS.DATA_DIR}")
    print(f"   📁 Config directory: {PATHS.CONFIG_DIR}")
    print(f"   🤖 Ollama model: {OLLAMA.MODEL}")
    print(f"   🎯 Target R²: {ANALYSIS.TARGET_R2:.0%}")

if __name__ == "__main__":
    initialize()
    
    print("\n" + "="*80)
    print("📋 CURRENT CONFIGURATION")
    print("="*80)
    
    config = get_config_dict()
    
    for section, values in config.items():
        print(f"\n{section.upper()}:")
        for key, value in values.items():
            print(f"  • {key}: {value}")
    
    print("\n" + "="*80)
    print("✅ Configuration loaded successfully from JSON")
    print("="*80 + "\n")