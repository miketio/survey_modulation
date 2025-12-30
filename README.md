# Survey Archetypes: Semantic Decomposition System

**Transform survey data into interpretable AI personas using Archetypal Analysis and Large Language Models.**

---

## 🎯 Overview

This system performs:

1. **Archetypal Analysis**: Decompose survey responses into "pure types" (archetypes)
2. **Semantic Inversion**: Convert mathematical patterns into human personas using LLMs
3. **AI Agents**: Create agents that embody these personas
4. **Population Simulation**: Generate large synthetic populations with realistic variance

Instead of abstract principal components, we extract interpretable personas like:
- *"Progressive Student, 22, values innovation"*
- *"Conservative Professional, 55, values stability"*

---

## 📁 Project Structure

```
survey_archetypes/
│
├── data/                          # All data storage
│   ├── input/                     # Original survey data (CSV)
│   ├── personas/                  # Generated personas (JSON)
│   └── output/                    # Results and visualizations
│       ├── plots/                 # All PNG visualizations
│       ├── generated_survey.csv
│       ├── simulated_survey.csv
│       └── analysis_summary.json
│
├── config/                        # Configuration
│   ├── settings.py                # Paths, Ollama, parameters
│   └── questions.py               # Survey question definitions
│
├── core/                          # Core algorithms
│   ├── archetypal_analyzer.py     # NMF/PCHA decomposition
│   ├── encoding.py                # Type conversion utilities
│   └── validation.py              # Quality metrics (TODO)
│
├── generators/                    # Data creation
│   ├── survey_data_generator.py   # Synthetic survey data
│   └── persona_generator.py       # LLM-based persona creation
│
├── agents/                        # AI behavior
│   └── survey_agent.py            # Survey-taking AI agents
│
├── simulation/                    # Population simulation
│   ├── population_simulator.py    # Main simulation logic
│   └── calibrator.py              # Uncertainty calibration (TODO)
│
├── analysis/                      # Analysis and visualization
│   ├── visualization.py           # All plotting functions
│   └── results_analyzer.py        # Post-simulation analysis (TODO)
│
├── utils/                         # Utilities
│   ├── ollama_client.py           # Ollama connection wrapper
│   └── file_io.py                 # Loading/saving helpers
│
├── scripts/                       # Orchestration scripts
│   └── run_simulation.py          # Population simulation workflow
│
├── tests/                         # Unit tests
│   └── (TODO)
│
├── gui/                           # Future web interface
│   └── (TODO)
│
├── main.py                        # Main pipeline script
├── requirements.txt               # Python dependencies
└── README.md                      # This file
```

---

## 🚀 Quick Start

### 1. Installation

```bash
# Clone repository
cd survey_archetypes

# Install Python dependencies
pip install -r requirements.txt

# Install Ollama (for persona generation)
# Visit: https://ollama.ai
# Then:
ollama pull gemma3:4b
ollama serve
```

### 2. Run Main Pipeline

```bash
python main.py
```

This will:
- Generate synthetic survey data (200 respondents)
- Perform archetypal analysis (find k archetypes)
- Generate AI personas using Ollama
- Create visualizations
- Save results to `data/output/`

### 3. Run Population Simulation

```bash
python scripts/run_simulation.py
```

This will:
- Load personas from main pipeline
- Calibrate response uncertainty (10 samples per agent)
- Simulate large population (1000 respondents)
- Generate analysis plots

---

## 📊 Usage

### Basic Workflow

```python
from config.settings import PATHS, initialize
from generators.survey_data_generator import SurveyDataGenerator
from core.archetypal_analyzer import ArchetypalAnalyzer
from core.encoding import SurveyEncoder
from generators.persona_generator import PersonaGenerator
from agents.survey_agent import SurveyAgent

# Initialize
initialize()

# 1. Generate data
generator = SurveyDataGenerator()
df = generator.generate(n_respondents=200)

# 2. Encode for analysis
encoder = SurveyEncoder()
questions = generator.get_opinion_questions()
encoder.fit(questions)
data = encoder.encode_dataframe(df, questions, use_opinion_only=True)

# 3. Archetypal analysis
analyzer = ArchetypalAnalyzer(n_archetypes=4)
archetypes, weights, proportions = analyzer.fit(data)

# 4. Generate personas
persona_gen = PersonaGenerator()
personas = persona_gen.generate_batch(
    archetypes=archetypes,
    questions=questions,
    weights=proportions,
    r2_scores=analyzer.get_r2_scores()
)

# 5. Create agents
agents = [SurveyAgent(p) for p in personas]

# 6. Run new survey
from config.questions import SECOND_SURVEY_QUESTIONS
results = []
for agent in agents:
    answers = agent.answer_survey(SECOND_SURVEY_QUESTIONS)
    results.append({
        'name': agent.get_name(),
        'weight': agent.get_weight(),
        'answers': answers
    })
```

### Using Real Survey Data

```python
from utils.file_io import load_dataframe
import pandas as pd

# Load your CSV (format: rows=respondents, cols=questions)
df = load_dataframe(filename="your_survey.csv")

# Continue with encoding and analysis
encoder = SurveyEncoder()
# ... rest of workflow
```

---

## 🔧 Configuration

### Paths

Edit `config/settings.py` to change file locations:

```python
class Paths:
    BASE_DIR = Path(__file__).parent.parent
    DATA_DIR = BASE_DIR / "data"
    # ...
```

### Ollama Settings

```python
class OllamaConfig:
    MODEL = "gemma3:4b"
    URL = "http://localhost:11434"
    TEMPERATURE_PERSONA = 0.7
    TEMPERATURE_AGENT = 0.5
```

### Analysis Parameters

```python
class AnalysisConfig:
    N_ARCHETYPES = 4
    TARGET_R2 = 0.80
    RANDOM_SEED = 42
```

### Simulation Parameters

```python
class SimulationConfig:
    N_CALIBRATION_SAMPLES = 10
    N_SIMULATED_RESPONDENTS = 1000
```

---

## 📈 Visualizations

The system generates the following plots (all saved to `data/output/plots/`):

1. **1_input_data_distributions.png** - Original survey response distributions
2. **2_archetype_patterns.png** - Discovered archetype profiles
3. **4_archetype_distributions.png** - Calibrated transition probabilities (Likert only)
4. **5_simulated_population.png** - Simulated population distributions (Likert only)

---

## 🧪 Testing

### Test Individual Modules

Each module has built-in tests:

```bash
# Test encoding
python core/encoding.py

# Test persona generator
python generators/persona_generator.py

# Test survey agent
python agents/survey_agent.py

# Test visualizer
python analysis/visualization.py
```

### Integration Test

The main pipeline serves as an integration test:

```bash
python main.py
```

---

## 🛠 Troubleshooting

### Ollama Connection Error

```
Error: Connection refused to localhost:11434
```

**Solution**:
1. Check Ollama is running: `ollama list`
2. Start Ollama: `ollama serve`
3. Check firewall settings

### Import Errors

```
ModuleNotFoundError: No module named 'sklearn'
```

**Solution**:
```bash
pip install -r requirements.txt
```

### Persona Generation Fails

```
Timeout generating persona
```

**Solution**:
- Increase timeout in `config/settings.py`:
```python
OLLAMA.TIMEOUT = 300  # 5 minutes
```
- Use a smaller model: `ollama pull mistral:7b-instruct`

### Visualization Shows No Categorical Questions

This is by design - only Likert questions (1-5 scale) are visualized in distribution plots because categorical questions don't have meaningful numeric distributions.

---

## 📚 Methodology

### Archetypal Analysis

We use **NMF (Non-negative Matrix Factorization)** or **PCHA (Principal Convex Hull Analysis)**:

```
X ≈ W × H

X: response matrix (n_respondents × n_questions)
W: weights matrix (n_respondents × n_archetypes)
H: archetypes matrix (n_archetypes × n_questions)
```

Benefits over PCA:
- Archetypes are **interpretable** (no negative values)
- Represent **extreme points** in the data cloud
- Any respondent = **weighted combination** of archetypes

### Contrastive Persona Generation

Each persona is generated with awareness of previous personas to ensure distinctness:

```python
Persona_2 = LLM(
    answers_2, 
    "This person DIFFERS from Persona_1"
)
```

### Transition Probabilities

Instead of single-point answers, agents provide probability distributions:

```python
# Agent answers "4" most often, but:
p_lower = 0.20  # 20% chance of answering "3"
p_stay  = 0.60  # 60% chance of answering "4"
p_higher = 0.20  # 20% chance of answering "5"
```

This creates realistic response variance in simulations.

---

## 🎯 Use Cases

1. **Market Research**: Test new product concepts on synthetic consumer segments
2. **Public Policy**: Simulate reactions to proposed laws across demographics
3. **Data Augmentation**: Expand small datasets into larger populations
4. **Method Testing**: Validate clustering algorithms with known ground truth
5. **Survey Design**: Identify which questions discriminate best between groups

---

## 📝 Output Files

After running workflows:

### From main.py:
- `data/output/generated_survey.csv` - Synthetic survey data
- `data/personas/personas.json` - Generated persona descriptions
- `data/output/analysis_summary.json` - Analysis metadata
- `data/output/plots/1_input_data_distributions.png`
- `data/output/plots/2_archetype_patterns.png`

### From run_simulation.py:
- `data/output/archetype_distributions.csv` - Calibration statistics
- `data/output/simulated_survey.csv` - Simulated population (1000+ respondents)
- `data/output/plots/4_archetype_distributions.png`
- `data/output/plots/5_simulated_population.png`

---

## 🔬 Advanced Features

### Custom Questions

Edit `config/questions.py`:

```python
from config.questions import Question, QuestionType

CUSTOM_QUESTIONS = [
    Question(
        id="CQ1",
        text="Your custom question here",
        type=QuestionType.LIKERT,
        category="opinion",
        scale=(1, 5)
    )
]
```

### Demographic Constraints

Change the target population in `config/settings.py`:

```python
class DataGenerationConfig:
    DEMOGRAPHIC_CONTEXT = "Healthcare Workers in Rural Areas"
```

The system will automatically enforce these constraints during persona generation.

---

## 🚧 TODO / Future Work

- [ ] Add comprehensive unit tests (`tests/`)
- [ ] Implement validation framework (`core/validation.py`)
- [ ] Add results analyzer (`analysis/results_analyzer.py`)
- [ ] Create web interface (`gui/`)
- [ ] Support for matrix questions
- [ ] Real-time dashboard
- [ ] API endpoints for integration

---

## 📄 License

MIT License - feel free to use and modify for your research.

---

## 🤝 Contributing

Contributions welcome! Areas for improvement:

1. **Better archetypal methods**: Implement true AA (not NMF approximation)
2. **Advanced persona generation**: Use RAG or fine-tuned models
3. **Validation metrics**: More sophisticated similarity measures
4. **Real-world datasets**: Test on actual survey data
5. **Web interface**: Build interactive dashboard

---

## 📧 Support

For questions or issues:
1. Check the troubleshooting section above
2. Run module tests to isolate the problem
3. Check Ollama connection status
4. Review configuration in `config/settings.py`

---

**Happy Analyzing! 🎉**