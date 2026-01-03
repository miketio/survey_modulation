# Survey Archetypes: Transform Data Patterns into AI Personas

> **Discover hidden audience segments in your survey data, then simulate how they'll respond to new questions—without surveying anyone again.**

---

## 🎯 The Problem

You've run a survey and collected responses. Now you want to:

- **Test new questions** without re-surveying everyone
- **Understand audience segments** beyond simple demographics  
- **Predict reactions** to new products, policies, or messaging
- **Generate larger datasets** for statistical validity

Traditional approaches fall short:

❌ **Manual personas are subjective** - "Sarah, 25, tech-savvy" lacks rigor  
❌ **Simple clustering misses nuance** - People aren't just one type  
❌ **Small samples limit analysis** - Hard to detect patterns with 50 respondents

---

## ✨ The Solution

**Survey Archetypes** discovers the fundamental "personality types" hidden in your data, converts them into believable AI personas, then deploys them as intelligent agents that can answer *new* questions while maintaining their worldview.

### What You Get

1. **Data-Driven Personas**: Not guessed—mathematically discovered from response patterns
2. **AI Survey Agents**: Personas that can think and answer new questions consistently  
3. **Synthetic Populations**: Generate 1,000+ responses from a 50-person survey
4. **Mock Survey Mode**: Start with hypothetical archetypes to design and validate surveys

### Perfect For

✅ **Product managers** testing concepts before launch  
✅ **Researchers** expanding small datasets  
✅ **Survey designers** validating instruments before fielding  
✅ **Policy analysts** predicting stakeholder reactions  
✅ **Market researchers** understanding audience segments

---

## 🚀 Quick Start

### Prerequisites & Installation

**1. Install Ollama** (local LLM for persona generation):

```bash
# Visit: https://ollama.ai and download for your OS
# After installation:
ollama pull gemma3:4b
ollama serve  # Keep running in background
```

**2. Install Survey Archetypes:**

```bash
git clone https://github.com/yourusername/survey-archetypes
cd survey-archetypes
pip install -r requirements.txt

# Install frontend
cd frontend
npm install
cd ..
```

### Launch the Application

**Terminal 1 - Backend:**
```bash
python api/server.py
# → Backend running at http://localhost:8000
```

**Terminal 2 - Frontend:**
```bash
cd frontend
npm run dev
# → Frontend at http://localhost:5173
```

**Open browser:** http://localhost:5173

---

## 🔄 Workflow Overview

The web interface guides you through 7 stages:

| Stage | Purpose | Key Actions | Duration |
|-------|---------|-------------|----------|
| **0. Archetypes** | Define or edit initial personality types | Edit patterns, weights, demographic context | 2-5 min |
| **1. Setup** | Configure questions and parameters | Load templates or create custom questions | 3-5 min |
| **2. Discovery** | Find optimal number of archetypes (k) | Run k-analysis (k=2 to k=8), select best k | 1-2 min |
| **3. Personas** | Generate rich AI personas from patterns | Review/edit LLM-generated descriptions | 2-3 min |
| **4. Survey** | Define new questions for validation | Add Likert/categorical/ordinal questions | 2-3 min |
| **5. Calibration** | Agents answer questions multiple times | Watch live as probability distributions build | 1-2 min |
| **6. Simulation** | Generate full synthetic population | Configure size (default: 1,000), run simulation | 30 sec |
| **7. Analysis** | Visualize & export results | Download CSV, generate plots | 1 min |

**Total Time:** ~15-20 minutes from start to synthetic dataset

---

## 🔬 How It Works

### Stage 1: Pattern Discovery (Archetypal Analysis)

**What happens:** Mathematical decomposition finds "pure personality types" in your data.

**The Math:**
```
Survey Data = Mixing Weights × Pure Archetypes

Where:
- Survey Data: How each person answered (n_people × n_questions)
- Mixing Weights: How much of each type is in each person
- Pure Archetypes: The extreme personalities everyone blends from
```

**Why archetypes, not clustering?**

| Method | What It Finds | Example |
|--------|---------------|---------|
| **K-Means Clustering** | Groups of similar people with hard boundaries | "You're in Group A or Group B" |
| **Archetypal Analysis** | Extreme types that everyone is a mixture of | "You're 70% Type A + 30% Type B" |

**Output Example:**
```
Your 200 respondents decompose into:
- Type A: Trust institutions, risk-averse (40% of sample)
- Type B: Tech optimist, environmentalist (30%)  
- Type C: Traditional, skeptical of change (20%)
- Type D: Disengaged, neutral on most topics (10%)

Person #137 = 70% Type A + 30% Type B
```

**Key Insight:** Like primary colors mixing to create all shades—any respondent is a weighted combination of these pure archetypes.

---

### Stage 2: Semantic Translation (LLM Persona Generation)

**What happens:** Local LLM converts mathematical patterns into human narratives.

**Transformation Example:**

```
Mathematical Pattern (Type B):
├─ Q1 (Trust govt): 2/5
├─ Q2 (Tech optimism): 5/5
├─ Q3 (Tradition): 2/5
├─ Q4 (Ecology): 5/5
└─ Q5 (Risk-taking): 4/5

        ↓ LLM Translation ↓

Generated Persona:
"Leo Maxwell, 22
Computer Science Junior at NYU
Values: Innovation, autonomy, environmental sustainability
Fears: Institutional overreach, stagnation, ecological collapse
Worldview: Deeply skeptical of government but optimistic about 
technology solving problems. Believes in taking calculated risks 
for progress while protecting the planet."
```

**Features:**
- ✅ **Contrastive Generation**: Each persona explicitly differs from previous ones
- ✅ **Demographic Constraints**: Enforces realistic age, occupation, location
- ✅ **Explainable**: Shows why this persona gave specific scores
- ✅ **Editable**: Refine manually—system regenerates reasoning automatically

---

### Stage 3: Agent Construction (Survey-Taking AI)

**What happens:** Personas become AI agents that can answer new questions using Chain-of-Thought reasoning.

**Agent Architecture:**

```
┌─────────────────────────────────────┐
│  PERSONA (System Prompt)            │
│  "You are Leo Maxwell, 22...        │
│   Values: Innovation | Fears: ..."  │
└─────────────────────────────────────┘
              ↓
┌─────────────────────────────────────┐
│  NEW QUESTION                        │
│  "Trust social media companies?"    │
└─────────────────────────────────────┘
              ↓
┌─────────────────────────────────────┐
│  CHAIN-OF-THOUGHT REASONING          │
│  "I'm optimistic about tech (Q2=5)  │
│   BUT distrust institutions (Q1=2)  │
│   → Social media = tech + corporate │
│   → Leaning toward skepticism..."   │
└─────────────────────────────────────┘
              ↓
┌─────────────────────────────────────┐
│  ANSWER: 2/5 (Disagree)             │
│  + Explainable reasoning            │
└─────────────────────────────────────┘
```

**Why AI agents vs simple sampling?**

Traditional sampling: `if archetype == "Progressive": answer = random.choice([4,5])`  
❌ Can't handle new questions | ❌ No reasoning | ❌ No consistency

AI Agent approach: Reasons about each question in context  
✅ Answers unseen questions | ✅ Maintains worldview | ✅ Explains logic | ✅ Adapts to phrasing

**Key Design:**
- **Multiple Sampling**: Each agent answers 10+ times per question
- **Calibration**: Builds probability distributions (not single answers)
- **Local LLM**: Privacy-first (Ollama), no data leaves your machine. Can be used as much as you want

---

### Stage 4: Calibration & Simulation

**Calibration:** Agents answer each new question multiple times to capture uncertainty.

```
Leo (Type B) answers "Trust social media?" 10 times:
[2, 2, 3, 2, 2, 1, 2, 3, 2, 2]

Statistics:
- Modal answer: 2 (most common)
- P(answer 1): 10%
- P(answer 2): 70%  ← Most likely
- P(answer 3): 20%
- Mean: 2.1, StdDev: 0.5
```

**Simulation:** Generate full population using calibrated distributions.

```
1. Assign archetypes by weight:
   Respondent #1 → Type A (40% weight)
   Respondent #2 → Type B (30% weight)
   ...

2. Sample responses from calibrated distributions:
   Respondent #2 (Type B) on Q: "Trust social media?"
   → Samples from [10% chance=1, 70% chance=2, 20% chance=3]
   → Gets "2"

3. Result: 1,000 respondents with realistic variance
```

---

### Stage 5: Validation & Export

**Validation Methods:**
- Distribution comparison (simulated vs expected)
- R² scores (how well archetypes explain variance)
- Statistical tests (are type differences significant?)
- Visual plots (side-by-side real vs synthetic)

**Example:**
```
Original Survey (200 people):
  Progressive: 40% | Traditionalist: 20%

Simulated (1,000 people):  
  Progressive: 39.8% | Traditionalist: 20.3%
  
✅ Distributions match within statistical error
✅ R² = 0.87 (strong pattern preservation)
```

**Export:**
- `simulated_survey.csv` - Full synthetic dataset (1,000+ rows)
- `calibration_data.csv` - Probability distributions per agent
- `personas.json` - Complete persona descriptions
- Visualization plots (distribution comparisons)

---

## 📁 Project Structure

```
survey_archetypes/
├── data/
│   ├── config/              # JSON configurations
│   │   ├── questions/       # Question templates
│   │   ├── archetypes/      # Archetype definitions
│   │   └── system_config.json
│   ├── input/               # Your CSV files
│   └── output/              # Results & plots
│
├── config/                  # Python config layer
│   ├── settings.py          # Paths & parameters
│   ├── questions.py         # Question schemas
│   └── loader.py            # JSON loader
│
├── core/                    # Core algorithms
│   ├── archetypal_analyzer.py  # NMF/PCHA decomposition
│   └── encoding.py          # Data type conversion
│
├── generators/              # Creation modules
│   ├── survey_data_generator.py
│   └── persona_generator.py
│
├── agents/                  # AI behavior
│   └── survey_agent.py      # Survey-taking agents
│
├── simulation/              # Population simulation
│   └── population_simulator.py
│
├── analysis/                # Visualization
│   └── visualization.py
│
├── api/                     # Web backend (FastAPI)
│   └── server.py
│
├── frontend/                # React UI
│   └── src/App.jsx
│
└── requirements.txt
```

---

## 📋 Usage Guidelines

### What This Tool CAN Do

✅ Discover mathematically-grounded audience segments  
✅ Simulate responses to new questions within same domain  
✅ Generate statistically valid synthetic datasets  
✅ Test survey designs before fielding  
✅ Scale small datasets (50 → 1,000+)

### What This Tool CANNOT Do

❌ Replace real human insight and qualitative research  
❌ Predict responses to completely unrelated questions  
❌ Capture complex emotions or unstructured feedback  
❌ Guarantee AI reasoning perfectly matches human reasoning  
❌ Work with fewer than ~30-50 initial respondents

### Best Practices

**Data Quality:**
- Minimum 50 respondents recommended
- Include diverse question types (Likert + categorical)
- Ensure questions are clear and well-designed

**Validation:**
- Always validate against held-out real data when possible
- Check archetype distributions make intuitive sense
- Review generated personas for appropriateness

**Usage:**
- Use for hypothesis generation, not final decisions
- Test new questions *related* to original survey domain
- Combine with qualitative research methods
- Don't extrapolate too far from training data

**LLM Settings:**
- Keep Ollama running during all operations
- Larger models (7B+) produce more consistent reasoning
- Default temperature (0.7) balances creativity/consistency
- Chain-of-Thought prompting is crucial for quality

---

## 🔧 Troubleshooting

### Ollama Connection Issues

**Error:** `Connection refused to localhost:11434`

```bash
# Check if running:
ollama list

# Start server:
ollama serve

# Verify model:
ollama pull gemma3:4b
```

### Persona Generation Slow/Fails

**Solutions:**
1. Use smaller model: `ollama pull gemma3:2b`
2. Increase timeout in `data/config/system_config.json`:
   ```json
   "ollama": {"timeout": 300}
   ```
3. Check Ollama logs for errors

### Frontend Can't Connect to Backend

```bash
# 1. Verify backend running:
python api/server.py  # Should show port 8000

# 2. Check API_URL in frontend/src/App.jsx:
const API_URL = 'http://localhost:8000/api';

# 3. Clear browser cache
```

### Synthetic Data Seems Random

**Fixes:**
1. Increase `calibration_samples` to 15-20
2. Review initial archetypes (Tab 0)
3. Lower `temperature_agent` in config
4. Ensure training data has variance

### Python Import Errors

```bash
pip install -r requirements.txt
pip install ollama
```

---

## 🎓 Learn More

### Example Applications

**Market Research:**
- Test product concepts on synthetic consumer panels
- Predict adoption rates by segment before launch

**Public Policy:**
- Simulate stakeholder reactions to regulations
- Identify opposition sources and design communication strategies

**Academic Research:**
- Generate training data for ML models
- Conduct power analysis with synthetic samples
- Validate survey instruments before fielding

### Related Research

- **Cutler & Breiman (1994)**: "Archetypal Analysis" - Original method
- **Wei et al. (2022)**: "Chain-of-Thought Prompting" - CoT reasoning foundation
- **El Emam et al. (2020)**: "Practical Synthetic Data Generation" - Validation best practices

---

## 🤝 Contributing

We welcome contributions! Priority areas:

1. **Better archetypal methods**: Implement true PCHA (not NMF approximation)
2. **Advanced persona generation**: RAG or fine-tuned models
3. **Validation metrics**: More sophisticated similarity measures
4. **Real-world datasets**: Test on diverse survey types
5. **Documentation**: More examples and tutorials

---

## 📄 License

MIT License - Free for research and commercial use.

---

## 📧 Support

**Issues?** 
1. Check Troubleshooting section above
2. Verify Ollama is running: `ollama list`
3. Check configuration: `data/config/system_config.json`
4. Run module tests: `python core/archetypal_analyzer.py`

---

**Ready to discover your hidden audience segments?** 🚀

```bash
python api/server.py  # Start backend
cd frontend && npm run dev  # Start frontend
# Open http://localhost:5173
```