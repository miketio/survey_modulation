"""
scripts/run_simulation.py

Complete population simulation workflow.
Calibrates personas and simulates large populations (1000+ respondents).
"""

import numpy as np

# Configuration
from config.settings import PATHS, SIMULATION, initialize
from config.questions import SECOND_SURVEY_QUESTIONS

# Simulation
from simulation.population_simulator import PopulationSimulator

# Analysis
from analysis.visualization import SurveyVisualizer

# Utils
from utils.file_io import load_personas, save_dataframe, check_file_exists
from utils.ollama_client import test_ollama_connection


def load_or_fail_personas():
    """Load personas or exit if not found"""
    
    if not check_file_exists(filepath=PATHS.PERSONAS_JSON):
        print(f"❌ Error: Personas file not found at {PATHS.PERSONAS_JSON}")
        print("\nPlease run main.py first to generate personas:")
        print("   python main.py")
        return None
    
    try:
        personas = load_personas()
        print(f"✅ Loaded {len(personas)} personas from {PATHS.PERSONAS_JSON}")
        
        # Extract proportions
        proportions = np.array([p['weight'] for p in personas])
        proportions = proportions / proportions.sum()
        
        print(f"\nPersona Distribution:")
        for p, prop in zip(personas, proportions):
            print(f"   • {p['name']}: {prop:.1%}")
        
        return personas, proportions
    
    except Exception as e:
        print(f"❌ Error loading personas: {e}")
        return None


def main():
    """Run complete simulation workflow"""
    
    print("""
╔═══════════════════════════════════════════════════════════════════════════╗
║              COMPLETE SURVEY POPULATION SIMULATION                        ║
║                                                                           ║
║  Workflow:                                                                ║
║    1. Load personas from previous analysis                                ║
║    2. Calibrate transition probabilities                                  ║
║    3. Simulate 1000+ respondents                                          ║
║    4. Analyze and visualize results                                       ║
╚═══════════════════════════════════════════════════════════════════════════╝
    """)
    
    # Initialize
    initialize()
    
    # Configuration
    N_CALIBRATION_SAMPLES = SIMULATION.N_CALIBRATION_SAMPLES
    N_SIMULATED_RESPONDENTS = SIMULATION.N_SIMULATED_RESPONDENTS
    
    print(f"\n📋 Configuration:")
    print(f"   • Calibration samples: {N_CALIBRATION_SAMPLES}")
    print(f"   • Simulated respondents: {N_SIMULATED_RESPONDENTS}")
    print(f"   • Test questions: {len(SECOND_SURVEY_QUESTIONS)}")
    
    # Count question types
    from config.questions import QuestionType
    n_likert = sum(1 for q in SECOND_SURVEY_QUESTIONS if q.type == QuestionType.LIKERT)
    n_other = len(SECOND_SURVEY_QUESTIONS) - n_likert
    print(f"      - Likert: {n_likert}")
    print(f"      - Categorical/Ordinal: {n_other}")
    
    # Step 0: Test Ollama
    print("\n" + "="*80)
    print("[STEP 0] Testing Ollama Connection")
    print("="*80)
    
    if not test_ollama_connection():
        print("\n❌ Ollama is required for simulation.")
        print("   Start Ollama with: ollama serve")
        return
    
    # Step 1: Load Personas
    print("\n" + "="*80)
    print("[STEP 1] Loading Personas")
    print("="*80)
    
    result = load_or_fail_personas()
    if result is None:
        return
    
    personas, proportions = result
    
    # Step 2: Create Simulator
    print("\n" + "="*80)
    print("[STEP 2] Initializing Simulator")
    print("="*80)
    
    simulator = PopulationSimulator(
        personas=personas,
        proportions=proportions,
        random_seed=SIMULATION.RANDOM_SEED if hasattr(SIMULATION, 'RANDOM_SEED') else 42
    )
    
    print(f"✅ Simulator initialized with {len(personas)} archetypes")
    
    # Step 3: Calibration
    print("\n" + "="*80)
    print("[STEP 3] Calibrating Response Distributions")
    print("="*80)
    print("   (This involves AI agents answering questions multiple times)")
    
    try:
        distributions = simulator.calibrate(
            SECOND_SURVEY_QUESTIONS,
            n_samples=N_CALIBRATION_SAMPLES,
            verbose=True
        )
        
        # Save calibration results
        summary_df = simulator.get_archetype_summary()
        save_dataframe(summary_df, filename="archetype_distributions.csv")
        print(f"\n✅ Saved: {PATHS.OUTPUT_DIR}/archetype_distributions.csv")
        
        # Check what we got
        print(f"\n📊 Calibration Summary:")
        print(f"   • Total distributions: {len(summary_df)}")
        print(f"   • Likert questions: {(summary_df['question_type'] == 'likert').sum()}")
        print(f"   • Categorical questions: {(summary_df['question_type'] == 'categorical').sum()}")
        
    except Exception as e:
        print(f"\n❌ Calibration failed: {e}")
        print("   This usually means Ollama is not running properly.")
        import traceback
        traceback.print_exc()
        return
    
    # Step 4: Simulate Population
    print("\n" + "="*80)
    print("[STEP 4] Simulating Population")
    print("="*80)
    
    try:
        simulated_df = simulator.simulate_population(
            SECOND_SURVEY_QUESTIONS,
            n_respondents=N_SIMULATED_RESPONDENTS,
            verbose=True
        )
        
        # Save simulated data
        save_dataframe(simulated_df, filename="simulated_survey.csv")
        print(f"\n✅ Saved: {PATHS.OUTPUT_DIR}/simulated_survey.csv")
        
    except Exception as e:
        print(f"\n❌ Simulation failed: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Step 5: Analyze Results
    print("\n" + "="*80)
    print("[STEP 5] Analyzing Simulated Data")
    print("="*80)
    
    print(f"\n📈 Population Statistics:")
    print(f"   • Total respondents: {len(simulated_df)}")
    
    print(f"\n   Archetype distribution:")
    arch_counts = simulated_df['archetype_name'].value_counts()
    for name, count in arch_counts.items():
        print(f"      • {name}: {count} ({count/len(simulated_df):.1%})")
    
    # Response statistics for Likert questions
    from config.questions import QuestionType
    likert_questions = [q for q in SECOND_SURVEY_QUESTIONS if q.type == QuestionType.LIKERT]
    
    if len(likert_questions) > 0:
        print(f"\n   Response statistics (Likert questions):")
        for q in likert_questions[:3]:  # Show first 3
            if q.id in simulated_df.columns:
                values = simulated_df[q.id].dropna()
                if len(values) > 0:
                    mean = values.mean()
                    std = values.std()
                    print(f"      • {q.id}: μ={mean:.2f}, σ={std:.2f}")
    
    # Step 6: Visualizations
    print("\n" + "="*80)
    print("[STEP 6] Creating Visualizations")
    print("="*80)
    
    viz = SurveyVisualizer()
    
    # Plot 1: Archetype distributions (transition probabilities)
    viz.plot_archetype_distributions(
        summary_df,
        filename="4_archetype_distributions.png"
    )
    
    # Plot 2: Simulated population
    viz.plot_simulated_population(
        simulated_df,
        SECOND_SURVEY_QUESTIONS,
        filename="5_simulated_population.png"
    )
    
    # Final Summary
    print("\n" + "="*80)
    print("✅ SIMULATION COMPLETE!")
    print("="*80)
    
    print(f"\n📁 Generated Files:")
    print(f"   • {PATHS.OUTPUT_DIR}/archetype_distributions.csv")
    print(f"   • {PATHS.OUTPUT_DIR}/simulated_survey.csv")
    print(f"   • {PATHS.PLOTS_DIR}/4_archetype_distributions.png")
    print(f"   • {PATHS.PLOTS_DIR}/5_simulated_population.png")
    
    print(f"\n📊 Quick Statistics:")
    print(f"   • Simulated: {len(simulated_df)} respondents")
    print(f"   • Based on: {summary_df['archetype_idx'].nunique()} archetypes")
    print(f"   • Questions: {len(SECOND_SURVEY_QUESTIONS)}")
    
    print(f"\n💡 Next Steps:")
    print(f"   1. Analyze simulated_survey.csv like real survey data")
    print(f"   2. Compare with original archetypes")
    print(f"   3. Use for power analysis or method testing")
    print(f"   4. Adjust N_SIMULATED_RESPONDENTS in config for different sizes")
    
    print("\n" + "="*80 + "\n")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted by user")
    except Exception as e:
        print(f"\n\n❌ Error: {e}")
        import traceback
        traceback.print_exc()