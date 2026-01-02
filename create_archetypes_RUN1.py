"""
main.py

Complete survey archetypes pipeline orchestration.
Runs the full workflow from data generation to visualization.
"""

import numpy as np
from pathlib import Path

# Configuration
from config.settings import PATHS, ANALYSIS, DATA_GEN, initialize
from config.questions import get_opinion_questions, get_demographic_questions, questions_to_dict_list

# Core
from core.archetypal_analyzer import ArchetypalAnalyzer
from core.encoding import SurveyEncoder

# Generators
from generators.survey_data_generator import SurveyDataGenerator
from generators.persona_generator import PersonaGenerator

# Analysis
from analysis.visualization import SurveyVisualizer

# Utils
from utils.ollama_client import test_ollama_connection
from utils.file_io import save_personas, save_dataframe, save_json


def main():
    """Run complete pipeline"""
    
    print("="*80)
    print(" 🚀  SURVEY ARCHETYPES COMPLETE PIPELINE")
    print("="*80)
    
    # Initialize
    initialize()
    
    # Step 0: Test Ollama connection
    print("\n" + "="*80)
    print("[STEP 0] Testing Ollama Connection")
    print("="*80)
    
    ollama_available = test_ollama_connection()
    if not ollama_available:
        print("⚠️  Warning: Ollama not available. Persona generation will use fallbacks.")
        response = input("Continue anyway? (y/n): ")
        if response.lower() != 'y':
            print("Exiting...")
            return
    
    # Step 1: Generate Data
    print("\n" + "="*80)
    print("[STEP 1] Generating Synthetic Survey Data")
    print("="*80)
    
    generator = SurveyDataGenerator(seed=ANALYSIS.RANDOM_SEED)
    df = generator.generate(n_respondents=DATA_GEN.N_RESPONDENTS)
    
    print(f"✅ Generated {len(df)} respondents")
    print(f"   Columns: {len(df.columns)}")
    
    # Save raw data
    save_dataframe(df, filename="generated_survey.csv")
    
    # Step 2: Encode for Analysis
    print("\n" + "="*80)
    print("[STEP 2] Encoding Data for Analysis")
    print("="*80)
    
    encoder = SurveyEncoder()
    opinion_questions = get_opinion_questions()
    encoder.fit(opinion_questions)
    
    opinion_data = encoder.encode_dataframe(df, opinion_questions, use_opinion_only=True)
    
    print(f"✅ Encoded matrix shape: {opinion_data.shape}")
    
    # Step 3: Archetypal Analysis
    print("\n" + "="*80)
    print("[STEP 3] Performing Archetypal Analysis")
    print("="*80)
    
    analyzer = ArchetypalAnalyzer(random_state=ANALYSIS.RANDOM_SEED)
    
    # Find optimal k
    optimal_k = analyzer.find_optimal_k(
        opinion_data,
        target_r2=ANALYSIS.TARGET_R2,
        max_k=ANALYSIS.MAX_ARCHETYPES_TO_TEST
    )
    
    analyzer.n_archetypes = optimal_k
    
    # Fit model
    archetypes, weights, proportions = analyzer.fit(opinion_data, method='auto', verbose=True)
    
    print(f"✅ Found {optimal_k} archetypes (R²: {analyzer.get_total_r2():.1%})")
    
    # Decode archetypes
    decoded_archetypes = []
    for archetype in archetypes:
        decoded = encoder.decode_archetype(archetype, opinion_questions, use_opinion_only=True)
        decoded_archetypes.append(decoded)
    
    # Step 4: Generate Personas
    print("\n" + "="*80)
    print("[STEP 4] Generating AI Personas")
    print("="*80)
    
    persona_gen = PersonaGenerator(demographic_context=DATA_GEN.DEMOGRAPHIC_CONTEXT)
    
    personas = persona_gen.generate_batch(
        archetypes=archetypes,
        questions=opinion_questions,
        weights=proportions,
        r2_scores=analyzer.get_r2_scores()
    )
    
    # Add decoded answers to personas
    for i, persona in enumerate(personas):
        persona['decoded_answers'] = decoded_archetypes[i]
    
    # Save personas
    persona_gen.save()
    print(f"✅ Generated and saved {len(personas)} personas")
    
    # Step 5: Visualization
    print("\n" + "="*80)
    print("[STEP 5] Creating Visualizations")
    print("="*80)
    
    viz = SurveyVisualizer()
    
    all_questions = generator.get_questions()
    
    # Plot 1: Input data distributions
    viz.plot_response_distributions(df, all_questions, filename="1_input_data_distributions.png")
    
    # Plot 2: Archetype patterns
    viz.plot_archetype_patterns(
        archetypes,
        decoded_archetypes,
        opinion_questions,
        proportions,
        personas,
        filename="2_archetype_patterns.png"
    )
    
    # Step 6: Save Summary
    print("\n" + "="*80)
    print("[STEP 6] Saving Analysis Summary")
    print("="*80)
    
    summary = {
        'configuration': {
            'demographic_context': DATA_GEN.DEMOGRAPHIC_CONTEXT,
            'n_respondents': DATA_GEN.N_RESPONDENTS,
            'n_archetypes': optimal_k,
            'target_r2': ANALYSIS.TARGET_R2,
            'random_seed': ANALYSIS.RANDOM_SEED
        },
        'results': {
            'method_used': analyzer.method_used,
            'total_r2': float(analyzer.get_total_r2()),
            'archetype_proportions': [float(p) for p in proportions],
            'r2_scores': [float(r) for r in analyzer.get_r2_scores()]
        },
        'archetypes': [
            {
                'index': i,
                'name': personas[i]['name'],
                'weight': float(proportions[i]),
                'r2': float(analyzer.get_r2_scores()[i]),
                'pattern': decoded_archetypes[i]
            }
            for i in range(len(archetypes))
        ]
    }
    
    save_json(summary, filename="analysis_summary.json")
    
    # Final Summary
    print("\n" + "="*80)
    print("✅ PIPELINE COMPLETE!")
    print("="*80)
    
    print(f"\n📊 Results Summary:")
    print(f"   • Method: {analyzer.method_used}")
    print(f"   • Archetypes: {optimal_k}")
    print(f"   • R²: {analyzer.get_total_r2():.1%}")
    print(f"   • Demographic: {DATA_GEN.DEMOGRAPHIC_CONTEXT}")
    
    print(f"\n📁 Generated Files:")
    print(f"   • {PATHS.OUTPUT_DIR}/generated_survey.csv")
    print(f"   • {PATHS.PERSONAS_DIR}/personas.json")
    print(f"   • {PATHS.OUTPUT_DIR}/analysis_summary.json")
    print(f"   • {PATHS.PLOTS_DIR}/1_input_data_distributions.png")
    print(f"   • {PATHS.PLOTS_DIR}/2_archetype_patterns.png")
    
    print(f"\n🎯 Next Steps:")
    print(f"   1. Review personas in: {PATHS.PERSONAS_DIR}/personas.json")
    print(f"   2. Run simulation: python scripts/run_simulation.py")
    print(f"   3. Or explore with your own questions")
    
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