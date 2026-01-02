"""
migrate_to_json.py

Migrate existing Python configurations to JSON format.
Run this once to create initial JSON config files.
"""

import json
from pathlib import Path
from typing import List, Dict


def migrate_questions():
    """Migrate questions from config/questions.py to JSON"""
    from config.questions import (
        OPINION_QUESTIONS, DEMOGRAPHIC_QUESTIONS, 
        SECOND_SURVEY_QUESTIONS, Question
    )
    
    config_dir = Path("data/config/questions")
    config_dir.mkdir(parents=True, exist_ok=True)
    
    def questions_to_json(questions: List[Question]) -> List[Dict]:
        result = []
        for q in questions:
            q_dict = {
                'id': q.id,
                'text': q.text,
                'type': q.type.value,
                'category': q.category
            }
            if q.scale:
                q_dict['scale'] = list(q.scale)
            if q.options:
                q_dict['options'] = q.options
            result.append(q_dict)
        return result
    
    # Opinion Survey
    opinion_data = {
        "name": "Opinion Survey (Default)",
        "description": "Standard 5-question opinion survey for archetypal analysis",
        "version": "1.0",
        "questions": questions_to_json(OPINION_QUESTIONS)
    }
    
    with open(config_dir / "opinion_survey.json", 'w') as f:
        json.dump(opinion_data, f, indent=2)
    print(f"✅ Created: {config_dir}/opinion_survey.json")
    
    # Demographics
    demo_data = {
        "name": "Demographics",
        "description": "Standard demographic questions",
        "version": "1.0",
        "questions": questions_to_json(DEMOGRAPHIC_QUESTIONS)
    }
    
    with open(config_dir / "demographics.json", 'w') as f:
        json.dump(demo_data, f, indent=2)
    print(f"✅ Created: {config_dir}/demographics.json")
    
    # Validation Survey
    validation_data = {
        "name": "Validation Questions",
        "description": "Second survey questions for validation",
        "version": "1.0",
        "questions": questions_to_json(SECOND_SURVEY_QUESTIONS)
    }
    
    with open(config_dir / "validation_survey.json", 'w') as f:
        json.dump(validation_data, f, indent=2)
    print(f"✅ Created: {config_dir}/validation_survey.json")


def migrate_archetypes():
    """Migrate archetypes from survey_data_generator.py to JSON"""
    from generators.survey_data_generator import SurveyDataGenerator
    
    config_dir = Path("data/config/archetypes")
    config_dir.mkdir(parents=True, exist_ok=True)
    
    # Create a generator to get archetypes
    generator = SurveyDataGenerator()
    true_archetypes = generator.true_archetypes
    
    # Convert to JSON-serializable format
    archetypes_list = []
    for arch in true_archetypes:
        arch_dict = {
            'name': arch['name'],
            'description': f"Archetype: {arch['name']}",
            'opinion_pattern': arch['opinion_pattern'],
            'weight': arch['weight'],
            'variance': arch['variance'],
            'demographic_pattern': arch.get('demographic_pattern', {})
        }
        archetypes_list.append(arch_dict)
    
    data = {
        "name": "Default Archetypes",
        "description": "Standard 4-archetype configuration for survey generation",
        "version": "1.0",
        "demographic_context": "University Students in New York",
        "archetypes": archetypes_list
    }
    
    with open(config_dir / "default.json", 'w') as f:
        json.dump(data, f, indent=2)
    print(f"✅ Created: {config_dir}/default.json")


def migrate_system_config():
    """Migrate system config from settings.py to JSON"""
    from config.settings import OLLAMA, ANALYSIS, SIMULATION, DATA_GEN, VIZ
    
    config_dir = Path("data/config")
    config_dir.mkdir(parents=True, exist_ok=True)
    
    config = {
        "version": "1.0",
        "ollama": {
            "model": OLLAMA.MODEL,
            "url": OLLAMA.URL,
            "temperature_persona": OLLAMA.TEMPERATURE_PERSONA,
            "temperature_agent": OLLAMA.TEMPERATURE_AGENT,
            "timeout": OLLAMA.TIMEOUT,
            "max_retries": OLLAMA.MAX_RETRIES
        },
        "analysis": {
            "n_archetypes": ANALYSIS.N_ARCHETYPES,
            "target_r2": ANALYSIS.TARGET_R2,
            "random_seed": ANALYSIS.RANDOM_SEED,
            "max_archetypes_to_test": ANALYSIS.MAX_ARCHETYPES_TO_TEST,
            "max_iterations": ANALYSIS.MAX_ITERATIONS
        },
        "simulation": {
            "n_calibration_samples": SIMULATION.N_CALIBRATION_SAMPLES,
            "n_simulated_respondents": SIMULATION.N_SIMULATED_RESPONDENTS,
            "stratified_sampling": SIMULATION.STRATIFIED_SAMPLING,
            "use_existing_personas": SIMULATION.USE_EXISTING_PERSONAS
        },
        "data_generation": {
            "n_respondents": DATA_GEN.N_RESPONDENTS,
            "missing_rate": DATA_GEN.MISSING_RATE,
            "demographic_context": DATA_GEN.DEMOGRAPHIC_CONTEXT
        },
        "visualization": {
            "dpi": VIZ.DPI,
            "figure_format": VIZ.FIGURE_FORMAT,
            "style": VIZ.STYLE,
            "color_palette": VIZ.COLOR_PALETTE,
            "n_colors": VIZ.N_COLORS
        }
    }
    
    with open(config_dir / "system_config.json", 'w') as f:
        json.dump(config, f, indent=2)
    print(f"✅ Created: {config_dir}/system_config.json")


def create_directory_structure():
    """Create necessary directory structure"""
    dirs = [
        "data/config",
        "data/config/questions",
        "data/config/archetypes",
        "data/config/archetypes/discovered",
        "data/config/personas",
        "data/config/schema"
    ]
    
    for dir_path in dirs:
        Path(dir_path).mkdir(parents=True, exist_ok=True)
    
    print(f"✅ Created directory structure")


def create_schemas():
    """Create JSON schemas for validation"""
    schema_dir = Path("data/config/schema")
    schema_dir.mkdir(parents=True, exist_ok=True)
    
    # Question schema
    question_schema = {
        "$schema": "http://json-schema.org/draft-07/schema#",
        "type": "object",
        "required": ["name", "version", "questions"],
        "properties": {
            "name": {"type": "string"},
            "description": {"type": "string"},
            "version": {"type": "string"},
            "questions": {
                "type": "array",
                "items": {
                    "type": "object",
                    "required": ["id", "text", "type", "category"],
                    "properties": {
                        "id": {"type": "string"},
                        "text": {"type": "string"},
                        "type": {"enum": ["likert", "categorical", "ordinal"]},
                        "category": {"enum": ["opinion", "demographic"]},
                        "scale": {"type": "array", "items": {"type": "number"}},
                        "options": {"type": "array", "items": {"type": "string"}}
                    }
                }
            }
        }
    }
    
    with open(schema_dir / "question_schema.json", 'w') as f:
        json.dump(question_schema, f, indent=2)
    print(f"✅ Created: {schema_dir}/question_schema.json")
    
    # Archetype schema
    archetype_schema = {
        "$schema": "http://json-schema.org/draft-07/schema#",
        "type": "object",
        "required": ["name", "version", "archetypes"],
        "properties": {
            "name": {"type": "string"},
            "description": {"type": "string"},
            "version": {"type": "string"},
            "demographic_context": {"type": "string"},
            "archetypes": {
                "type": "array",
                "items": {
                    "type": "object",
                    "required": ["name", "opinion_pattern", "weight"],
                    "properties": {
                        "name": {"type": "string"},
                        "description": {"type": "string"},
                        "opinion_pattern": {"type": "array", "items": {"type": "number"}},
                        "weight": {"type": "number", "minimum": 0, "maximum": 1},
                        "variance": {"type": "object"},
                        "demographic_pattern": {"type": "object"}
                    }
                }
            }
        }
    }
    
    with open(schema_dir / "archetype_schema.json", 'w') as f:
        json.dump(archetype_schema, f, indent=2)
    print(f"✅ Created: {schema_dir}/archetype_schema.json")


def main():
    """Run all migration steps"""
    print("\n" + "="*80)
    print("🔄 MIGRATING TO JSON CONFIGURATION")
    print("="*80 + "\n")
    
    print("Step 1: Creating directory structure...")
    create_directory_structure()
    
    print("\nStep 2: Creating JSON schemas...")
    create_schemas()
    
    print("\nStep 3: Migrating questions...")
    migrate_questions()
    
    print("\nStep 4: Migrating archetypes...")
    migrate_archetypes()
    
    print("\nStep 5: Migrating system config...")
    migrate_system_config()
    
    print("\n" + "="*80)
    print("✅ MIGRATION COMPLETE!")
    print("="*80)
    
    print("\n📋 Summary:")
    print("  • Questions: data/config/questions/")
    print("  • Archetypes: data/config/archetypes/")
    print("  • System Config: data/config/system_config.json")
    print("  • Schemas: data/config/schema/")
    
    print("\n📖 Next Steps:")
    print("  1. Review generated JSON files")
    print("  2. Update config/settings.py to use loader")
    print("  3. Update config/questions.py to use loader")
    print("  4. Update api/server.py to use loader")
    print("  5. Update App.jsx to load from API")
    
    print("\n⚠️  Note: Old Python configs are still in place for backward compatibility")
    print("   You can gradually migrate to using JSON configs only.")
    
    print("\n" + "="*80 + "\n")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n❌ Migration failed: {e}")
        import traceback
        traceback.print_exc()