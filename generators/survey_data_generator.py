"""
generators/survey_data_generator.py

Generates synthetic survey data with embedded archetypal patterns.
Now loads archetype definitions from JSON via loader.
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Optional

from config.settings import DATA_GEN, ANALYSIS
from config.questions import (
    Question, QuestionType, 
    get_opinion_questions, get_demographic_questions
)
from config.loader import load_archetypes


class SurveyDataGenerator:
    """
    Generates synthetic survey responses with archetypal structure.
    Archetypes loaded from JSON configuration files.
    """
    
    def __init__(
        self, 
        archetypes: Optional[List[Dict]] = None,
        archetype_name: str = "default",
        seed: int = None
    ):
        """
        Initialize generator.
        
        Args:
            archetypes: Explicit archetype definitions (overrides archetype_name)
            archetype_name: Name of archetype set to load from JSON
            seed: Random seed for reproducibility
        """
        self.seed = seed if seed is not None else getattr(ANALYSIS, 'RANDOM_SEED', 42)
        np.random.seed(self.seed)
        
        self.opinion_questions = get_opinion_questions()
        self.demographic_questions = get_demographic_questions()
        self.all_questions = self.opinion_questions + self.demographic_questions
        
        # Load archetypes from JSON or use provided
        if archetypes is not None:
            self.true_archetypes = archetypes
        else:
            self.true_archetypes = load_archetypes(archetype_name)
        
        # Build complete archetype patterns
        self._build_complete_patterns()
    
    def _build_complete_patterns(self):
        """
        Build complete response patterns for each archetype.
        Maps opinion_pattern to question IDs and fills in demographic patterns.
        """
        for archetype in self.true_archetypes:
            full_pattern = {}
            
            # Map opinion questions
            opinion_pattern = archetype.get('opinion_pattern', [])
            for i, q in enumerate(self.opinion_questions):
                if i < len(opinion_pattern):
                    full_pattern[q.id] = opinion_pattern[i]
                else:
                    full_pattern[q.id] = 3.0  # Default neutral
            
            # Map demographic questions
            demo_pattern = archetype.get('demographic_pattern', {})
            for q in self.demographic_questions:
                q_text_lower = q.text.lower()
                
                # Try to intelligently map demographics
                if 'age' in q_text_lower:
                    full_pattern[q.id] = demo_pattern.get('age', q.options[2] if q.options else None)
                elif 'live' in q_text_lower or 'location' in q_text_lower:
                    full_pattern[q.id] = demo_pattern.get('location', q.options[0] if q.options else None)
                elif 'education' in q_text_lower:
                    full_pattern[q.id] = demo_pattern.get('education', q.options[2] if q.options else None)
                elif 'political' in q_text_lower:
                    full_pattern[q.id] = demo_pattern.get('politics', q.options[2] if q.options else None)
                elif 'news' in q_text_lower:
                    full_pattern[q.id] = demo_pattern.get('news', q.options[2] if q.options else None)
                else:
                    # Default to middle option
                    if q.options:
                        full_pattern[q.id] = q.options[len(q.options)//2]
                    else:
                        full_pattern[q.id] = None
            
            # Store complete pattern
            archetype['pattern'] = full_pattern
            
            # Ensure variance exists
            if 'variance' not in archetype:
                archetype['variance'] = {
                    'likert': 0.8,
                    'categorical': 0.2,
                    'ordinal': 0.15
                }
    
    def generate(
        self, 
        n_respondents: int = None,
        missing_rate: float = None
    ) -> pd.DataFrame:
        """
        Generate synthetic survey responses.
        
        Args:
            n_respondents: Number of respondents (default from config)
            missing_rate: Proportion of missing data (default from config)
        
        Returns:
            DataFrame with responses
        """
        n_respondents = n_respondents if n_respondents is not None else DATA_GEN.N_RESPONDENTS
        missing_rate = missing_rate if missing_rate is not None else DATA_GEN.MISSING_RATE
        
        responses = []
        metadata = []
        
        for archetype_idx, archetype in enumerate(self.true_archetypes):
            n_samples = int(n_respondents * archetype['weight'])
            pattern = archetype['pattern']
            variance = archetype['variance']
            
            for respondent_id in range(n_samples):
                response_dict = {}
                
                for q in self.all_questions:
                    q_id = q.id
                    
                    # Get base value
                    base_value = pattern.get(q_id)
                    if base_value is None:
                        # Default values
                        if q.type == QuestionType.LIKERT:
                            base_value = 3.0
                        else:
                            base_value = q.options[len(q.options)//2] if q.options else None
                    
                    # Add missing data
                    if np.random.random() < missing_rate:
                        response_dict[q_id] = np.nan
                        continue
                    
                    # Generate noisy response based on type
                    if q.type == QuestionType.LIKERT:
                        noise = np.random.randn() * variance['likert']
                        noisy_value = base_value + noise
                        
                        # Response bias for extremes
                        if noisy_value <= 1.5:
                            noisy_value = np.random.choice([1, 2], p=[0.3, 0.7])
                        elif noisy_value >= 4.5:
                            noisy_value = np.random.choice([4, 5], p=[0.7, 0.3])
                        
                        response_dict[q_id] = int(np.clip(np.round(noisy_value), 1, 5))
                    
                    elif q.type == QuestionType.CATEGORICAL:
                        if base_value not in q.options:
                            base_value = q.options[0]
                        
                        if np.random.random() < variance['categorical']:
                            response_dict[q_id] = np.random.choice(q.options)
                        else:
                            response_dict[q_id] = base_value
                    
                    elif q.type == QuestionType.ORDINAL:
                        if base_value not in q.options:
                            base_value = q.options[0]
                        
                        base_idx = q.options.index(base_value)
                        
                        if np.random.random() < variance['ordinal']:
                            shift = np.random.choice([-1, 1])
                            new_idx = np.clip(base_idx + shift, 0, len(q.options) - 1)
                            response_dict[q_id] = q.options[new_idx]
                        else:
                            response_dict[q_id] = base_value
                
                responses.append(response_dict)
                metadata.append({
                    'respondent_id': len(responses),
                    'true_archetype': archetype_idx,
                    'archetype_name': archetype['name']
                })
        
        # Combine
        df_responses = pd.DataFrame(responses)
        df_metadata = pd.DataFrame(metadata)
        
        result = pd.concat([df_metadata, df_responses], axis=1)
        
        return result
    
    def get_questions(self) -> List[Question]:
        """Get all questions"""
        return self.all_questions
    
    def get_opinion_questions(self) -> List[Question]:
        """Get opinion questions"""
        return self.opinion_questions
    
    def get_demographic_questions(self) -> List[Question]:
        """Get demographic questions"""
        return self.demographic_questions
    
    def get_true_archetypes(self) -> List[Dict]:
        """Get ground truth archetypes"""
        return self.true_archetypes
    
    def print_archetype_summary(self):
        """Print summary of loaded archetypes"""
        print("\n" + "="*80)
        print("ARCHETYPE SUMMARY")
        print("="*80 + "\n")
        
        total_weight = sum(a['weight'] for a in self.true_archetypes)
        
        for i, arch in enumerate(self.true_archetypes):
            print(f"Archetype {i+1}: {arch['name']}")
            print(f"  Weight: {arch['weight']:.1%} (normalized: {arch['weight']/total_weight:.1%})")
            print(f"  Opinion Pattern: {arch.get('opinion_pattern', [])}")
            
            variance = arch.get('variance', {})
            print(f"  Variance: Likert={variance.get('likert', 0):.2f}, "
                  f"Cat={variance.get('categorical', 0):.2f}, "
                  f"Ord={variance.get('ordinal', 0):.2f}")
            print()
        
        print(f"Total archetypes: {len(self.true_archetypes)}")
        print(f"Total weight: {total_weight:.3f}")
        print("="*80 + "\n")


if __name__ == "__main__":
    print("\n" + "="*80)
    print("🧪 TESTING SURVEY DATA GENERATOR")
    print("="*80 + "\n")
    
    try:
        # Create generator
        generator = SurveyDataGenerator(seed=42)
        
        # Print summary
        generator.print_archetype_summary()
        
        # Print questions
        print(f"Opinion questions: {len(generator.opinion_questions)}")
        for q in generator.opinion_questions:
            print(f"  • {q.id}: {q.text}")
        
        print(f"\nDemographic questions: {len(generator.demographic_questions)}")
        for q in generator.demographic_questions:
            print(f"  • {q.id}: {q.text}")
        
        # Generate data
        print(f"\nGenerating data...")
        df = generator.generate(n_respondents=50)
        
        print(f"✅ Generated {len(df)} respondents")
        print(f"   Columns: {list(df.columns)}")
        print(f"\nFirst 3 rows:")
        print(df.head(3))
        
        print(f"\nArchetype distribution:")
        print(df['archetype_name'].value_counts())
        
        print("\n" + "="*80)
        print("✅ DATA GENERATOR TESTS PASSED")
        print("="*80 + "\n")
    
    except FileNotFoundError as e:
        print(f"❌ Error: {e}")
        print("\n⚠️  Missing JSON configuration files!")
        print("   Run: python migrate_to_json.py")
        print("\n" + "="*80 + "\n")