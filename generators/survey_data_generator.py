"""
generators/survey_data_generator.py

Generates synthetic survey data with embedded archetypal patterns.
"""

import numpy as np
import pandas as pd
from typing import List, Dict

from config.settings import DATA_GEN
from config.questions import (
    Question, QuestionType, 
    get_opinion_questions, get_demographic_questions, ALL_QUESTIONS
)


class SurveyDataGenerator:
    """
    Generates synthetic survey responses with archetypal structure.
    """
    
    def __init__(self, seed: int = None):
        """
        Initialize generator.
        
        Args:
            seed: Random seed for reproducibility
        """
        self.seed = seed or DATA_GEN.RANDOM_SEED if hasattr(DATA_GEN, 'RANDOM_SEED') else 42
        np.random.seed(self.seed)
        
        self.opinion_questions = get_opinion_questions()
        self.demographic_questions = get_demographic_questions()
        self.all_questions = self.opinion_questions + self.demographic_questions
        
        # Build ground truth archetypes
        self.true_archetypes = self._build_archetypes()
    
    def _build_archetypes(self) -> List[Dict]:
        """Build ground truth archetypal patterns"""
        
        archetypes = [
            {
                'name': 'Young Urban Progressive',
                'opinion_pattern': [3, 5, 2, 5, 5],  # Trust, Tech+, Traditional-, Ecology+, Risk+
                'demographic_pattern': {
                    'age': '18-24',
                    'location': 'Urban/City',
                    'education': "Bachelor's",
                    'politics': 'Very Liberal',
                    'news': 'Daily'
                },
                'variance': {'likert': 0.8, 'categorical': 0.2, 'ordinal': 0.15},
                'weight': 0.30
            },
            {
                'name': 'Conservative Rural Traditionalist',
                'opinion_pattern': [2, 2, 5, 2, 2],  # Trust-, Tech-, Traditional+, Ecology-, Risk-
                'demographic_pattern': {
                    'age': '45-54',
                    'location': 'Rural/Countryside',
                    'education': 'High School',
                    'politics': 'Conservative',
                    'news': 'Sometimes'
                },
                'variance': {'likert': 0.7, 'categorical': 0.15, 'ordinal': 0.1},
                'weight': 0.25
            },
            {
                'name': 'Middle-Aged Suburban Moderate',
                'opinion_pattern': [3, 4, 3, 4, 3],  # Neutral overall
                'demographic_pattern': {
                    'age': '35-44',
                    'location': 'Suburban',
                    'education': "Bachelor's",
                    'politics': 'Moderate',
                    'news': 'Often'
                },
                'variance': {'likert': 0.6, 'categorical': 0.1, 'ordinal': 0.1},
                'weight': 0.30
            },
            {
                'name': 'Apathetic Low-Engagement',
                'opinion_pattern': [3, 3, 3, 3, 2],  # Mostly neutral, risk-averse
                'demographic_pattern': {
                    'age': '25-34',
                    'location': 'Urban/City',
                    'education': 'Some College',
                    'politics': 'Moderate',
                    'news': 'Rarely'
                },
                'variance': {'likert': 1.0, 'categorical': 0.3, 'ordinal': 0.2},
                'weight': 0.15
            }
        ]
        
        # Map patterns to question IDs
        for archetype in archetypes:
            full_pattern = {}
            
            # Map opinion questions
            for i, q in enumerate(self.opinion_questions):
                if i < len(archetype['opinion_pattern']):
                    full_pattern[q.id] = archetype['opinion_pattern'][i]
                else:
                    full_pattern[q.id] = 3.0
            
            # Map demographic questions
            demo_map = archetype['demographic_pattern']
            for q in self.demographic_questions:
                q_text_lower = q.text.lower()
                
                if 'age' in q_text_lower:
                    full_pattern[q.id] = demo_map.get('age', q.options[2])
                elif 'live' in q_text_lower or 'location' in q_text_lower:
                    full_pattern[q.id] = demo_map.get('location', q.options[0])
                elif 'education' in q_text_lower:
                    full_pattern[q.id] = demo_map.get('education', q.options[2])
                elif 'political' in q_text_lower:
                    full_pattern[q.id] = demo_map.get('politics', q.options[2])
                elif 'news' in q_text_lower:
                    full_pattern[q.id] = demo_map.get('news', q.options[2])
                else:
                    full_pattern[q.id] = q.options[len(q.options)//2]
            
            archetype['pattern'] = full_pattern
        
        return archetypes
    
    def generate(
        self, 
        n_respondents: int = None,
        missing_rate: float = None
    ) -> pd.DataFrame:
        """
        Generate synthetic survey responses.
        
        Args:
            n_respondents: Number of respondents
            missing_rate: Proportion of missing data
        
        Returns:
            DataFrame with responses
        """
        n_respondents = n_respondents or DATA_GEN.N_RESPONDENTS
        missing_rate = missing_rate or DATA_GEN.MISSING_RATE
        
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
                    
                    if q_id not in pattern:
                        # Default values
                        if q.type == QuestionType.LIKERT:
                            pattern[q_id] = 3.0
                        else:
                            pattern[q_id] = q.options[len(q.options)//2]
                    
                    base_value = pattern[q_id]
                    
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


if __name__ == "__main__":
    print("\n" + "="*80)
    print("🧪 TESTING SURVEY DATA GENERATOR")
    print("="*80 + "\n")
    
    generator = SurveyDataGenerator(seed=42)
    
    print(f"Opinion questions: {len(generator.opinion_questions)}")
    for q in generator.opinion_questions:
        print(f"  • {q.id}: {q.text}")
    
    print(f"\nDemographic questions: {len(generator.demographic_questions)}")
    for q in generator.demographic_questions:
        print(f"  • {q.id}: {q.text}")
    
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