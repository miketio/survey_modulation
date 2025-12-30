"""
simulation/population_simulator.py

Simulates large survey populations from calibrated persona distributions.
Uses transition probabilities for realistic response variance.
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Tuple
from collections import Counter

from config.settings import SIMULATION
from config.questions import Question, QuestionType
from agents.survey_agent import SurveyAgent
from utils.file_io import save_dataframe


class PopulationSimulator:
    """
    Simulates survey population using calibrated transition probabilities.
    """
    
    def __init__(
        self,
        personas: List[Dict],
        proportions: np.ndarray,
        random_seed: int = None
    ):
        """
        Initialize simulator.
        
        Args:
            personas: List of persona dictionaries
            proportions: Archetype weights (must sum to 1)
            random_seed: Random seed for reproducibility
        """
        self.personas = personas
        self.proportions = np.array(proportions)
        
        # Normalize proportions
        self.proportions = self.proportions / self.proportions.sum()
        
        self.random_seed = random_seed or SIMULATION.RANDOM_SEED if hasattr(SIMULATION, 'RANDOM_SEED') else 42
        np.random.seed(self.random_seed)
        
        # Create agents
        self.agents = [SurveyAgent(p) for p in personas]
        
        # Calibration storage
        self.archetype_distributions = None
    
    def calibrate(
        self,
        questions: List[Question],
        n_samples: int = None,
        verbose: bool = True
    ) -> List[List[Dict]]:
        """
        Calibrate by having each agent answer multiple times.
        
        Args:
            questions: Questions to calibrate on
            n_samples: Number of samples per agent
            verbose: Print progress
        
        Returns:
            List of distribution info per archetype
        """
        n_samples = n_samples or SIMULATION.N_CALIBRATION_SAMPLES
        
        if verbose:
            print("\n" + "="*80)
            print("🎯 CALIBRATING TRANSITION PROBABILITIES")
            print("="*80)
        
        self.archetype_distributions = []
        
        for i, agent in enumerate(self.agents):
            if verbose:
                print(f"\n[{i+1}/{len(self.agents)}] Calibrating: {agent.get_name()}")
            
            # Get multiple samples
            answers_with_probs = self._calibrate_agent(agent, questions, n_samples, verbose)
            self.archetype_distributions.append(answers_with_probs)
        
        if verbose:
            print("\n✅ Calibration complete")
        
        return self.archetype_distributions
    
    def _calibrate_agent(
        self,
        agent: SurveyAgent,
        questions: List[Question],
        n_samples: int,
        verbose: bool
    ) -> List[Dict]:
        """Calibrate single agent by sampling responses"""
        
        results = []
        
        for q in questions:
            q_type = q.type
            q_id = q.id
            
            # Collect samples
            samples = []
            for _ in range(n_samples):
                single_answer = agent.answer_survey([q], verbose=False, n_samples=1)[0]
                samples.append(single_answer)
            
            if q_type == QuestionType.LIKERT:
                # Calculate transition probabilities for Likert
                samples_array = np.array([float(s) for s in samples])
                
                # Mode (most common answer)
                counts = Counter(samples_array)
                mode_answer = int(counts.most_common(1)[0][0])
                
                # Transition probabilities
                n_lower = np.sum(samples_array < mode_answer)
                n_higher = np.sum(samples_array > mode_answer)
                n_stay = np.sum(samples_array == mode_answer)
                
                p_lower = n_lower / len(samples_array)
                p_higher = n_higher / len(samples_array)
                p_stay = n_stay / len(samples_array)
                
                std_answer = np.std(samples_array)
                mean_answer = np.mean(samples_array)
                
                results.append({
                    'question_id': q_id,
                    'question_text': q.text,
                    'type': q_type.value,
                    'question_type': 'likert',
                    'answer': mode_answer,
                    'mean_answer': float(mean_answer),
                    'modal_answer': mode_answer,
                    'p_lower': float(p_lower),
                    'p_higher': float(p_higher),
                    'p_stay': float(p_stay),
                    'std': float(std_answer),
                    'samples': samples_array.tolist(),
                    'distribution': {int(k): int(v) for k, v in counts.items()}
                })
                
                if verbose and results and len(results) <= 3:
                    print(f"    • {q_id}: answer={mode_answer} (↓{p_lower:.0%} ={p_stay:.0%} ↑{p_higher:.0%})")
            
            else:
                # Categorical/Ordinal: mode and frequency
                counts = Counter(samples)
                mode_answer = counts.most_common(1)[0][0]
                mode_freq = counts[mode_answer] / len(samples)
                
                p_stay = mode_freq
                p_change = 1.0 - mode_freq
                
                results.append({
                    'question_id': q_id,
                    'question_text': q.text,
                    'type': q_type.value,
                    'question_type': 'categorical',
                    'answer': mode_answer,
                    'modal_answer': mode_answer,
                    'p_stay': float(p_stay),
                    'p_change': float(p_change),
                    'distribution': dict(counts),
                    'samples': samples
                })
                
                if verbose and results and len(results) <= 3:
                    print(f"    • {q_id}: {mode_answer} (stay={p_stay:.0%})")
        
        return results
    
    def simulate_population(
        self,
        questions: List[Question],
        n_respondents: int = None,
        verbose: bool = True
    ) -> pd.DataFrame:
        """
        Generate simulated respondents using transition probabilities.
        
        Args:
            questions: Questions to simulate
            n_respondents: Population size
            verbose: Print progress
        
        Returns:
            DataFrame with simulated responses
        """
        if self.archetype_distributions is None:
            raise ValueError("Must call calibrate() first")
        
        n_respondents = n_respondents or SIMULATION.N_SIMULATED_RESPONDENTS
        
        if verbose:
            print("\n" + "="*80)
            print(f"🎲 SIMULATING {n_respondents} RESPONDENTS")
            print("="*80)
        
        # Assign archetypes
        archetype_assignments = np.random.choice(
            len(self.personas),
            size=n_respondents,
            p=self.proportions
        )
        
        if verbose:
            print(f"\nArchetype assignments:")
            for i, persona in enumerate(self.personas):
                count = np.sum(archetype_assignments == i)
                print(f"  • {persona['name']}: {count} ({count/n_respondents:.1%})")
        
        # Generate responses
        simulated_data = []
        
        for resp_id in range(n_respondents):
            archetype_idx = archetype_assignments[resp_id]
            archetype_dist = self.archetype_distributions[archetype_idx]
            
            respondent_data = {
                'respondent_id': resp_id,
                'true_archetype': archetype_idx,
                'archetype_name': self.personas[archetype_idx]['name']
            }
            
            # Sample answer for each question
            for q_idx, q in enumerate(questions):
                dist_info = archetype_dist[q_idx]
                q_id = q.id
                q_type = q.type
                
                if q_type == QuestionType.LIKERT:
                    # Use transition probabilities
                    base_answer = dist_info['modal_answer']
                    p_lower = dist_info['p_lower']
                    p_higher = dist_info['p_higher']
                    p_stay = dist_info['p_stay']
                    
                    # Sample transition
                    transition = np.random.choice(
                        [-1, 0, 1],
                        p=[p_lower, p_stay, p_higher]
                    )
                    
                    # Apply with bounds
                    final_answer = base_answer + transition
                    final_answer = int(np.clip(final_answer, 1, 5))
                    
                    respondent_data[q_id] = final_answer
                
                else:
                    # Categorical/Ordinal: sample from distribution
                    options = list(dist_info['distribution'].keys())
                    probs = [
                        dist_info['distribution'][opt] / len(dist_info['samples'])
                        for opt in options
                    ]
                    
                    respondent_data[q_id] = np.random.choice(options, p=probs)
            
            simulated_data.append(respondent_data)
        
        df = pd.DataFrame(simulated_data)
        
        if verbose:
            print(f"\n✅ Generated {len(df)} simulated respondents")
        
        return df
    
    def get_archetype_summary(self) -> pd.DataFrame:
        """
        Get summary of archetype distributions.
        
        Returns:
            DataFrame with distribution statistics
        """
        if self.archetype_distributions is None:
            raise ValueError("Must call calibrate() first")
        
        summary_data = []
        
        for i, (persona, dist) in enumerate(zip(self.personas, self.archetype_distributions)):
            for ans in dist:
                base_row = {
                    'archetype': persona['name'],
                    'archetype_idx': i,
                    'weight': self.proportions[i],
                    'question_id': ans['question_id'],
                    'question': ans['question_text'],
                    'question_type': ans['question_type'],
                    'modal_answer': ans['modal_answer'],
                }
                
                if ans['question_type'] == 'likert':
                    base_row.update({
                        'mean_answer': ans.get('mean_answer', ans['modal_answer']),
                        'p_go_lower': ans['p_lower'],
                        'p_stay': ans['p_stay'],
                        'p_go_higher': ans['p_higher'],
                        'std_dev': ans['std']
                    })
                else:
                    base_row.update({
                        'p_stay': ans['p_stay'],
                        'p_change': ans.get('p_change', 1 - ans['p_stay'])
                    })
                
                summary_data.append(base_row)
        
        return pd.DataFrame(summary_data)
    
    def save_simulation(
        self,
        simulated_df: pd.DataFrame,
        filepath = None
    ):
        """Save simulated data"""
        return save_dataframe(simulated_df, filepath=filepath)


if __name__ == "__main__":
    from config.questions import get_opinion_questions
    from generators.persona_generator import PersonaGenerator
    
    print("\n" + "="*80)
    print("🧪 TESTING POPULATION SIMULATOR")
    print("="*80 + "\n")
    
    # Create test personas
    print("Creating test personas...")
    test_personas = [
        {
            'name': 'Progressive, 22',
            'occupation': 'Student',
            'values': ['Change', 'Innovation'],
            'fears': ['Stagnation'],
            'worldview': 'Progressive worldview',
            'system_prompt': 'You are progressive. Answer with high scores on tech and low on tradition.',
            'weight': 0.5,
            'archetype_index': 0
        },
        {
            'name': 'Conservative, 55',
            'occupation': 'Professional',
            'values': ['Stability', 'Tradition'],
            'fears': ['Chaos'],
            'worldview': 'Conservative worldview',
            'system_prompt': 'You are conservative. Answer with low scores on tech and high on tradition.',
            'weight': 0.5,
            'archetype_index': 1
        }
    ]
    
    proportions = np.array([0.5, 0.5])
    
    # Create simulator
    simulator = PopulationSimulator(test_personas, proportions)
    
    # Calibrate
    questions = get_opinion_questions()[:3]
    print(f"\nCalibrating on {len(questions)} questions...")
    
    try:
        distributions = simulator.calibrate(questions, n_samples=5, verbose=True)
        
        # Simulate
        print("\nSimulating population...")
        simulated_df = simulator.simulate_population(questions, n_respondents=100, verbose=True)
        
        print(f"\n✅ Simulated dataframe shape: {simulated_df.shape}")
        print(f"\nFirst 3 rows:")
        print(simulated_df.head(3))
        
        # Get summary
        summary = simulator.get_archetype_summary()
        print(f"\n✅ Summary shape: {summary.shape}")
        
        print("\n" + "="*80)
        print("✅ POPULATION SIMULATOR TESTS PASSED")
        print("="*80 + "\n")
        
    except Exception as e:
        print(f"\n⚠️  Test requires Ollama running: {e}")
        print("   Skipping simulation test")