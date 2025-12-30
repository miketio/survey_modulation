"""
analysis/visualization.py

Comprehensive visualization functions for survey analysis.
All plots are saved (not shown) as per requirement.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Tuple
from pathlib import Path

from config.settings import PATHS, VIZ
from config.questions import Question, QuestionType


# Set style
try:
    plt.style.use(VIZ.STYLE)
except:
    plt.style.use('default')


class SurveyVisualizer:
    """Creates publication-quality visualizations"""
    
    def __init__(self):
        self.colors = sns.color_palette(VIZ.COLOR_PALETTE, VIZ.N_COLORS)
        sns.set_palette(self.colors)
        
        # Ensure plots directory exists
        PATHS.PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    
    def _save_figure(self, filename: str):
        """Save figure to plots directory"""
        filepath = PATHS.PLOTS_DIR / filename
        plt.savefig(filepath, dpi=VIZ.DPI, bbox_inches='tight', format=VIZ.FIGURE_FORMAT)
        print(f"  ✓ Saved: {filepath}")
        plt.close()
    
    def plot_response_distributions(
        self,
        df: pd.DataFrame,
        questions: List[Question],
        filename: str = "1_input_data_distributions.png"
    ):
        """
        Plot distributions for original input data.
        
        Args:
            df: DataFrame with responses
            questions: Question objects
            filename: Output filename
        """
        n_questions = len(questions)
        n_cols = 3
        n_rows = (n_questions + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 4 * n_rows))
        axes = axes.flatten() if n_rows > 1 else [axes] if n_cols == 1 else axes
        
        for i, q in enumerate(questions):
            ax = axes[i]
            q_id = q.id
            
            if q_id not in df.columns:
                ax.axis('off')
                continue
            
            data = df[q_id].dropna()
            
            if len(data) == 0:
                ax.axis('off')
                continue
            
            if q.type == QuestionType.LIKERT:
                counts = data.value_counts().sort_index()
                ax.bar(counts.index, counts.values, color=self.colors[0], alpha=0.7, edgecolor='black')
                ax.set_xticks([1, 2, 3, 4, 5])
                ax.set_xlabel('Response (1-5)')
            else:
                counts = data.value_counts()
                y_pos = np.arange(len(counts))
                ax.barh(y_pos, counts.values, color=self.colors[0], alpha=0.7, edgecolor='black')
                ax.set_yticks(y_pos)
                ax.set_yticklabels(counts.index, fontsize=8)
                ax.set_xlabel('Count')
            
            title_text = q.text[:40] + "..." if len(q.text) > 40 else q.text
            ax.set_title(f"{q.id}: {title_text}", fontsize=10, fontweight='bold')
            ax.grid(True, alpha=0.3, axis='x' if q.type == QuestionType.LIKERT else 'y')
        
        # Hide unused subplots
        for i in range(n_questions, len(axes)):
            axes[i].axis('off')
        
        plt.tight_layout()
        self._save_figure(filename)
    
    def plot_archetype_patterns(
        self,
        archetypes: np.ndarray,
        decoded_archetypes: List[Dict],
        questions: List[Question],
        proportions: np.ndarray,
        personas: List[Dict],
        filename: str = "2_archetype_patterns.png"
    ):
        """
        Plot defining traits of each archetype.
        
        Args:
            archetypes: Archetype patterns matrix
            decoded_archetypes: Decoded patterns
            questions: Question objects
            proportions: Archetype weights
            personas: Persona descriptions
            filename: Output filename
        """
        n_archetypes = len(archetypes)
        fig, axes = plt.subplots(1, n_archetypes, figsize=(5 * n_archetypes, 6))
        
        if n_archetypes == 1:
            axes = [axes]
        
        for i, ax in enumerate(axes):
            decoded = decoded_archetypes[i]
            name = personas[i]['name'] if i < len(personas) else f"Type {i+1}"
            
            # Get Likert questions only
            values = []
            labels = []
            for q in questions:
                if q.type == QuestionType.LIKERT and q.id in decoded:
                    values.append(decoded[q.id]['value'])
                    labels.append(q.id)
            
            if not values:
                ax.axis('off')
                continue
            
            y_pos = np.arange(len(values))
            color = self.colors[i % len(self.colors)]
            ax.barh(y_pos, values, color=color, alpha=0.8, edgecolor='black')
            ax.set_yticks(y_pos)
            ax.set_yticklabels(labels, fontsize=9)
            ax.set_xlabel('Response (1-5)', fontsize=10)
            ax.set_title(f"{name}\n({proportions[i]:.1%} of population)", 
                        fontsize=11, fontweight='bold')
            ax.set_xlim(0, 5.5)
            ax.grid(True, axis='x', alpha=0.3)
            ax.axvline(3, color='gray', linestyle=':', alpha=0.5)
        
        plt.tight_layout()
        self._save_figure(filename)
    
    def plot_second_survey_results(
        self,
        results: Dict,
        questions: List[Question],
        proportions: np.ndarray,
        filename: str = "3_second_survey_results.png"
    ):
        """
        Grouped bar chart: Personas vs Weighted Population Average.
        
        Args:
            results: Dictionary of {persona_name: [answer_dicts]}
            questions: Question objects
            proportions: Archetype weights (must match order of results)
            filename: Output filename
        """
        # Filter Likert questions only
        likert_qs = [q for q in questions if q.type == QuestionType.LIKERT]
        
        if not likert_qs:
            print("⚠️  No Likert questions to visualize")
            return
        
        persona_names = list(results.keys())
        n_personas = len(persona_names)
        n_questions = len(likert_qs)
        
        # Setup plot
        fig, ax = plt.subplots(figsize=(15, 8))
        
        total_bars = n_personas + 1
        bar_width = 0.8 / total_bars
        indices = np.arange(n_questions)
        
        # Store persona scores
        persona_scores_matrix = np.zeros((n_personas, n_questions))
        
        # Plot individual personas
        for i, name in enumerate(persona_names):
            persona_answers = []
            for j, q in enumerate(likert_qs):
                ans_entry = next((item for item in results[name] if item["question_id"] == q.id), None)
                score = ans_entry['answer'] if ans_entry and isinstance(ans_entry['answer'], (int, float)) else 0
                persona_answers.append(score)
            
            persona_scores_matrix[i, :] = persona_answers
            
            pos = indices - 0.4 + (i * bar_width) + (bar_width / 2)
            color = self.colors[i % len(self.colors)]
            ax.bar(pos, persona_answers, width=bar_width, label=name, color=color, alpha=0.8, edgecolor='black')
        
        # Calculate and plot weighted average
        society_means = []
        society_stds = []
        
        for j in range(n_questions):
            col_values = persona_scores_matrix[:, j]
            weighted_mean, weighted_std = self._calculate_weighted_stats(col_values, proportions)
            society_means.append(weighted_mean)
            society_stds.append(weighted_std)
        
        soc_pos = indices - 0.4 + (n_personas * bar_width) + (bar_width / 2)
        ax.bar(soc_pos, society_means, width=bar_width, label="Society (Weighted Avg)",
               color='#333333', hatch='//', alpha=0.9, yerr=society_stds, capsize=4, edgecolor='black')
        
        # Formatting
        ax.set_ylabel('Score (1-5)', fontsize=12)
        ax.set_title('Second Survey: Archetypes vs Society Average', fontsize=16, fontweight='bold')
        ax.set_xticks(indices)
        
        x_labels = [f"{q.id}\n{q.text[:25]}..." for q in likert_qs]
        ax.set_xticklabels(x_labels, rotation=0, fontsize=10)
        
        ax.legend(title="Groups", bbox_to_anchor=(1.0, 1), loc='upper left', fontsize=9)
        ax.grid(True, axis='y', alpha=0.3)
        ax.set_ylim(0, 6)
        
        plt.tight_layout()
        self._save_figure(filename)
    
    def _calculate_weighted_stats(self, values: np.ndarray, weights: np.ndarray) -> Tuple[float, float]:
        """Calculate weighted mean and standard deviation"""
        weights = weights / weights.sum()
        weighted_mean = np.average(values, weights=weights)
        weighted_variance = np.average((values - weighted_mean)**2, weights=weights)
        weighted_std = np.sqrt(weighted_variance)
        return weighted_mean, weighted_std
    
    def plot_archetype_distributions(
        self,
        summary_df: pd.DataFrame,
        filename: str = "4_archetype_distributions.png"
    ):
        """
        Plot archetype response distributions with transition probabilities.
        Only plots Likert questions.
        
        Args:
            summary_df: Summary DataFrame from simulator
            filename: Output filename
        """
        # Filter for Likert questions only
        likert_only = summary_df[summary_df['question_type'] == 'likert'].copy()
        
        if len(likert_only) == 0:
            print("⚠️  No Likert questions found for visualization")
            return
        
        n_archetypes = likert_only['archetype_idx'].nunique()
        archetypes = sorted(likert_only['archetype_idx'].unique())
        
        fig, axes = plt.subplots(1, n_archetypes, figsize=(6*n_archetypes, 6))
        
        if n_archetypes == 1:
            axes = [axes]
        
        for idx, arch_idx in enumerate(archetypes):
            ax = axes[idx]
            arch_data = likert_only[likert_only['archetype_idx'] == arch_idx]
            
            if len(arch_data) == 0:
                ax.axis('off')
                continue
            
            y_pos = np.arange(len(arch_data))
            
            # Plot modal answers
            modal_answers = arch_data['modal_answer'].values.astype(float)
            color = self.colors[idx % len(self.colors)]
            ax.barh(y_pos, modal_answers, color=color, alpha=0.7, edgecolor='black')
            
            # Draw transition probability indicators
            p_lower = arch_data['p_go_lower'].values
            p_higher = arch_data['p_go_higher'].values
            
            for j, (pos, modal, pl, ph) in enumerate(zip(y_pos, modal_answers, p_lower, p_higher)):
                if pl > 0.1:
                    ax.plot([modal-1, modal], [pos, pos], 'k--', alpha=0.5, linewidth=2)
                    ax.text(modal-0.5, pos, f'{pl:.0%}', fontsize=8, va='bottom')
                
                if ph > 0.1:
                    ax.plot([modal, modal+1], [pos, pos], 'k--', alpha=0.5, linewidth=2)
                    ax.text(modal+0.5, pos, f'{ph:.0%}', fontsize=8, va='bottom')
            
            # Labels
            ax.set_yticks(y_pos)
            ax.set_yticklabels(arch_data['question_id'].values, fontsize=9)
            ax.set_xlabel('Response (1-5)', fontsize=10)
            ax.set_xlim(0, 6)
            
            name = arch_data['archetype'].iloc[0]
            weight = arch_data['weight'].iloc[0]
            ax.set_title(f"{name}\n({weight:.1%} of population)", 
                        fontsize=11, fontweight='bold')
            
            ax.grid(True, alpha=0.3, axis='x')
        
        plt.tight_layout()
        self._save_figure(filename)
    
    def plot_simulated_population(
        self,
        simulated_df: pd.DataFrame,
        questions: List[Question],
        filename: str = "5_simulated_population.png"
    ):
        """
        Visualize simulated population responses with mean ± std.
        Only plots Likert questions.
        
        Args:
            simulated_df: Simulated population DataFrame
            questions: Question objects
            filename: Output filename
        """
        # Filter Likert questions only
        likert_questions = [q for q in questions if q.type == QuestionType.LIKERT]
        
        if len(likert_questions) == 0:
            print("⚠️  No Likert questions to visualize")
            return
        
        n_questions = len(likert_questions)
        n_cols = 3
        n_rows = (n_questions + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 4*n_rows))
        axes = axes.flatten() if n_rows > 1 else [axes] if n_cols == 1 else axes
        
        archetype_names = sorted(simulated_df['archetype_name'].unique())
        colors_map = {name: self.colors[i % len(self.colors)] for i, name in enumerate(archetype_names)}
        
        for i, q in enumerate(likert_questions):
            ax = axes[i]
            q_id = q.id
            
            if q_id not in simulated_df.columns:
                ax.axis('off')
                continue
            
            # Convert to numeric
            all_values = pd.to_numeric(simulated_df[q_id], errors='coerce').dropna()
            
            if len(all_values) == 0:
                ax.axis('off')
                continue
            
            # Stacked histogram by archetype
            bottom = np.zeros(5)
            
            for arch_name in archetype_names:
                arch_data = simulated_df[simulated_df['archetype_name'] == arch_name]
                values = pd.to_numeric(arch_data[q_id], errors='coerce').dropna()
                
                if len(values) == 0:
                    continue
                
                counts = [int((values == score).sum()) for score in range(1, 6)]
                
                ax.bar(range(1, 6), counts, bottom=bottom,
                      label=arch_name, color=colors_map[arch_name],
                      edgecolor='black', linewidth=0.5, alpha=0.8)
                
                bottom += np.array(counts)
            
            # Mean and std
            mean_val = all_values.mean()
            std_val = all_values.std()
            
            ax.axvline(mean_val, color='red', linestyle='--', linewidth=2,
                      label=f'Mean: {mean_val:.2f}', zorder=10)
            ax.axvspan(mean_val - std_val, mean_val + std_val,
                      alpha=0.15, color='red', zorder=5)
            
            # Text box
            textstr = f'μ = {mean_val:.2f}\nσ = {std_val:.2f}'
            props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
            ax.text(0.98, 0.98, textstr, transform=ax.transAxes,
                   fontsize=9, verticalalignment='top', horizontalalignment='right',
                   bbox=props)
            
            title_text = q.text[:40] + "..." if len(q.text) > 40 else q.text
            ax.set_xlabel('Response', fontsize=10)
            ax.set_ylabel('Count', fontsize=10)
            ax.set_title(f"{q.id}: {title_text}", fontsize=10, fontweight='bold')
            ax.set_xticks([1, 2, 3, 4, 5])
            ax.set_xlim(0.5, 5.5)
            ax.legend(fontsize=7, loc='upper left')
            ax.grid(True, alpha=0.3, axis='y')
        
        # Hide unused subplots
        for i in range(n_questions, len(axes)):
            axes[i].axis('off')
        
        plt.tight_layout()
        self._save_figure(filename)


if __name__ == "__main__":
    from config.questions import get_opinion_questions
    
    print("\n" + "="*80)
    print("🧪 TESTING VISUALIZER")
    print("="*80 + "\n")
    
    # Create test data
    questions = get_opinion_questions()
    
    # Test DataFrame
    test_data = {
        'Q1': np.random.randint(1, 6, 100),
        'Q4': np.random.randint(1, 6, 100),
        'Q6': np.random.randint(1, 6, 100),
    }
    df = pd.DataFrame(test_data)
    
    viz = SurveyVisualizer()
    
    print("Testing response distributions plot...")
    viz.plot_response_distributions(df, questions[:3], filename="test_distributions.png")
    
    print("\n✅ Visualization tests passed")
    print("   (Check data/output/plots/ for test files)")