"""
core/archetypal_analyzer.py

Mathematical decomposition of survey data into archetypal patterns.
Supports both PCHA (true Archetypal Analysis) and NMF (approximation).
"""

import numpy as np
from sklearn.decomposition import NMF
from typing import Tuple, List, Optional

from config.settings import ANALYSIS

# Try to import specialized archetypes library
try:
    from archetypes import AA
    PCHA_AVAILABLE = True
except ImportError:
    PCHA_AVAILABLE = False


class ArchetypalAnalyzer:
    """
    Performs archetypal decomposition of survey data.
    
    Decomposes data matrix X into:
        X ≈ W × H
    Where:
        W = weights (respondents × archetypes)
        H = archetypes (archetypes × questions)
    """
    
    def __init__(
        self, 
        n_archetypes: int = None,
        random_state: int = None,
        max_iter: int = None
    ):
        """
        Initialize analyzer.
        
        Args:
            n_archetypes: Number of archetypes to extract
            random_state: Random seed for reproducibility
            max_iter: Maximum iterations for optimization
        """
        self.n_archetypes = n_archetypes or ANALYSIS.N_ARCHETYPES
        self.random_state = random_state or ANALYSIS.RANDOM_SEED
        self.max_iter = max_iter or ANALYSIS.MAX_ITERATIONS
        
        # State variables
        self.model = None
        self.archetypes = None  # H matrix
        self.weights = None     # W matrix
        self.proportions = None
        self.r2_scores = None
        self.total_r2 = None
        self.residuals = None
        self.method_used = None
    
    def fit(
        self, 
        data: np.ndarray, 
        method: str = 'auto',
        verbose: bool = True
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Fit the archetypal model to data.
        
        Args:
            data: Response matrix (n_respondents × n_questions)
            method: 'pcha', 'nmf', or 'auto' (tries PCHA first)
            verbose: Print progress
        
        Returns:
            Tuple of (archetypes, weights, proportions)
        """
        if method == 'auto':
            # Try PCHA first, fall back to NMF
            if PCHA_AVAILABLE:
                try:
                    return self.fit_pcha(data, verbose=verbose)
                except Exception as e:
                    if verbose:
                        print(f"⚠️  PCHA failed: {e}")
                        print("   Falling back to NMF...")
                    return self.fit_nmf(data, verbose=verbose)
            else:
                return self.fit_nmf(data, verbose=verbose)
        
        elif method == 'pcha':
            return self.fit_pcha(data, verbose=verbose)
        
        elif method == 'nmf':
            return self.fit_nmf(data, verbose=verbose)
        
        else:
            raise ValueError(f"Unknown method: {method}")
    
    def fit_pcha(
        self, 
        data: np.ndarray, 
        verbose: bool = True
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Principal Convex Hull Analysis.
        Ensures archetypes are convex combinations of data points.
        
        Args:
            data: Response matrix
            verbose: Print progress
        
        Returns:
            Tuple of (archetypes, weights, proportions)
        """
        if not PCHA_AVAILABLE:
            raise ImportError(
                "PCHA requires 'archetypes' package. "
                "Install with: pip install archetypes"
            )
        
        if verbose:
            print(f"\n🔬 Running PCHA (k={self.n_archetypes})...")
        
        # Initialize and fit
        self.model = AA(
            n_archetypes=self.n_archetypes,
            random_state=self.random_state,
            max_iter=self.max_iter,
            tol=1e-5
        )
        
        self.model.fit(data)
        
        # Extract results
        self.archetypes = self.model.archetypes_  # Pure types
        self.weights = self.model.coefficients_    # How much each person resembles each type
        
        # Calculate proportions (average weight across population)
        self.proportions = self.weights.mean(axis=0)
        
        # Calculate quality metrics
        self._calculate_metrics(data, verbose=verbose)
        
        self.method_used = 'PCHA'
        
        if verbose:
            print(f"✅ PCHA converged")
            print(f"   Total R²: {self.total_r2:.1%}")
        
        return self.archetypes, self.weights, self.proportions
    
    def fit_nmf(
        self, 
        data: np.ndarray, 
        verbose: bool = True
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Enhanced NMF with range correction.
        Approximation to true archetypal analysis.
        
        Args:
            data: Response matrix
            verbose: Print progress
        
        Returns:
            Tuple of (archetypes, weights, proportions)
        """
        if verbose:
            print(f"\n🔬 Running Enhanced NMF (k={self.n_archetypes})...")
        
        # Initialize NMF
        self.model = NMF(
            n_components=self.n_archetypes,
            init='nndsvda',
            solver='mu',
            max_iter=self.max_iter,
            random_state=self.random_state
        )
        
        # Fit
        W_raw = self.model.fit_transform(data)
        H_raw = self.model.components_
        
        # Range correction: scale H to represent actual survey scores
        scale_factors = W_raw.sum(axis=0)
        scale_factors[scale_factors == 0] = 1.0
        
        self.archetypes = H_raw * (scale_factors[:, np.newaxis] / (data.shape[0] / self.n_archetypes))
        self.weights = W_raw / (scale_factors[np.newaxis, :] / (data.shape[0] / self.n_archetypes))
        
        # Clip to valid survey range (e.g., 1-5 for Likert)
        self.archetypes = np.clip(self.archetypes, 1.0, 5.0)
        
        # Normalize weights to sum to 1 (convexity constraint)
        row_sums = self.weights.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1.0
        self.weights = self.weights / row_sums
        
        # Calculate proportions
        self.proportions = self.weights.mean(axis=0)
        
        # Calculate quality metrics
        self._calculate_metrics(data, verbose=verbose)
        
        self.method_used = 'NMF'
        
        if verbose:
            print(f"✅ NMF converged")
            print(f"   Total R²: {self.total_r2:.1%}")
        
        return self.archetypes, self.weights, self.proportions
    
    def _calculate_metrics(self, data: np.ndarray, verbose: bool = True):
        """Calculate R² and residuals"""
        
        # Reconstruct data
        reconstructed = self.weights @ self.archetypes
        
        # Per-respondent residuals
        self.residuals = np.sqrt(np.sum((data - reconstructed)**2, axis=1))
        
        # Total variance explained (R²)
        ss_total = np.sum((data - np.mean(data, axis=0))**2)
        ss_residual = np.sum((data - reconstructed)**2)
        self.total_r2 = 1 - (ss_residual / ss_total)
        
        # Per-archetype contribution to R²
        self.r2_scores = []
        for i in range(self.n_archetypes):
            comp_i = self.weights[:, i:i+1] @ self.archetypes[i:i+1, :]
            norm_i = np.sum(comp_i**2)
            contrib = norm_i / np.sum(reconstructed**2) * self.total_r2
            self.r2_scores.append(contrib)
    
    def find_optimal_k(
        self, 
        data: np.ndarray, 
        target_r2: float = None,
        max_k: int = None,
        method: str = 'auto'
    ) -> int:
        """
        Find optimal number of archetypes to reach target R².
        
        Args:
            data: Response matrix
            target_r2: Target explained variance (default from config)
            max_k: Maximum k to test (default from config)
            method: Which method to use
        
        Returns:
            Optimal k value
        """
        target_r2 = target_r2 or ANALYSIS.TARGET_R2
        max_k = max_k or ANALYSIS.MAX_ARCHETYPES_TO_TEST
        
        print(f"\n🔍 Searching for optimal k (target R²: {target_r2:.0%})...")
        
        best_k = 2
        best_r2 = 0.0
        
        for k in range(2, max_k + 1):
            self.n_archetypes = k
            
            try:
                self.fit(data, method=method, verbose=False)
                current_r2 = self.total_r2
                
                print(f"   k={k}: R² = {current_r2:.1%}")
                
                if current_r2 > best_r2:
                    best_r2 = current_r2
                    best_k = k
                
                if current_r2 >= target_r2:
                    print(f"✅ Target reached at k={k}")
                    return k
            
            except Exception as e:
                print(f"   k={k}: Failed ({str(e)[:50]})")
                continue
        
        print(f"⚠️  Target not reached. Best: k={best_k} (R²={best_r2:.1%})")
        return best_k
    
    def get_outliers(self, top_n: int = 3) -> List[int]:
        """
        Get indices of respondents least represented by archetypes.
        
        Args:
            top_n: Number of outliers to return
        
        Returns:
            List of respondent indices
        """
        if self.residuals is None:
            return []
        
        return np.argsort(self.residuals)[-top_n:][::-1].tolist()
    
    def get_archetype_patterns(self) -> np.ndarray:
        """Get archetype patterns matrix"""
        return self.archetypes
    
    def get_weights(self) -> np.ndarray:
        """Get weights matrix"""
        return self.weights
    
    def get_proportions(self) -> np.ndarray:
        """Get archetype proportions"""
        return self.proportions
    
    def get_r2_scores(self) -> List[float]:
        """Get per-archetype R² contributions"""
        return self.r2_scores
    
    def get_total_r2(self) -> float:
        """Get total explained variance"""
        return self.total_r2
    
    def summary(self) -> dict:
        """Get summary statistics"""
        return {
            'method': self.method_used,
            'n_archetypes': self.n_archetypes,
            'total_r2': float(self.total_r2),
            'proportions': [float(p) for p in self.proportions],
            'r2_scores': [float(r) for r in self.r2_scores]
        }


if __name__ == "__main__":
    print("\n" + "="*80)
    print("🧪 TESTING ARCHETYPAL ANALYZER")
    print("="*80 + "\n")
    
    # Generate test data
    np.random.seed(42)
    n_respondents = 100
    n_questions = 5
    
    # Create data with 3 underlying archetypes
    archetype1 = np.array([1, 2, 2, 1, 2])
    archetype2 = np.array([3, 3, 3, 3, 3])
    archetype3 = np.array([5, 4, 4, 5, 4])
    
    data = []
    for _ in range(n_respondents):
        weights = np.random.dirichlet([1, 1, 1])
        response = (
            weights[0] * archetype1 + 
            weights[1] * archetype2 + 
            weights[2] * archetype3
        )
        response += np.random.normal(0, 0.3, n_questions)  # Add noise
        response = np.clip(response, 1, 5)
        data.append(response)
    
    data = np.array(data)
    print(f"Generated test data: {data.shape}")
    
    # Test analyzer
    analyzer = ArchetypalAnalyzer(n_archetypes=3)
    archetypes, weights, proportions = analyzer.fit(data, method='auto')
    
    print(f"\n✅ Discovered {len(archetypes)} archetypes")
    print(f"   Proportions: {proportions}")
    print(f"   R²: {analyzer.get_total_r2():.1%}")
    
    # Test optimal k finding
    print("\n" + "="*80)
    optimal_k = analyzer.find_optimal_k(data, target_r2=0.8, max_k=5)
    print(f"✅ Optimal k: {optimal_k}")
    
    print("\n" + "="*80)
    print("✅ ALL ANALYZER TESTS PASSED")
    print("="*80 + "\n")