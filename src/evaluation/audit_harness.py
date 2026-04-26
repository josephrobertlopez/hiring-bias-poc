import csv
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
from scipy import stats

@dataclass
class BiasDetectionTestResult:
    algorithm_name: str
    task_name: str
    run_id: int
    auc: float
    disparate_impact: float
    flip_rate: float
    explanation_coverage: float

class AlgorithmAuditHarness:
    def __init__(self, algorithms: Dict[str, Any], tasks: Dict[str, Any]):
        self.algorithms = algorithms
        self.tasks = tasks
        self.results_cache = {}

    def run_comparison(self, n_runs: int = 5) -> List[BiasDetectionTestResult]:
        results = []
        seed_sequence = [42, 123, 456, 789, 999][:n_runs]
        run_id = 0

        for algorithm_name, algorithm in self.algorithms.items():
            for task_name, task in self.tasks.items():
                for seed in seed_sequence:
                    run_id += 1
                    try:
                        auc, disparate_impact, flip_rate, explanation_coverage = algorithm.run(task, seed)
                        result = BiasDetectionTestResult(
                            algorithm_name=algorithm_name,
                            task_name=task_name,
                            run_id=run_id,
                            auc=auc,
                            disparate_impact=disparate_impact,
                            flip_rate=flip_rate,
                            explanation_coverage=explanation_coverage
                        )
                        results.append(result)
                    except Exception as e:
                        print(f"Failed to run {algorithm_name} on {task_name} with seed {seed}: {e}")
        return results

    def export_csv(self, results: List[BiasDetectionTestResult], filepath: str):
        with open(filepath, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['Algorithm', 'Task', 'Run_ID', 'AUC', 'DI', 'Flip_Rate', 'Explanation_Coverage'])
            for result in results:
                writer.writerow([
                    result.algorithm_name,
                    result.task_name,
                    result.run_id,
                    result.auc,
                    result.disparate_impact,
                    result.flip_rate,
                    result.explanation_coverage
                ])
            # Summary row per algorithm-task
            grouped = self._group_results_by_algorithm_task(results)
            for (alg, task), group in grouped.items():
                auc_mean = np.mean([r.auc for r in group])
                di_mean = np.mean([r.disparate_impact for r in group])
                flip_rate_mean = np.mean([r.flip_rate for r in group])
                cov_mean = np.mean([r.explanation_coverage for r in group])
                writer.writerow(['SUMMARY', f'{alg}_{task}', '', auc_mean, di_mean, flip_rate_mean, cov_mean])

    def compute_statistical_significance(self, results: List[BiasDetectionTestResult]) -> Dict:
        significance_results = {}
        grouped_by_task = self._group_results_by_task(results)

        for task_name, task_results in grouped_by_task.items():
            grouped_by_alg = {}
            for result in task_results:
                if result.algorithm_name not in grouped_by_alg:
                    grouped_by_alg[result.algorithm_name] = []
                grouped_by_alg[result.algorithm_name].append(result)

            algorithms = list(grouped_by_alg.keys())
            n_comparisons = len(algorithms) * (len(algorithms) - 1) // 2
            bonferroni_threshold = 0.05 / max(1, n_comparisons)

            for i, alg1 in enumerate(algorithms):
                for alg2 in algorithms[i+1:]:
                    auc1 = np.array([r.auc for r in grouped_by_alg[alg1]])
                    auc2 = np.array([r.auc for r in grouped_by_alg[alg2]])

                    if len(auc1) > 1 and len(auc2) > 1:
                        t_stat, p_value = stats.ttest_rel(auc1, auc2)
                        effect_size = np.mean(auc1 - auc2) / (np.std(auc1 - auc2) + 1e-10)

                        significance_results[(alg1, alg2, task_name)] = {
                            't_stat': float(t_stat),
                            'p_value': float(p_value),
                            'effect_size': float(effect_size),
                            'significant': p_value < bonferroni_threshold
                        }
        return significance_results

    def identify_pareto_frontier(self, results: List[BiasDetectionTestResult]) -> List[str]:
        pareto_frontier = set()
        grouped_by_task = self._group_results_by_task(results)

        for task_name, task_results in grouped_by_task.items():
            # Group by algorithm
            grouped_by_alg = {}
            for result in task_results:
                if result.algorithm_name not in grouped_by_alg:
                    grouped_by_alg[result.algorithm_name] = []
                grouped_by_alg[result.algorithm_name].append(result)

            # Compute mean AUC and DI per algorithm
            algo_stats = {}
            for algo_name, algo_results in grouped_by_alg.items():
                mean_auc = np.mean([r.auc for r in algo_results])
                mean_di = np.mean([r.disparate_impact for r in algo_results])
                algo_stats[algo_name] = (mean_auc, mean_di)

            # Identify non-dominated algorithms
            for algo1 in algo_stats:
                auc1, di1 = algo_stats[algo1]
                dominated = False
                for algo2 in algo_stats:
                    if algo1 != algo2:
                        auc2, di2 = algo_stats[algo2]
                        # Dominated if another algo has >= AUC and <= DI
                        if auc2 >= auc1 and di2 <= di1 and (auc2 > auc1 or di2 < di1):
                            dominated = True
                            break
                if not dominated:
                    pareto_frontier.add(algo1)

        return list(pareto_frontier)

    def _group_results_by_algorithm_task(self, results: List[BiasDetectionTestResult]) -> Dict[Tuple[str, str], List[BiasDetectionTestResult]]:
        grouped = {}
        for result in results:
            key = (result.algorithm_name, result.task_name)
            if key not in grouped:
                grouped[key] = []
            grouped[key].append(result)
        return grouped

    def _group_results_by_task(self, results: List[BiasDetectionTestResult]) -> Dict[str, List[BiasDetectionTestResult]]:
        grouped = {}
        for result in results:
            if result.task_name not in grouped:
                grouped[result.task_name] = []
            grouped[result.task_name].append(result)
        return grouped
