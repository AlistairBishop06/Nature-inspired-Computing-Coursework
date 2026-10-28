#each trial uses 10,000 fitness evaluations.

from __future__ import annotations

import argparse
import csv
import random
from dataclasses import dataclass
from typing import Iterable, Sequence, Tuple


#the four parameter settings required by the spec.
REQUIRED_SETTINGS: tuple[tuple[int, float], ...] = (
    (50, 0.05),
    (50, 0.30),
    (10, 0.05),
    (10, 0.30),
)

ExperimentSetting = Tuple[str, int, float]

REQUIRED_EXPERIMENT_PLAN: tuple[ExperimentSetting, ...] = tuple(
    (problem_name, population_size, evaporation_rate)
    for problem_name in ("BPP1", "BPP2")
    for population_size, evaporation_rate in REQUIRED_SETTINGS
)

ADDITIONAL_EXPERIMENT_PLAN: tuple[ExperimentSetting, ...] = (
    ("BPP1", 25, 0.15),
    ("BPP2", 25, 0.15),
    ("BPP1", 10, 0.15),
    ("BPP1", 10, 0.45),
    ("BPP1", 10, 0.60),
    ("BPP2", 50, 0.15),
    ("BPP2", 50, 0.45),
    ("BPP2", 50, 0.60),
)

REQUIRED_BASE_SEED = 20_261_111
ADDITIONAL_BASE_SEED = 20_261_151


@dataclass(frozen=True)
class Problem:
    #singular problem instance

    name: str
    item_weights: tuple[float, ...]
    number_of_bins: int

    @property
    def number_of_items(self) -> int:
        #returns the number of items from the problem instance

        return len(self.item_weights)


@dataclass(frozen=True)
class TrialResult:
    #records the results of one ACO trial

    problem: str
    population_size: int
    evaporation_rate: float
    seed: int
    evaluations: int
    final_population_best_fitness: float
    best_fitness_seen: float


def make_problem(problem_name: str) -> Problem:
    #function to create the problem brief
    name = problem_name.upper()

    if name == "BPP1":
        # w(i) = 2i + 1, for i = 1, ..., 600; b = 12.
        weights = tuple(float(2 * i + 1) for i in range(1, 601))
        return Problem(name="BPP1", item_weights=weights, number_of_bins=12)

    if name == "BPP2":
        # w(i) = i^2 / 4, for i = 1, ..., 400; b = 40.
        weights = tuple(float(i * i) / 4.0 for i in range(1, 401))
        return Problem(name="BPP2", item_weights=weights, number_of_bins=40)

    raise ValueError("problem_name must be 'BPP1' or 'BPP2'")


def route_fitness(
    route: Sequence[int], item_weights: Sequence[float], number_of_bins: int
) -> float:
    #calculate the fitness of a route, where a route contains one bin number for every item
    #fitness is the difference between the biggest and smallest load

    bin_loads = [0.0] * number_of_bins

    for item_index, bin_index in enumerate(route):
        bin_loads[bin_index] += item_weights[item_index]

    return max(bin_loads) - min(bin_loads)


def choose_bin(pheromone_row: Sequence[float], rng: random.Random) -> int:
    #selects a bin using a roulette wheel style selection
    #the probability of selecting a bin is proportional to its pheromone value.

    total_pheromone = sum(pheromone_row)

    #this is only a defensive fallback for numerical underflow or if a all-zero row is somehow supplied
    if total_pheromone <= 0.0:
        return rng.randrange(len(pheromone_row))

    selection_point = rng.random() * total_pheromone
    cumulative_pheromone = 0.0

    for bin_index, pheromone in enumerate(pheromone_row):
        cumulative_pheromone += pheromone
        if selection_point < cumulative_pheromone:
            return bin_index

    #a floating-point round-off could leave selection_point just beyond the final cumulative value. The final bin is the correct boundary choice.
    return len(pheromone_row) - 1


def construct_route(
    pheromone: Sequence[Sequence[float]],
    number_of_bins: int,
    rng: random.Random,
) -> list[int]:
    #constructs one ant route from the start to end nodes
    #Each row corresponds to one item and each column to a possible bin. The route can then make one pheromone-biased decision per item.

    route: list[int] = []

    for pheromone_row in pheromone:
        if len(pheromone_row) != number_of_bins:
            raise ValueError("pheromone graph has an invalid number of bins")
        route.append(choose_bin(pheromone_row, rng))

    return route


def update_pheromone(
    pheromone: list[list[float]],
    routes_and_fitnesses: Iterable[tuple[Sequence[int], float]],
    evaporation_rate: float,
) -> None:
    #reward ant routes and remove the edges
    #for an ant with fitness d, every edge in its route receives the specified deposit 100/(d+1).  All pheromone values are then multiplied by (1-e).

    for route, fitness in routes_and_fitnesses:
        deposit = 100.0 / (fitness + 1.0)

        for item_index, bin_index in enumerate(route):
            pheromone[item_index][bin_index] += deposit

    retained_pheromone = 1.0 - evaporation_rate

    for row in pheromone:
        for bin_index in range(len(row)):
            row[bin_index] *= retained_pheromone


def run_trial(
    problem: Problem,
    population_size: int,
    evaporation_rate: float,
    seed: int,
    max_evaluations: int = 10000,
) -> TrialResult:
    #run a full ACO trial and return results

    #the reported result is the best ant in the final population

    if population_size <= 0:
        raise ValueError("population_size must be positive")
    if not 0.0 <= evaporation_rate <= 1.0:
        raise ValueError("evaporation_rate must be between 0 and 1")
    if max_evaluations <= 0:
        raise ValueError("max_evaluations must be positive")

    rng = random.Random(seed)

    #the construction graph has one row per item and one column per bin
    #pheromone is initialised randomly in the interval (0, 1)
    pheromone = [
        [rng.random() for _ in range(problem.number_of_bins)]
        for _ in range(problem.number_of_items)
    ]

    evaluations = 0
    best_fitness_seen = float("inf")
    final_population_best_fitness = float("inf")

    while evaluations < max_evaluations:
        #the required population sizes divide 10,000,  min() also keeps the function correct for a custom evaluation limit.
        current_population_size = min(population_size, max_evaluations - evaluations)
        routes_and_fitnesses: list[tuple[list[int], float]] = []

        for _ in range(current_population_size):
            route = construct_route(pheromone, problem.number_of_bins, rng)
            fitness = route_fitness(
                route, problem.item_weights, problem.number_of_bins
            )
            routes_and_fitnesses.append((route, fitness))

        evaluations += current_population_size
        final_population_best_fitness = min(
            fitness for _, fitness in routes_and_fitnesses
        )
        best_fitness_seen = min(best_fitness_seen, final_population_best_fitness)

        #the required order is reward/update first, followed by evaporation.
        update_pheromone(pheromone, routes_and_fitnesses, evaporation_rate)

    return TrialResult(
        problem=problem.name,
        population_size=population_size,
        evaporation_rate=evaporation_rate,
        seed=seed,
        evaluations=evaluations,
        final_population_best_fitness=final_population_best_fitness,
        best_fitness_seen=best_fitness_seen,
    )


def run_experiment_plan(
    experiment_plan: Sequence[ExperimentSetting],
    trials_per_setting: int = 5,
    base_seed: int = REQUIRED_BASE_SEED,
    max_evaluations: int = 10_000,
    show_progress: bool = False,
) -> list[TrialResult]:
    #run all experiments - every trial receives a distinct deterministic seed derived from ``base_seed``
    #this makes runs reproducible while still ensuring that different trials use different random number seeds.

    if trials_per_setting <= 0:
        raise ValueError("trials_per_setting must be positive")
    if not experiment_plan:
        raise ValueError("experiment_plan must contain at least one setting")

    results: list[TrialResult] = []
    trial_number = 0
    total_trials = len(experiment_plan) * trials_per_setting

    if show_progress:
        print(f"Running {total_trials} trials...", flush=True)

    for problem_name, population_size, evaporation_rate in experiment_plan:
        problem = make_problem(problem_name)

        for _ in range(trials_per_setting):
            #each trial has a different seed, including trials from different problems and parameter settings.
            seed = base_seed + trial_number
            results.append(
                run_trial(
                    problem=problem,
                    population_size=population_size,
                    evaporation_rate=evaporation_rate,
                    seed=seed,
                    max_evaluations=max_evaluations,
                )
            )
            trial_number += 1

            if show_progress:
                print(
                    f"Completed trial {trial_number}/{total_trials} "
                    f"({problem_name}, p={population_size}, "
                    f"e={evaporation_rate:.2f})",
                    flush=True,
                )

    return results


def run_required_experiments(
    trials_per_setting: int = 5,
    base_seed: int = REQUIRED_BASE_SEED,
    max_evaluations: int = 10_000,
    show_progress: bool = False,
) -> list[TrialResult]:
    return run_experiment_plan(
        REQUIRED_EXPERIMENT_PLAN,
        trials_per_setting=trials_per_setting,
        base_seed=base_seed,
        max_evaluations=max_evaluations,
        show_progress=show_progress,
    )


def run_additional_experiments(
    trials_per_setting: int = 5,
    base_seed: int = ADDITIONAL_BASE_SEED,
    max_evaluations: int = 10_000,
    show_progress: bool = False,
) -> list[TrialResult]:
    return run_experiment_plan(
        ADDITIONAL_EXPERIMENT_PLAN,
        trials_per_setting=trials_per_setting,
        base_seed=base_seed,
        max_evaluations=max_evaluations,
        show_progress=show_progress,
    )


def print_trial_results(results: Sequence[TrialResult]) -> None:
    #print a compact table containing every trial result

    headings = (
        "Problem",
        "p",
        "e",
        "Trial seed",
        "Evaluations",
        "Final population best",
        "Best seen",
    )
    rows = [
        (
            result.problem,
            str(result.population_size),
            f"{result.evaporation_rate:.2f}",
            str(result.seed),
            str(result.evaluations),
            f"{result.final_population_best_fitness:.2f}",
            f"{result.best_fitness_seen:.2f}",
        )
        for result in results
    ]
    widths = [
        max(len(headings[index]), *(len(row[index]) for row in rows))
        for index in range(len(headings))
    ]

    print(" | ".join(headings[index].ljust(widths[index]) for index in range(len(headings))))
    print("-+-".join("-" * width for width in widths))

    for row in rows:
        print(" | ".join(row[index].ljust(widths[index]) for index in range(len(row))))


def write_csv(results: Sequence[TrialResult], output_path: str) -> None:
    #write trial-level results to a CSV file for report tables and graphs.

    fieldnames = [
        "problem",
        "population_size",
        "evaporation_rate",
        "seed",
        "evaluations",
        "final_population_best_fitness",
        "best_fitness_seen",
    ]

    with open(output_path, "w", newline="", encoding="utf-8") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=fieldnames)
        writer.writeheader()
        for result in results:
            writer.writerow(
                {
                    "problem": result.problem,
                    "population_size": result.population_size,
                    "evaporation_rate": result.evaporation_rate,
                    "seed": result.seed,
                    "evaluations": result.evaluations,
                    "final_population_best_fitness": result.final_population_best_fitness,
                    "best_fitness_seen": result.best_fitness_seen,
                }
            )


def parse_arguments() -> argparse.Namespace:
    #parse command-line options for the required experiment runner

    parser = argparse.ArgumentParser(
        description="Run the ECM3412 ACO bin-packing experiments."
    )
    parser.add_argument(
        "--experiment",
        choices=("required", "additional"),
        default="required",
        help=(
            "experiment plan to run: the required grid or the additional "
            "midpoint and evaporation sweep (default: required)"
        ),
    )
    parser.add_argument(
        "--trials",
        type=int,
        default=5,
        help="number of trials per problem and parameter setting (default: 5)",
    )
    parser.add_argument(
        "--evaluations",
        type=int,
        default=10_000,
        help="fitness evaluations per trial (default: 10000)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        help=(
            "optional base seed; defaults to 20261111 for required trials "
            "and 20261151 for additional trials"
        ),
    )
    parser.add_argument(
        "--csv",
        metavar="PATH",
        help="optional path for saving trial results as CSV",
    )
    return parser.parse_args()


def main() -> None:
    #run and display the coursework experiments

    arguments = parse_arguments()
    if arguments.experiment == "required":
        base_seed = (
            REQUIRED_BASE_SEED if arguments.seed is None else arguments.seed
        )
        results = run_required_experiments(
            trials_per_setting=arguments.trials,
            base_seed=base_seed,
            max_evaluations=arguments.evaluations,
            show_progress=True,
        )
    else:
        base_seed = (
            ADDITIONAL_BASE_SEED if arguments.seed is None else arguments.seed
        )
        results = run_additional_experiments(
            trials_per_setting=arguments.trials,
            base_seed=base_seed,
            max_evaluations=arguments.evaluations,
            show_progress=True,
        )
    print_trial_results(results)

    if arguments.csv:
        write_csv(results, arguments.csv)
        print(f"\nResults written to {arguments.csv}")


if __name__ == "__main__":
    main()
