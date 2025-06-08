#!/usr/bin/env python3
"""
Employee-scheduler solver using Google OR-Tools CP-SAT.

Usage:
    python algorithms/scheduler_lp.py input.json output.json

Input JSON shape:
{
  "employees": [
    { "name": "Kyle", "weekday_cost": [1,2,3,4,5,6,7] }
  ],
  "dates": [
    { "date": "2025-06-03", "weekday": 1 }   # weekday: 0 = Sunday … 6 = Saturday
  ]
}

Output JSON:  [{ "date": "YYYY-MM-DD", "employee": "Name" }, …]
"""

import json
import sys
from pathlib import Path
from typing import Dict, List
import time

from ortools.sat.python import cp_model


def configure_solver_optimally(solver, problem_size):
    """Configure solver based on problem characteristics"""
    
    num_variables = problem_size['variables']
    num_constraints = problem_size['constraints']
    
    if num_variables > 1000:
        solver.parameters.max_time_in_seconds = 60
        solver.parameters.num_search_workers = min(4, 8)  # Parallel search, but not too many
    else:  
        solver.parameters.max_time_in_seconds = 30
        
    # Use better search strategy for scheduling problems
    solver.parameters.search_branching = cp_model.PORTFOLIO_SEARCH
    solver.parameters.cp_model_presolve = True
    solver.parameters.symmetry_level = 2  # Detect symmetries
    
    # Early termination if good enough solution found
    solver.parameters.optimize_with_core = True


def solve_with_base_constraints(payload: Dict) -> List[Dict]:
    """Solve with the standard constraint set"""
    employees = payload["employees"]
    dates = payload["dates"]
    max_consecutive_days = payload.get("max_consecutive_days", 2)

    num_emp = len(employees)
    num_days = len(dates)

    model = cp_model.CpModel()

    # Decision variables: x[i][d] == 1 if employee i works day d
    x = [
        [model.NewBoolVar(f"x_{i}_{d}") for d in range(num_days)]
        for i in range(num_emp)
    ]

    # C1: Exactly one employee per day
    for d in range(num_days):
        model.Add(sum(x[i][d] for i in range(num_emp)) == 1)

    # C2: No employee works > X days in a row
    for i in range(num_emp):
        for d in range(num_days - max_consecutive_days):
            model.Add(sum(x[i][d + k] for k in range(max_consecutive_days + 1)) <= max_consecutive_days)

    # C3: Balanced workload (original logic - more reliable)
    min_load = num_days // num_emp if num_emp > 0 else num_days
    max_load = (num_days + num_emp - 1) // num_emp if num_emp > 0 else num_days
    
    for i in range(num_emp):
        total_days_worked_by_emp = sum(x[i][d] for d in range(num_days))
        model.Add(total_days_worked_by_emp >= min_load)
        model.Add(total_days_worked_by_emp <= max_load)

    # Precompute costs and preference day indices
    emp_rank1_day_indices = [[] for _ in range(num_emp)]
    emp_top3_pref_day_indices = [[] for _ in range(num_emp)]
    cost_matrix = [[0] * num_days for _ in range(num_emp)]

    for i, emp in enumerate(employees):
        for d, date_info in enumerate(dates):
            weekday = date_info["weekday"]
            cost = emp["weekday_cost"][weekday]
            cost_matrix[i][d] = cost
            if cost == 0:  # Rank 1 preference
                emp_rank1_day_indices[i].append(d)
            if cost <= 40:  # Top 3 preferences (costs 0, 20, 40)
                emp_top3_pref_day_indices[i].append(d)

    # C4: Each employee must get at least one of their rank-1 days (if they have any)
    for i in range(num_emp):
        if emp_rank1_day_indices[i]:
            model.Add(sum(x[i][d] for d in emp_rank1_day_indices[i]) >= 1)

    # C5: Each employee should get at least one top-3 preference day (if possible)
    MIN_TOP_N_DAYS_GUARANTEE = 1
    actual_min_top_n_days = min(MIN_TOP_N_DAYS_GUARANTEE, min_load)
    if actual_min_top_n_days > 0:
        for i in range(num_emp):
            distinct_top3_pref_days = set(emp_top3_pref_day_indices[i])
            if len(distinct_top3_pref_days) >= actual_min_top_n_days:
                model.Add(sum(x[i][d] for d in distinct_top3_pref_days) >= actual_min_top_n_days)

    return model, x, cost_matrix, emp_rank1_day_indices


def solve_with_relaxed_constraints(payload: Dict) -> List[Dict]:
    """Solve with relaxed constraints when base version fails"""
    employees = payload["employees"]
    dates = payload["dates"]
    max_consecutive_days = payload.get("max_consecutive_days", 2)

    num_emp = len(employees)
    num_days = len(dates)

    model = cp_model.CpModel()

    # Decision variables
    x = [
        [model.NewBoolVar(f"x_{i}_{d}") for d in range(num_days)]
        for i in range(num_emp)
    ]

    # C1: Exactly one employee per day (never relax this)
    for d in range(num_days):
        model.Add(sum(x[i][d] for i in range(num_emp)) == 1)

    # C2: Relaxed consecutive days constraint (increase limit by 1)
    relaxed_consecutive = max_consecutive_days + 1
    for i in range(num_emp):
        for d in range(num_days - relaxed_consecutive):
            model.Add(sum(x[i][d + k] for k in range(relaxed_consecutive + 1)) <= relaxed_consecutive)

    # C3: More flexible workload balancing
    min_load = max(1, (num_days // num_emp) - 1)  # Allow one less day
    max_load = (num_days + num_emp - 1) // num_emp + 1  # Allow one more day
    
    for i in range(num_emp):
        total_days_worked_by_emp = sum(x[i][d] for d in range(num_days))
        model.Add(total_days_worked_by_emp >= min_load)
        model.Add(total_days_worked_by_emp <= max_load)

    # Precompute costs
    cost_matrix = [[0] * num_days for _ in range(num_emp)]
    emp_rank1_day_indices = [[] for _ in range(num_emp)]
    
    for i, emp in enumerate(employees):
        for d, date_info in enumerate(dates):
            weekday = date_info["weekday"]
            cost = emp["weekday_cost"][weekday]
            cost_matrix[i][d] = cost
            if cost == 0:
                emp_rank1_day_indices[i].append(d)

    # C4: Relaxed - try to give rank-1 days but don't require it
    # (Remove hard constraint, let objective function handle it)

    return model, x, cost_matrix, emp_rank1_day_indices


def create_improved_objective(model, x, cost_matrix, emp_rank1_day_indices, num_emp, num_days):
    """Create an improved objective function with better weights"""
    
    # Calculate objective components
    employee_total_costs = []
    rank1_assignments = []
    
    for i in range(num_emp):
        # Total cost for employee i
        emp_cost = sum(cost_matrix[i][d] * x[i][d] for d in range(num_days))
        employee_total_costs.append(emp_cost)
        
        # Count rank-1 assignments for employee i
        if emp_rank1_day_indices[i]:
            rank1_count = sum(x[i][d] for d in emp_rank1_day_indices[i])
            rank1_assignments.append(rank1_count)

    # P1: Minimize maximum individual cost (fairness)
    max_individual_cost = model.NewIntVar(0, num_days * 100, "max_individual_cost")
    for cost_var in employee_total_costs:
        model.Add(cost_var <= max_individual_cost)

    # P2: Minimize total cost (efficiency)
    total_cost = sum(employee_total_costs)
    
    # P3: Maximize total rank-1 assignments
    total_rank1_assignments = sum(rank1_assignments) if rank1_assignments else 0

    # Improved objective with better balanced weights
    W_FAIRNESS = 1000     # Minimize max individual cost
    W_EFFICIENCY = 10     # Minimize total cost
    W_RANK1_BONUS = 500   # Maximize rank-1 assignments
    
    if rank1_assignments:
        model.Minimize(
            W_FAIRNESS * max_individual_cost +
            W_EFFICIENCY * total_cost -
            W_RANK1_BONUS * total_rank1_assignments
        )
    else:
        model.Minimize(
            W_FAIRNESS * max_individual_cost +
            W_EFFICIENCY * total_cost
        )

    return model


def solve_with_strategy(payload: Dict, relaxed: bool = False) -> List[Dict]:
    """Solve using either base or relaxed constraints"""
    
    if relaxed:
        model, x, cost_matrix, emp_rank1_day_indices = solve_with_relaxed_constraints(payload)
    else:
        model, x, cost_matrix, emp_rank1_day_indices = solve_with_base_constraints(payload)
    
    employees = payload["employees"]
    dates = payload["dates"]
    num_emp = len(employees)
    num_days = len(dates)
    
    # Add improved objective function
    model = create_improved_objective(model, x, cost_matrix, emp_rank1_day_indices, num_emp, num_days)
    
    # Configure and solve
    solver = cp_model.CpSolver()
    configure_solver_optimally(solver, {'variables': num_emp * num_days, 'constraints': num_emp + num_days})
    
    status = solver.Solve(model)
    
    if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        # Build output
        schedule = []
        for d_idx, date_info in enumerate(dates):
            assigned_emp_name = "<UNASSIGNED>"
            for emp_idx, emp in enumerate(employees):
                if solver.Value(x[emp_idx][d_idx]) == 1:
                    assigned_emp_name = emp["name"]
                    break
            schedule.append({
                "date": date_info["date"], 
                "employee": assigned_emp_name, 
                "weekday": date_info["weekday"]
            })
        return schedule
    
    return None


def solve(payload: Dict) -> List[Dict]:
    """Main solve function with automatic constraint relaxation"""
    
    employees = payload["employees"]
    dates = payload["dates"]
    num_emp = len(employees)
    num_days = len(dates)
    
    # Quick feasibility checks
    if num_emp == 0:
        raise RuntimeError("No employees provided.")
    
    if num_days == 0:
        raise RuntimeError("No dates provided.")
    
    if num_days < num_emp:
        raise RuntimeError(f"There are {num_days} days to schedule but {num_emp} employees. Each employee needs at least one day, but there aren't enough days to go around.")
    
    # Check if any employee has no available days
    for i, emp in enumerate(employees):
        has_available_day = False
        for date_info in dates:
            weekday = date_info["weekday"]
            if emp["weekday_cost"][weekday] < 1000:  # Available day
                has_available_day = True
                break
        if not has_available_day:
            raise RuntimeError(f"Employee '{emp['name']}' has no available days to work based on their preferences and the selected schedulable days.")
    
    # Strategy 1: Try with base constraints
    try:
        result = solve_with_strategy(payload, relaxed=False)
        if result:
            return result
    except Exception:
        pass
    
    # Strategy 2: Try with relaxed constraints
    try:
        result = solve_with_strategy(payload, relaxed=True)
        if result:
            return result
    except Exception:
        pass
    
    # Strategy 3: Try with very relaxed constraints (no consecutive limit, flexible workload)
    try:
        employees = payload["employees"]
        dates = payload["dates"]
        num_emp = len(employees)
        num_days = len(dates)

        model = cp_model.CpModel()
        x = [
            [model.NewBoolVar(f"x_{i}_{d}") for d in range(num_days)]
            for i in range(num_emp)
        ]

        # Only enforce: one employee per day
        for d in range(num_days):
            model.Add(sum(x[i][d] for i in range(num_emp)) == 1)

        # Very flexible workload: just ensure everyone gets at least 1 day if possible
        for i in range(num_emp):
            if num_days >= num_emp:  # Only if there are enough days
                model.Add(sum(x[i][d] for d in range(num_days)) >= 1)

        # Simple objective: minimize total cost
        cost_matrix = [[0] * num_days for _ in range(num_emp)]
        for i, emp in enumerate(employees):
            for d, date_info in enumerate(dates):
                weekday = date_info["weekday"]
                cost_matrix[i][d] = emp["weekday_cost"][weekday]

        total_cost = sum(cost_matrix[i][d] * x[i][d] for i in range(num_emp) for d in range(num_days))
        model.Minimize(total_cost)

        solver = cp_model.CpSolver()
        solver.parameters.max_time_in_seconds = 30
        status = solver.Solve(model)

        if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
            schedule = []
            for d_idx, date_info in enumerate(dates):
                assigned_emp_name = "<UNASSIGNED>"
                for emp_idx, emp in enumerate(employees):
                    if solver.Value(x[emp_idx][d_idx]) == 1:
                        assigned_emp_name = emp["name"]
                        break
                schedule.append({
                    "date": date_info["date"], 
                    "employee": assigned_emp_name, 
                    "weekday": date_info["weekday"]
                })
            return schedule
            
    except Exception:
        pass
    
    # If all strategies fail
    error_message = "Unable to find a valid schedule with current constraints. This could be due to:\n"
    error_message += "1. Not enough days to satisfy everyone's preferences\n"
    error_message += "2. Too many consecutive work day restrictions\n"
    error_message += "3. Conflicting employee availability patterns\n"
    error_message += "Try adjusting the date range, excluded dates, or employee preferences."
    
    raise RuntimeError(error_message)


# ----------------------------------------------------------------------
# CLI wrapper
# ----------------------------------------------------------------------
if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python algorithms/scheduler_lp.py input.json output.json")
        sys.exit(1)

    in_path, out_path = map(Path, sys.argv[1:3])
    payload = json.loads(in_path.read_text())
    out = solve(payload)
    out_path.write_text(json.dumps(out, indent=2))
    print(f"Wrote {len(out)} assignments to {out_path}")

 