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
        
    # PHASE 2 IMPROVEMENT: Better search strategy for preference optimization
    solver.parameters.search_branching = cp_model.PORTFOLIO_SEARCH
    solver.parameters.cp_model_presolve = True
    solver.parameters.symmetry_level = 2  # Detect symmetries
    
    # Focus on finding good solutions quickly rather than proving optimality
    solver.parameters.optimize_with_core = True
    solver.parameters.linearization_level = 2  # Better linearization for complex objectives
    
    # PHASE 2: Allow more time to find preference-optimal solutions
    solver.parameters.relative_gap_limit = 0.01  # Accept solutions within 1% of optimal
    solver.parameters.absolute_gap_limit = 10    # Accept if gap is small in absolute terms


# Legacy constraint functions removed in Phase 3 optimization
# These have been replaced by the adaptive constraint approach in build_adaptive_model()


def create_adaptive_objective(model, x, cost_matrix, workload_penalty_vars, consecutive_penalty_vars, preference_balance_penalty_vars, min_top3_guarantee_penalty_vars, preference_equity_penalty_vars, num_emp, num_days):
    """PHASE 4: Enhanced adaptive objective with preference balancing and equity"""
    
    # Precompute preference day indices for all ranks
    emp_rank1_day_indices = [[] for _ in range(num_emp)]  # cost = 0
    emp_rank2_day_indices = [[] for _ in range(num_emp)]  # cost = 20
    emp_rank3_day_indices = [[] for _ in range(num_emp)]  # cost = 40
    
    for i in range(num_emp):
        for d in range(num_days):
            cost = cost_matrix[i][d]
            if cost == 0:
                emp_rank1_day_indices[i].append(d)
            elif cost == 20:
                emp_rank2_day_indices[i].append(d)
            elif cost == 40:
                emp_rank3_day_indices[i].append(d)
    
    # Calculate preference satisfaction
    total_rank1_assignments = 0
    total_rank2_assignments = 0
    total_rank3_assignments = 0
    total_cost = 0
    
    for i in range(num_emp):
        # Total cost for employee i
        total_cost += sum(cost_matrix[i][d] * x[i][d] for d in range(num_days))
        
        # Count preference assignments
        if emp_rank1_day_indices[i]:
            total_rank1_assignments += sum(x[i][d] for d in emp_rank1_day_indices[i])
        if emp_rank2_day_indices[i]:
            total_rank2_assignments += sum(x[i][d] for d in emp_rank2_day_indices[i])
        if emp_rank3_day_indices[i]:
            total_rank3_assignments += sum(x[i][d] for d in emp_rank3_day_indices[i])

    # WORKLOAD DISTRIBUTION FIXED: Now guaranteed by hard constraints
    W_PREFERENCE_COST = 1           # Basic cost (keep low for efficiency)
    W_RANK1_BONUS = 2000            # High priority for top preferences
    W_RANK2_BONUS = 1000            # Strong bonus for rank 2
    W_RANK3_BONUS = 500             # Good bonus for rank 3
    W_WORKLOAD_PENALTY = 0          # *** NO LONGER NEEDED *** - Workload guaranteed by hard constraints
    W_CONSECUTIVE_PENALTY = 500     # Penalty for consecutive violations
    W_PREFERENCE_BALANCE = 800      # Penalty for unfair rank-1 distribution
    W_TOP3_GUARANTEE = 1200         # Penalty for not getting top-3 preferences
    W_PREFERENCE_EQUITY = 400       # Penalty for preference satisfaction variance

    # Enhanced objective that prioritizes fairness and balanced preference satisfaction
    objective_terms = [
        W_PREFERENCE_COST * total_cost,
        -W_RANK1_BONUS * total_rank1_assignments,
        -W_RANK2_BONUS * total_rank2_assignments,
        -W_RANK3_BONUS * total_rank3_assignments,
        W_WORKLOAD_PENALTY * sum(workload_penalty_vars),
        W_CONSECUTIVE_PENALTY * sum(consecutive_penalty_vars)
    ]
    
    # Add Phase 4 balancing penalties
    if preference_balance_penalty_vars:
        objective_terms.append(W_PREFERENCE_BALANCE * sum(preference_balance_penalty_vars))
    
    if min_top3_guarantee_penalty_vars:
        objective_terms.append(W_TOP3_GUARANTEE * sum(min_top3_guarantee_penalty_vars))
    
    if preference_equity_penalty_vars:
        objective_terms.append(W_PREFERENCE_EQUITY * sum(preference_equity_penalty_vars))

    model.Minimize(sum(objective_terms))
    return model


def build_adaptive_model(payload: Dict) -> tuple:
    """PHASE 3: Build a single adaptive model with enhanced workload distribution"""
    
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

    # Precompute cost matrix and availability
    cost_matrix = [[0] * num_days for _ in range(num_emp)]
    MAX_ALLOWED_COST = 60
    
    for i, emp in enumerate(employees):
        for d, date_info in enumerate(dates):
            weekday = date_info["weekday"]
            cost_matrix[i][d] = emp["weekday_cost"][weekday]

    # ENHANCED WORKLOAD DISTRIBUTION FEASIBILITY CHECKING
    min_load = num_days // num_emp if num_emp > 0 else num_days
    max_load = (num_days + num_emp - 1) // num_emp if num_emp > 0 else num_days
    
    # Check each employee has enough available days for fair distribution
    for i, emp in enumerate(employees):
        available_days_count = sum(1 for d in range(num_days) 
                                 if cost_matrix[i][d] <= MAX_ALLOWED_COST)
        
        if available_days_count < min_load:
            # ADAPTIVE SOLUTION: Gradually increase cost threshold for this employee
            # until they have enough available days
            adaptive_threshold = MAX_ALLOWED_COST
            while available_days_count < min_load and adaptive_threshold < 100:
                adaptive_threshold += 10
                available_days_count = sum(1 for d in range(num_days) 
                                         if cost_matrix[i][d] <= adaptive_threshold)
            
            if available_days_count < min_load:
                raise RuntimeError(
                    f"Employee '{emp['name']}' only has {available_days_count} "
                    f"reasonably available days, but needs at least {min_load} days "
                    f"for fair distribution. Consider adjusting their preferences or "
                    f"the scheduling period."
                )
            
            # Use the adaptive threshold for this employee
            print(f"Adaptive threshold for {emp['name']}: {adaptive_threshold} "
                  f"(was {MAX_ALLOWED_COST})")
            for d in range(num_days):
                if cost_matrix[i][d] > adaptive_threshold:
                    model.Add(x[i][d] == 0)
        else:
            # Standard availability constraint
            for d in range(num_days):
                if cost_matrix[i][d] > MAX_ALLOWED_COST:
                    model.Add(x[i][d] == 0)

    # HARD CONSTRAINTS (never relax these)
    
    # C1: Exactly one employee per day (fundamental requirement)
    for d in range(num_days):
        model.Add(sum(x[i][d] for i in range(num_emp)) == 1)

    # C2: STRENGTHENED WORKLOAD DISTRIBUTION - ABSOLUTELY FAIR
    print(f"Enforcing workload distribution: min={min_load}, max={max_load} days per employee")
    
    for i in range(num_emp):
        total_days = sum(x[i][d] for d in range(num_days))
        model.Add(total_days >= min_load)  # Minimum days required
        model.Add(total_days <= max_load)  # Maximum days allowed
        
        # For perfect division cases, enforce exact equality
        if num_days % num_emp == 0:
            model.Add(total_days == min_load)  # Exactly equal distribution
    
    # Keep minimal penalty variables for objective function (but workload is now guaranteed)
    workload_penalty_vars = []

    # Consecutive days constraint with penalties
    consecutive_penalty_vars = []
    for i in range(num_emp):
        for d in range(num_days - max_consecutive_days):
            # Penalty for exceeding consecutive limit
            consecutive_violation = model.NewBoolVar(f"consec_penalty_{i}_{d}")
            model.Add(sum(x[i][d + k] for k in range(max_consecutive_days + 1)) <= max_consecutive_days + consecutive_violation)
            consecutive_penalty_vars.append(consecutive_violation)

    # PHASE 4: ADVANCED PREFERENCE BALANCING CONSTRAINTS
    
    # Precompute preference availability for balancing
    emp_rank1_days = [[] for _ in range(num_emp)]
    emp_rank2_days = [[] for _ in range(num_emp)]
    emp_rank3_days = [[] for _ in range(num_emp)]
    
    for i in range(num_emp):
        for d in range(num_days):
            cost = cost_matrix[i][d]
            if cost == 0:
                emp_rank1_days[i].append(d)
            elif cost == 20:
                emp_rank2_days[i].append(d)
            elif cost == 40:
                emp_rank3_days[i].append(d)
    
    # Preference balancing penalty variables
    preference_balance_penalty_vars = []
    
    # P4.1: Fair distribution of rank-1 preferences
    total_available_rank1_slots = sum(len(emp_rank1_days[i]) for i in range(num_emp))
    if total_available_rank1_slots > 0 and num_emp > 1:
        target_rank1_per_employee = total_available_rank1_slots / num_emp
        
        for i in range(num_emp):
            if emp_rank1_days[i]:  # Only for employees who have rank-1 days available
                emp_rank1_assignments = sum(x[i][d] for d in emp_rank1_days[i])
                
                # Penalty for getting too few rank-1 assignments
                rank1_deficit = model.NewIntVar(0, num_days, f"rank1_deficit_{i}")
                ideal_rank1 = max(1, int(target_rank1_per_employee * 0.7))  # At least 70% of fair share
                model.Add(rank1_deficit >= ideal_rank1 - emp_rank1_assignments)
                preference_balance_penalty_vars.append(rank1_deficit)
                
                # Penalty for hogging too many rank-1 assignments
                rank1_excess = model.NewIntVar(0, num_days, f"rank1_excess_{i}")
                max_rank1 = min(len(emp_rank1_days[i]), int(target_rank1_per_employee * 1.5))  # At most 150% of fair share
                model.Add(rank1_excess >= emp_rank1_assignments - max_rank1)
                preference_balance_penalty_vars.append(rank1_excess)

    # P4.2: Ensure everyone gets at least one top-3 preference when possible
    min_top3_guarantee_penalty_vars = []
    for i in range(num_emp):
        emp_top3_days = emp_rank1_days[i] + emp_rank2_days[i] + emp_rank3_days[i]
        if emp_top3_days and min_load > 0:
            emp_top3_assignments = sum(x[i][d] for d in emp_top3_days)
            
            # Penalty for not getting at least one top-3 preference
            top3_deficit = model.NewBoolVar(f"top3_deficit_{i}")
            model.Add(emp_top3_assignments >= 1 - top3_deficit)  # If deficit=1, then top3_assignments can be 0
            min_top3_guarantee_penalty_vars.append(top3_deficit)

    # P4.3: Preference satisfaction equity - minimize variance in preference quality
    preference_equity_penalty_vars = []
    if num_emp > 1:
        # Calculate average preference cost per employee
        emp_avg_costs = []
        for i in range(num_emp):
            total_days_emp = sum(x[i][d] for d in range(num_days))
            total_cost_emp = sum(cost_matrix[i][d] * x[i][d] for d in range(num_days))
            
            # Avoid division by zero: if no days assigned, cost is 0
            avg_cost = model.NewIntVar(0, 60, f"avg_cost_{i}")
            
            # Use conditional constraints to handle division
            # If total_days_emp > 0, then avg_cost = total_cost_emp / total_days_emp (approximated)
            # Otherwise avg_cost = 0
            no_days_assigned = model.NewBoolVar(f"no_days_{i}")
            model.Add(total_days_emp == 0).OnlyEnforceIf(no_days_assigned)
            model.Add(total_days_emp > 0).OnlyEnforceIf(no_days_assigned.Not())
            
            # When days are assigned, approximate average cost
            model.Add(avg_cost == 0).OnlyEnforceIf(no_days_assigned)
            # For simplicity in linear programming, we'll use total cost as proxy for average
            # and normalize in the objective function
            emp_avg_costs.append(total_cost_emp)
        
        # Add equity penalty based on preference cost variance
        max_emp_cost = model.NewIntVar(0, num_days * 60, "max_emp_cost")
        min_emp_cost = model.NewIntVar(0, num_days * 60, "min_emp_cost")
        
        for cost_var in emp_avg_costs:
            model.Add(max_emp_cost >= cost_var)
            model.Add(min_emp_cost <= cost_var)
        
        # Penalty for large cost variance (unfair preference distribution)
        cost_variance_penalty = model.NewIntVar(0, num_days * 60, "cost_variance_penalty")
        model.Add(cost_variance_penalty >= max_emp_cost - min_emp_cost)
        preference_equity_penalty_vars.append(cost_variance_penalty)

    return model, x, cost_matrix, workload_penalty_vars, consecutive_penalty_vars, preference_balance_penalty_vars, min_top3_guarantee_penalty_vars, preference_equity_penalty_vars


def validate_and_report_workload_distribution(schedule: List[Dict], employees: List[Dict]) -> None:
    """Validate and report on workload distribution fairness"""
    
    # Count days per employee
    days_per_employee = {emp["name"]: 0 for emp in employees}
    
    for assignment in schedule:
        employee_name = assignment["employee"]
        if employee_name in days_per_employee:
            days_per_employee[employee_name] += 1
    
    total_days = len(schedule)
    num_employees = len(employees)
    expected_days_per_employee = total_days / num_employees
    
    print(f"\n=== WORKLOAD DISTRIBUTION REPORT ===")
    print(f"Total days: {total_days}")
    print(f"Number of employees: {num_employees}")
    print(f"Expected days per employee: {expected_days_per_employee:.1f}")
    print(f"Days assigned per employee:")
    
    for emp_name, days_assigned in days_per_employee.items():
        difference = days_assigned - expected_days_per_employee
        print(f"  {emp_name}: {days_assigned} days (difference: {difference:+.1f})")
    
    # Check if distribution is fair
    min_days = min(days_per_employee.values())
    max_days = max(days_per_employee.values())
    
    if max_days - min_days <= 1:
        print("✅ WORKLOAD DISTRIBUTION: FAIR (difference ≤ 1 day)")
    else:
        print(f"❌ WORKLOAD DISTRIBUTION: UNFAIR (range: {min_days}-{max_days} days)")
        
    return max_days - min_days <= 1


def solve(payload: Dict) -> List[Dict]:
    """PHASE 3: Optimized single-pass solve function with adaptive constraints"""
    
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
    
    # Check if any employee has no available days (using our availability threshold)
    MAX_ALLOWED_COST = 60
    for i, emp in enumerate(employees):
        has_available_day = False
        for date_info in dates:
            weekday = date_info["weekday"]
            if emp["weekday_cost"][weekday] <= MAX_ALLOWED_COST:
                has_available_day = True
                break
        if not has_available_day:
            raise RuntimeError(f"Employee '{emp['name']}' has no days they can reasonably work based on their preferences. All their available days exceed the preference threshold.")
    
    # Check if each day has at least one available employee
    for d, date_info in enumerate(dates):
        available_employees_for_day = 0
        for emp in employees:
            weekday = date_info["weekday"]
            if emp["weekday_cost"][weekday] <= MAX_ALLOWED_COST:
                available_employees_for_day += 1
        if available_employees_for_day == 0:
            weekday_name = ["Sunday", "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday"][date_info["weekday"]]
            raise RuntimeError(f"No employees are available to work on {weekday_name} ({date_info['date']}) within their reasonable preference range.")
    
    # PHASE 3: Single adaptive model build and solve
    start_time = time.time()
    
    try:
        # Build the adaptive model with soft constraints
        model, x, cost_matrix, workload_penalty_vars, consecutive_penalty_vars, preference_balance_penalty_vars, min_top3_guarantee_penalty_vars, preference_equity_penalty_vars = build_adaptive_model(payload)
        
        # Add the adaptive objective function
        model = create_adaptive_objective(
            model, x, cost_matrix, workload_penalty_vars, consecutive_penalty_vars, 
            preference_balance_penalty_vars, min_top3_guarantee_penalty_vars, preference_equity_penalty_vars, 
            num_emp, num_days
        )
        
        # Configure and solve with optimized parameters
        solver = cp_model.CpSolver()
        configure_solver_optimally(solver, {'variables': num_emp * num_days, 'constraints': num_emp + num_days})
        
        status = solver.Solve(model)
        
        solve_time = time.time() - start_time
        
        if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
            # Build and return the schedule
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
            
            # VALIDATE WORKLOAD DISTRIBUTION
            validate_and_report_workload_distribution(schedule, employees)
            
            return schedule
        
        else:
            # If the adaptive approach fails, provide helpful error message
            error_message = f"Unable to find a valid schedule (solver status: {solver.StatusName(status)}). "
            error_message += "This could be due to:\n"
            error_message += "1. Not enough days to satisfy everyone's preferences\n"
            error_message += "2. Too many consecutive work day restrictions\n"
            error_message += "3. Conflicting employee availability patterns\n"
            error_message += "Try adjusting the date range, excluded dates, or employee preferences."
            raise RuntimeError(error_message)
            
    except Exception as e:
        # Handle any unexpected errors
        if "RuntimeError" in str(type(e)):
            raise e  # Re-raise our custom error messages
        else:
            raise RuntimeError(f"Unexpected error during scheduling: {str(e)}")


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

 