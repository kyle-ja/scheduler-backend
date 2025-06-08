from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict
import sys
import os

# Add the app directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.solver.scheduler_lp import solve

app = FastAPI()

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with your Vercel domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class Employee(BaseModel):
    name: str
    weekday_cost: List[int]

class Date(BaseModel):
    date: str
    weekday: int

class ScheduleRequest(BaseModel):
    employees: List[Employee]
    dates: List[Date]
    max_consecutive_days: int = 2

@app.post("/generate-schedule")
async def generate_schedule(request: ScheduleRequest):
    try:
        # Convert Pydantic model to dict for the solver
        payload = {
            "employees": [emp.dict() for emp in request.employees],
            "dates": [date.dict() for date in request.dates],
            "max_consecutive_days": request.max_consecutive_days
        }
        
        # Call the solver
        result = solve(payload)
        
        if result is None:
            raise HTTPException(status_code=400, detail="No valid schedule could be generated")
            
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    return {"status": "healthy"}