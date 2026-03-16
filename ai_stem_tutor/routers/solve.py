from fastapi import APIRouter
from ai_stem_tutor.models.schemas import MathProblemRequest, StepResponse
from ai_stem_tutor.services.math_solver import MathSolver

router = APIRouter(prefix="/solve", tags=["Solve"])

@router.post("/", response_model=StepResponse)
def solve_problem(request: MathProblemRequest):
    steps, answer = MathSolver.solve_problem(request.problem)
    return StepResponse(
        steps=steps,
        answer=answer,
        sources=[],
        success=True if answer and 'Could not solve' not in answer else False,
        message="Solved using SymPy" if answer and 'Could not solve' not in answer else answer
    )
