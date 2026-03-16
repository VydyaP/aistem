

from pydantic import BaseModel
from typing import Optional, List

class QuestionRequest(BaseModel):
    user_id: Optional[str] = None
    question: str

class MathProblemRequest(BaseModel):
    user_id: Optional[str] = None
    problem: str

class RetrievalRequest(BaseModel):
    user_id: Optional[str] = None
    query: str
    top_k: Optional[int] = 3

class StepResponse(BaseModel):
    steps: List[str]
    answer: Optional[str] = None
    sources: Optional[List[str]] = None
    success: bool = True
    message: Optional[str] = None
