from fastapi import APIRouter
from ai_stem_tutor.models.schemas import QuestionRequest, StepResponse
from ai_stem_tutor.services.rag_retriever import RAGRetriever, call_granite_llm

router = APIRouter(prefix="/ask", tags=["Ask"])
retriever = RAGRetriever()

# Simple STEM keyword list (expand as needed)
STEM_KEYWORDS = [
    # Math
    'math', 'algebra', 'geometry', 'calculus', 'equation', 'integral', 'derivative', 'function', 'variable',
    'matrix', 'vector', 'probability', 'statistics', 'trigonometry', 'logarithm', 'polynomial', 'graph', 'limit',
    'differential', 'series', 'mean', 'median', 'mode', 'standard deviation', 'quadratic', 'linear', 'exponent',
    # Science
    'science', 'physics', 'chemistry', 'biology', 'photosynthesis', 'atom', 'molecule', 'cell', 'organism',
    'force', 'energy', 'velocity', 'acceleration', 'gravity', 'electricity', 'magnetism', 'reaction', 'experiment',
    'data', 'hypothesis', 'theory', 'law', 'element', 'compound', 'solution', 'acid', 'base', 'salt', 'enzyme',
    'ecosystem', 'evolution', 'genetics', 'DNA', 'RNA', 'protein', 'mitosis', 'meiosis', 'respiration', 'circulation',
    # Technology
    'technology', 'computer', 'programming', 'robot', 'circuit', 'algorithm', 'software', 'hardware', 'AI', 'machine learning',
    'data science', 'coding', 'python', 'java', 'c++', 'html', 'css', 'javascript', 'database', 'network', 'encryption',
    # Engineering
    'engineering', 'mechanical', 'electrical', 'civil', 'chemical', 'aerospace', 'design', 'structure', 'bridge', 'machine',
    'thermodynamics', 'fluid', 'stress', 'strain', 'material', 'manufacturing', 'robotics', 'automation', 'CAD', 'CAM',
    # General STEM
    'STEM', 'scientific', 'experiment', 'research', 'innovation', 'invention', 'discovery', 'analysis', 'model', 'simulation'
]

def is_stem_question(question: str) -> bool:
    q = question.lower()
    return any(word in q for word in STEM_KEYWORDS)

@router.post("/", response_model=StepResponse)
def ask_question(request: QuestionRequest):
    question = request.question.strip()
    if is_stem_question(question):
        # STEM: Use RAG for context
        context_chunks = retriever.retrieve(question, top_k=3)
        context = "\n".join(context_chunks)
        prompt = (
            "You are a STEM tutor. Using the following textbook excerpts and the student's question, "
            "write a clear, step-by-step answer as you would explain to a high school student.\n\n"
            f"Textbook Excerpts:\n{context}\n\nQuestion: {question}\n\nStep-by-step Answer:"
        )
        answer = call_granite_llm(prompt)
        return StepResponse(
            steps=["RAG context retrieved", "Granite LLM called for STEM answer"],
            answer=answer,
            sources=context_chunks,
            success=True,
            message="STEM answer generated using RAG + Granite LLM"
        )
    else:
        # General: Just use Granite LLM
        prompt = (
            "You are a friendly tutor. Answer the following question in a simple, student-friendly way.\n\n"
            f"Question: {question}\n\nAnswer:"
        )
        answer = call_granite_llm(prompt)
        return StepResponse(
            steps=["Granite LLM called for general answer"],
            answer=answer,
            sources=[],
            success=True,
            message="General answer generated using Granite LLM"
        )
