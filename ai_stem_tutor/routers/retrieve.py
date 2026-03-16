from fastapi import APIRouter
from ai_stem_tutor.models.schemas import RetrievalRequest, StepResponse
from ai_stem_tutor.services.rag_retriever import RAGRetriever

router = APIRouter(prefix="/retrieve", tags=["Retrieve"])
retriever = RAGRetriever()

@router.post("/", response_model=StepResponse)
def retrieve_context(request: RetrievalRequest):
    sources = retriever.retrieve(request.query, top_k=request.top_k)
    print("Retrieved sources:", sources)
    return StepResponse(
        steps=[
            "Step 1: Embedding the query",
            "Step 2: Searching the vector database",
            "Step 3: Returning top-k results"
        ],
        answer="Top textbook chunks retrieved.",
        sources=sources,
        success=True,
        message="Retrieval request received. Fetching context..."
    )