import os
import sys
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
import json
import uvicorn

# Add the parent directory to the path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from pipelines.query_pipeline import query_pipeline
from utils.logger import log_event, get_logs

# Create FastAPI app
app = FastAPI(
    title="Payment Info Assistant API",
    description="AI-powered chatbot for querying payment data and documents",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure this properly for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic models for request/response
class QueryRequest(BaseModel):
    query: str
    
class QueryResponse(BaseModel):
    success: bool
    response: str
    error: Optional[str] = None
    metadata: Optional[dict] = None

class LogsResponse(BaseModel):
    success: bool
    logs: list
    total_logs: int
    error: Optional[str] = None

@app.get("/")
async def root():
    """
    Root endpoint to check if the API is running
    """
    return {
        "message": "Payment Info Assistant API is running",
        "version": "1.0.0",
        "status": "healthy"
    }

@app.post("/ask", response_model=QueryResponse)
async def ask_question(request: QueryRequest):
    """
    Main endpoint to ask questions about payment data
    """
    try:
        # Log the incoming request
        log_event("API_REQUEST", f"Received query: {request.query}")
        
        # Validate input
        if not request.query or not request.query.strip():
            log_event("API_ERROR", "Empty query received")
            raise HTTPException(status_code=400, detail="Query cannot be empty")
        
        # Process the query through the pipeline
        response = query_pipeline(request.query.strip())
        
        # Log successful response
        log_event("API_SUCCESS", f"Successfully processed query: {request.query[:50]}...")
        
        return QueryResponse(
            success=True,
            response=response,
            metadata={
                "query": request.query,
                "timestamp": log_event("RESPONSE", "Query processed successfully", return_timestamp=True)
            }
        )
        
    except HTTPException:
        # Re-raise HTTP exceptions
        raise
        
    except Exception as e:
        # Log the error
        error_msg = f"Error processing query: {str(e)}"
        log_event("API_ERROR", error_msg)
        
        # Return error response
        return QueryResponse(
            success=False,
            response="I apologize, but I encountered an error while processing your question. Please try again or contact support if the issue persists.",
            error=str(e),
            metadata={
                "query": request.query,
                "error_type": type(e).__name__
            }
        )

@app.get("/logs", response_model=LogsResponse)
async def get_logs_endpoint(
    limit: int = 100,
    level: Optional[str] = None,
    search: Optional[str] = None
):
    """
    Endpoint to retrieve application logs
    
    Args:
        limit: Maximum number of logs to return (default: 100)
        level: Filter by log level (ERROR, SUCCESS, PROCESS, INFO, etc.)
        search: Search term to filter logs
    """
    try:
        log_event("API_REQUEST", f"Logs requested - limit: {limit}, level: {level}, search: {search}")
        
        # Get logs from the logger
        logs = get_logs(limit=limit, level=level, search=search)
        
        log_event("API_SUCCESS", f"Retrieved {len(logs)} logs")
        
        return LogsResponse(
            success=True,
            logs=logs,
            total_logs=len(logs)
        )
        
    except Exception as e:
        error_msg = f"Error retrieving logs: {str(e)}"
        log_event("API_ERROR", error_msg)
        
        return LogsResponse(
            success=False,
            logs=[],
            total_logs=0,
            error=str(e)
        )

@app.get("/health")
async def health_check():
    """
    Health check endpoint
    """
    try:
        # Basic health checks
        health_status = {
            "status": "healthy",
            "timestamp": log_event("HEALTH_CHECK", "Health check performed", return_timestamp=True),
            "checks": {
                "api": "ok",
                "logging": "ok",
                "pipeline": "ok"
            }
        }
        
        # Test if key components are accessible
        try:
            # Test if we can import key modules
            from utils.synonym_resolver import get_synonym_resolver
            from utils.structured_query import StructuredDataSearcher
            from utils.unstructured_rag import UnstructuredRAGSearcher
            health_status["checks"]["modules"] = "ok"
        except Exception as e:
            health_status["checks"]["modules"] = f"error: {str(e)}"
            health_status["status"] = "degraded"
        
        return health_status
        
    except Exception as e:
        log_event("API_ERROR", f"Health check failed: {str(e)}")
        return {
            "status": "unhealthy",
            "error": str(e)
        }

@app.get("/logs/download")
async def download_logs():
    """
    Endpoint to download the full log file
    """
    try:
        from fastapi.responses import FileResponse
        
        log_file_path = "logs/logs.log"
        
        if not os.path.exists(log_file_path):
            log_event("API_ERROR", "Log file not found")
            raise HTTPException(status_code=404, detail="Log file not found")
        
        log_event("API_REQUEST", "Log file download requested")
        
        return FileResponse(
            path=log_file_path,
            filename="payment_assistant_logs.log",
            media_type="text/plain"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        log_event("API_ERROR", f"Error downloading logs: {str(e)}")
        raise HTTPException(status_code=500, detail="Error downloading log file")

@app.get("/stats")
async def get_stats():
    """
    Get basic statistics about the API usage
    """
    try:
        # Get recent logs to calculate stats
        logs = get_logs(limit=1000)
        
        # Count different types of events
        stats = {
            "total_requests": len([log for log in logs if log.get("level") == "API_REQUEST"]),
            "successful_queries": len([log for log in logs if log.get("level") == "API_SUCCESS"]),
            "errors": len([log for log in logs if log.get("level") == "API_ERROR"]),
            "health_checks": len([log for log in logs if log.get("level") == "HEALTH_CHECK"]),
            "uptime": "N/A",  # Could be calculated if we track startup time
            "status": "operational"
        }
        
        # Calculate success rate
        total_queries = stats["total_requests"]
        if total_queries > 0:
            stats["success_rate"] = f"{(stats['successful_queries'] / total_queries) * 100:.2f}%"
        else:
            stats["success_rate"] = "N/A"
        
        log_event("API_REQUEST", "Stats requested")
        
        return stats
        
    except Exception as e:
        log_event("API_ERROR", f"Error getting stats: {str(e)}")
        return {
            "error": str(e),
            "status": "error"
        }

if __name__ == "__main__":
    # Configure logging
    log_event("STARTUP", "Starting Payment Info Assistant API...")
    
    try:
        # Run the FastAPI server
        uvicorn.run(
            "main:app",
            host="0.0.0.0",
            port=8000,
            reload=True,  # Set to False in production
            log_level="info"
        )
    except Exception as e:
        log_event("STARTUP_ERROR", f"Failed to start server: {str(e)}")
        raise