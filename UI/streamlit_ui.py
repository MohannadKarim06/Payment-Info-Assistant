import streamlit as st
import requests
import json
import pandas as pd
from datetime import datetime
import time
import io

# Configuration
API_BASE_URL = "http://localhost:8000"  # Will be updated for Docker
TIMEOUT = 30

# Page configuration
st.set_page_config(
    page_title="Payment Info Assistant",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        text-align: center;
        margin-bottom: 2rem;
        color: #1f77b4;
    }
    .chat-container {
        background-color: #f8f9fa;
        padding: 2rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    .response-container {
        background-color: #e8f4fd;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #1f77b4;
        margin: 1rem 0;
    }
    .error-container {
        background-color: #fee;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #dc3545;
        margin: 1rem 0;
    }
    .success-container {
        background-color: #d4edda;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #28a745;
        margin: 1rem 0;
    }
    .logs-container {
        max-height: 600px;
        overflow-y: auto;
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        font-family: monospace;
        font-size: 0.9rem;
    }
</style>
""", unsafe_allow_html=True)

def check_api_health():
    """Check if the FastAPI backend is running"""
    try:
        response = requests.get(f"{API_BASE_URL}/health", timeout=5)
        return response.status_code == 200
    except:
        return False

def ask_question(query):
    """Send question to FastAPI backend"""
    try:
        payload = {"query": query}
        response = requests.post(
            f"{API_BASE_URL}/ask",
            json=payload,
            timeout=TIMEOUT
        )
        return response.json()
    except requests.exceptions.Timeout:
        return {
            "success": False,
            "error": "Request timed out. Please try again.",
            "response": "The request took too long to process. Please try a simpler question or try again later."
        }
    except requests.exceptions.ConnectionError:
        return {
            "success": False,
            "error": "Cannot connect to the API server.",
            "response": "Unable to connect to the backend service. Please check if the service is running."
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "response": "An unexpected error occurred. Please try again."
        }

def get_logs(limit=100, level=None, search=None):
    """Fetch logs from FastAPI backend"""
    try:
        params = {"limit": limit}
        if level:
            params["level"] = level
        if search:
            params["search"] = search
            
        response = requests.get(f"{API_BASE_URL}/logs", params=params, timeout=10)
        return response.json()
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "logs": [],
            "total_logs": 0
        }

def download_logs():
    """Download logs file from FastAPI backend"""
    try:
        response = requests.get(f"{API_BASE_URL}/logs/download", timeout=30)
        if response.status_code == 200:
            return response.content
        else:
            return None
    except Exception as e:
        st.error(f"Error downloading logs: {str(e)}")
        return None

def get_stats():
    """Get API statistics"""
    try:
        response = requests.get(f"{API_BASE_URL}/stats", timeout=10)
        return response.json()
    except Exception as e:
        return {"error": str(e)}

def main():
    # Header
    st.markdown('<h1 class="main-header">💳 Payment Info Assistant</h1>', unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.header("🔧 Settings")
        
        # API Status
        st.subheader("API Status")
        if check_api_health():
            st.success("✅ API Connected")
        else:
            st.error("❌ API Unavailable")
            st.warning("Please ensure the FastAPI backend is running on port 8000")
        
        # Navigation
        st.subheader("Navigation")
        page = st.radio("Select Page:", ["💬 Chat", "📋 Logs", "📊 Statistics"])
        
        # Quick Actions
        st.subheader("Quick Actions")
        if st.button("🔄 Refresh Status"):
            st.rerun()
    
    # Main content based on selected page
    if page == "💬 Chat":
        show_chat_page()
    elif page == "📋 Logs":
        show_logs_page()
    elif page == "📊 Statistics":
        show_stats_page()

def show_chat_page():
    """Display the chat interface"""
    st.header("💬 Ask Questions About Payment Data")
    
    # Instructions
    with st.expander("ℹ️ How to use", expanded=False):
        st.markdown("""
        **What you can ask:**
        - Questions about payment transactions
        - Payment failure analysis
        - Transaction statistics and trends
        - Payment method performance
        - Regional payment data
        
        **Examples:**
        - "Why did Apple Pay fail for customers in India?"
        - "Show me declined transactions from last week"
        - "What are the most common payment failures?"
        - "How many successful payments were processed yesterday?"
        """)
    
    # Chat interface
    with st.container():
        st.markdown('<div class="chat-container">', unsafe_allow_html=True)
        
        # Query input
        user_query = st.text_area(
            "Enter your question:",
            height=100,
            placeholder="Ask me anything about payment data...",
            key="query_input"
        )
        
        # Submit button
        col1, col2, col3 = st.columns([1, 1, 4])
        
        with col1:
            ask_button = st.button("🚀 Ask Question", type="primary")
        
        with col2:
            clear_button = st.button("🗑️ Clear")
        
        if clear_button:
            st.session_state.query_input = ""
            st.rerun()
        
        # Process query
        if ask_button and user_query.strip():
            with st.spinner("🔍 Processing your question..."):
                response_data = ask_question(user_query.strip())
            
            # Display response
            if response_data.get("success", False):
                st.markdown('<div class="success-container">', unsafe_allow_html=True)
                st.success("✅ Question processed successfully!")
                st.markdown('</div>', unsafe_allow_html=True)
                
                st.markdown('<div class="response-container">', unsafe_allow_html=True)
                st.markdown("**Response:**")
                st.write(response_data["response"])
                st.markdown('</div>', unsafe_allow_html=True)
                
                # Show metadata if available
                if response_data.get("metadata"):
                    with st.expander("📊 Query Details"):
                        st.json(response_data["metadata"])
            
            else:
                st.markdown('<div class="error-container">', unsafe_allow_html=True)
                st.error("❌ Error processing question")
                st.write(response_data.get("response", "Unknown error occurred"))
                
                if response_data.get("error"):
                    with st.expander("🔍 Error Details"):
                        st.code(response_data["error"])
                st.markdown('</div>', unsafe_allow_html=True)
        
        elif ask_button and not user_query.strip():
            st.warning("⚠️ Please enter a question before submitting.")
        
        st.markdown('</div>', unsafe_allow_html=True)

def show_logs_page():
    """Display the logs interface"""
    st.header("📋 System Logs")
    
    # Logs controls
    col1, col2, col3, col4 = st.columns([2, 2, 2, 2])
    
    with col1:
        log_limit = st.selectbox("Number of logs:", [50, 100, 200, 500, 1000], index=1)
    
    with col2:
        log_level = st.selectbox(
            "Filter by level:", 
            ["All", "ERROR", "SUCCESS", "INFO", "PROCESS", "API_REQUEST", "API_SUCCESS", "API_ERROR"],
            index=0
        )
        if log_level == "All":
            log_level = None
    
    with col3:
        search_term = st.text_input("Search logs:", placeholder="Enter search term...")
        if not search_term.strip():
            search_term = None
    
    with col4:
        st.write("") # Spacer
        refresh_logs = st.button("🔄 Refresh Logs")
    
    # Download logs button
    col1, col2 = st.columns([1, 4])
    with col1:
        if st.button("📥 Download Logs"):
            logs_content = download_logs()
            if logs_content:
                st.download_button(
                    label="💾 Download Log File",
                    data=logs_content,
                    file_name=f"payment_assistant_logs_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log",
                    mime="text/plain"
                )
    
    # Fetch and display logs
    if refresh_logs or "logs_data" not in st.session_state:
        with st.spinner("📋 Fetching logs..."):
            st.session_state.logs_data = get_logs(log_limit, log_level, search_term)
    
    logs_response = st.session_state.logs_data
    
    if logs_response.get("success", False):
        logs = logs_response["logs"]
        
        if logs:
            st.success(f"✅ Found {len(logs)} logs")
            
            # Display logs in a scrollable container
            st.markdown('<div class="logs-container">', unsafe_allow_html=True)
            
            for log in logs:
                # Color code by level
                level = log.get("level", "INFO")
                if level in ["ERROR", "API_ERROR"]:
                    color = "🔴"
                elif level in ["SUCCESS", "API_SUCCESS"]:
                    color = "🟢"
                elif level in ["PROCESS"]:
                    color = "🟡"
                else:
                    color = "🔵"
                
                # Format log entry
                timestamp = log.get("timestamp", "Unknown")
                event_type = log.get("event_type", "")
                message = log.get("message", "")
                
                log_display = f"{color} **{timestamp}** | **{level}**"
                if event_type:
                    log_display += f" | **{event_type}**"
                log_display += f"\n   {message}\n---"
                
                st.markdown(log_display)
            
            st.markdown('</div>', unsafe_allow_html=True)
            
            # Convert logs to DataFrame for additional analysis
            if st.checkbox("📊 Show logs as table"):
                df = pd.DataFrame(logs)
                st.dataframe(df, use_container_width=True)
        
        else:
            st.info("ℹ️ No logs found matching the criteria.")
    
    else:
        st.error(f"❌ Error fetching logs: {logs_response.get('error', 'Unknown error')}")

def show_stats_page():
    """Display API statistics"""
    st.header("📊 API Statistics")
    
    # Refresh button
    if st.button("🔄 Refresh Statistics"):
        st.session_state.pop("stats_data", None)
    
    # Fetch stats
    if "stats_data" not in st.session_state:
        with st.spinner("📊 Fetching statistics..."):
            st.session_state.stats_data = get_stats()
    
    stats = st.session_state.stats_data
    
    if "error" not in stats:
        # Display metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Requests", stats.get("total_requests", "N/A"))
        
        with col2:
            st.metric("Successful Queries", stats.get("successful_queries", "N/A"))
        
        with col3:
            st.metric("Errors", stats.get("errors", "N/A"))
        
        with col4:
            success_rate = stats.get("success_rate", "N/A")
            st.metric("Success Rate", success_rate)
        
        # Additional stats
        st.subheader("📈 Detailed Statistics")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.info(f"🏥 Health Checks: {stats.get('health_checks', 'N/A')}")
            st.info(f"⏱️ Uptime: {stats.get('uptime', 'N/A')}")
        
        with col2:
            status = stats.get("status", "unknown")
            if status == "operational":
                st.success(f"✅ Status: {status.title()}")
            else:
                st.warning(f"⚠️ Status: {status.title()}")
    
    else:
        st.error(f"❌ Error fetching statistics: {stats.get('error', 'Unknown error')}")

if __name__ == "__main__":
    main()