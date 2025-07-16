"""Main FastAPI application."""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, HttpUrl
from typing import Optional
import requests
from bs4 import BeautifulSoup
import os
from dotenv import load_dotenv
import re

# Load environment variables
load_dotenv()

# Configuration
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
MAX_CONTENT_LENGTH = 8000
DEFAULT_SUMMARY_LENGTH = 200

# Initialize OpenAI client with proper error handling
openai_client = None
try:
    if OPENAI_API_KEY:
        import openai
        openai_client = openai.OpenAI(api_key=OPENAI_API_KEY)
except ImportError:
    print("OpenAI library not installed. Please install it with: pip install openai")
except Exception as e:
    print(f"Error initializing OpenAI client: {e}")

# Pydantic models
class SummarizeRequest(BaseModel):
    url: HttpUrl
    summary_length: Optional[int] = 200

class SummarizeResponse(BaseModel):
    url: str
    title: str
    summary: str
    word_count: int

class ErrorResponse(BaseModel):
    error: str
    message: str

# Web scraper class
class WebScraper:
    def __init__(self, timeout: int = 10):
        self.timeout = timeout
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })
    
    def extract_content(self, url: str) -> dict:
        """Extract content from a website URL."""
        try:
            response = self.session.get(url, timeout=self.timeout)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Remove unwanted elements
            for element in soup(['script', 'style', 'nav', 'footer', 'header']):
                element.decompose()
            
            # Extract title
            title = soup.find('title')
            title_text = title.get_text().strip() if title else "No title found"
            
            # Extract main content
            main_content = self._extract_main_content(soup)
            
            return {
                'url': url,
                'title': title_text,
                'content': main_content,
                'word_count': len(main_content.split())
            }
            
        except Exception as e:
            raise Exception(f"Failed to scrape {url}: {str(e)}")
    
    def _extract_main_content(self, soup: BeautifulSoup) -> str:
        """Extract main text content from the page."""
        # Try to find main content areas
        main_selectors = [
            'main', 'article', '.content', '#content', 
            '.post-content', '.entry-content'
        ]
        
        main_content = ""
        for selector in main_selectors:
            element = soup.select_one(selector)
            if element:
                main_content = element.get_text(separator=' ', strip=True)
                break
        
        # Fallback to body content
        if not main_content:
            body = soup.find('body')
            if body:
                main_content = body.get_text(separator=' ', strip=True)
        
        # Clean up whitespace
        main_content = re.sub(r'\s+', ' ', main_content)
        return main_content.strip()

# Content summarizer class
class ContentSummarizer:
    def __init__(self):
        if not OPENAI_API_KEY:
            raise ValueError("OpenAI API key not configured")
        
        if not openai_client:
            raise ValueError("OpenAI client not initialized properly")
        
        self.client = openai_client
    
    def summarize(self, content: str, max_length: int = 200) -> str:
        """Summarize content using OpenAI."""
        try:
            # Truncate content if too long
            if len(content) > MAX_CONTENT_LENGTH:
                content = content[:MAX_CONTENT_LENGTH]
            
            prompt = f"""
            Please summarize the following website content in approximately {max_length} words.
            Focus on the main points and key information:
            
            {content}
            """
            
            response = self.client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that creates concise, informative summaries."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=max_length * 2,
                temperature=0.3
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            raise Exception(f"Failed to summarize content: {str(e)}")

# Create FastAPI app instance
app = FastAPI(
    title="Website Summarizer API",
    description="A simple FastAPI application to summarize website content",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# API Routes
@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "message": "Website Summarizer API",
        "docs": "/docs",
        "health": "/api/v1/health"
    }

@app.get("/api/v1/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "openai_configured": bool(OPENAI_API_KEY and openai_client)
    }

@app.post("/api/v1/summarize", response_model=SummarizeResponse)
async def summarize_website(request: SummarizeRequest):
    """Summarize a website's content."""
    try:
        # Check if OpenAI is configured
        if not OPENAI_API_KEY:
            raise HTTPException(
                status_code=500,
                detail="OpenAI API key not configured. Please set OPENAI_API_KEY in your .env file"
            )
        
        if not openai_client:
            raise HTTPException(
                status_code=500,
                detail="OpenAI client not initialized. Please check your OpenAI installation and API key"
            )
        
        # Initialize services
        scraper = WebScraper()
        summarizer = ContentSummarizer()
        
        # Extract website content
        website_data = scraper.extract_content(str(request.url))
        
        # Generate summary
        summary = summarizer.summarize(
            website_data['content'],
            request.summary_length
        )
        
        return SummarizeResponse(
            url=website_data['url'],
            title=website_data['title'],
            summary=summary,
            word_count=website_data['word_count']
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Run the application
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
