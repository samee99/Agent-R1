"""
Search tool implementation for simulating internet searches
"""

import time
import random
from typing import Dict, List, Any, Optional

from agent_r1.tool.tool_base import Tool

from txtai.embeddings import Embeddings

# Load the index from the HF Hub
# embeddings = Embeddings()
# embeddings.load(provider="huggingface-hub", container="neuml/txtai-wikipedia")

# Run a search
# embeddings.search("Roman Empire")

# Run a search matching only the Top 1% of articles
# embeddings.search("""
#    SELECT id, text, score, percentile FROM txtai WHERE similar('Boston') AND
#    percentile >= 0.99
# """)


class SearchTool(Tool):
    """
    Tool for simulating internet searches using the NeuML/txtai-wikipedia model
    """
    
    def __init__(self):
        """
        Initialize the search tool
        
        Args:
            search_db: Custom search database, if None, use default
        """
        name = "search"
        description = "Search for information on the internet using Wikipedia as a knowledge source. This tool utilizes the NeuML/txtai-wikipedia embeddings index which contains the first paragraph (abstract) of each Wikipedia article. It can filter results based on article popularity using the percentile field."
        parameters = {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Search query"
                },
                # "limit": {
                #     "type": "integer",
                #     "description": "Maximum number of results to return (default: 5)"
                # }
            },
            "required": ["query"]
        }
        
        super().__init__(name, description, parameters)
        
        # Initialize search database
        self.embeddings = Embeddings()
        self.embeddings.load(provider="huggingface-hub", container="neuml/txtai-wikipedia")
        # print(f"[DEBUG] EMBEDDINGS LOADED")
    
    def execute(self, args: Dict) -> str:
        """
        Execute search query
        
        Args:
            args: Tool parameters, containing:
                - "query": search query string
                - "limit": optional int to limit number of results
            
        Returns:
            Formatted search results
        """
        query = args.get("query", "").strip()
        limit = args.get("limit", 5)
        

        results = self.embeddings.search(query, limit=limit)
        
        return self._format_results(results)
    
    def _format_results(self, results: List[Dict]) -> str:
        """
        Format search results for better readability
        
        Args:
            results: List of search result dictionaries
            
        Returns:
            Formatted results as a string
        """
        if not results:
            return "No results found."
        
        formatted = "### Search Results\n\n"
        
        for i, result in enumerate(results, 1):
            title = result.get("id", f"Result {i}")
            text = result.get("text", "No content available")
            
            formatted += f"**{i}. {title}**"
            formatted += f"{text}\n\n"
        
        return formatted
    
    def calculate_reward(self, args: Dict, result: str) -> float:
        """
        Calculate reward for search action
        
        Args:
            args: Tool parameters
            result: Tool execution result
            
        Returns:
            Reward value
        """
        # Basic reward calculation based on whether results were found
        if "No results found" in result:
            return 0.1  # Small reward for trying
        
        # Count number of results found
        result_count = result.count("**")
        
        # Base reward for finding results
        reward = 0.5
        
        # Additional reward based on number of results (diminishing returns)
        reward += min(0.5, 0.1 * result_count)
        
        return reward