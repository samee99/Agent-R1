"""
Search tool implementation for simulating internet searches
"""

import time
import random
from typing import Dict, List, Any, Optional

from agent_r1.tool.tool_base import Tool

# from txtai.embeddings import Embeddings
import faiss
from FlagEmbedding import FlagAutoModel
import json

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
        description = "Search for information on the internet using Wikipedia as a knowledge source."
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
        # self.embeddings = Embeddings()
        # self.embeddings.load(provider="huggingface-hub", container="neuml/txtai-wikipedia")
        # print(f"[DEBUG] EMBEDDINGS LOADED")
        print(f"[DEBUG] EMBEDDINGS LOADING")
        self.index = faiss.read_index("/home/yanruiran/workspace/Agent-R1/data/corpus/hotpotqa/index.bin")
        self.model = FlagAutoModel.from_finetuned(
            'BAAI/bge-large-en-v1.5',
            query_instruction_for_retrieval="Represent this sentence for searching relevant passages: ",
            devices="cpu",   # if not specified, will use all available gpus or cpu when no gpu available
        )
        self.corpus = []
        with open("/home/yanruiran/workspace/Agent-R1/data/corpus/hotpotqa/hpqa_corpus.jsonl","r") as f:
            for idx, line in enumerate(f):
                data = json.loads(line)
                self.corpus.append(data['title'] + " " + data["text"])
        print("[DEBUG] EMBEDDINGS LOADING END")

    
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
        # query = args.get("query", "").strip()
        # limit = args.get("limit", 5)
        

        # # results = self.embeddings.search(query, limit=limit)
        # dist, ids = self.index.search(query, limit)
        
        # return self._format_results(results)
        pass
    
    def batch_execute(self, args_list: List[Dict]) -> List[str]:
        # [{'query':xxx, 'limit':xxx},{}]
        queries = [x["query"] for x in args_list]
        embeddings = self.model.encode_queries(queries)
        dist, ids = self.index.search(embeddings, 5) # ids: b*5
        results_str = [self._format_results(ids[i]) for i in range(len(ids))]
        return results_str

    def _format_results(self, results: List) -> str:
        """
        Format search results for better readability
        
        Args:
            results: List of search result List
            
        Returns:
            Formatted results as a string
        """
        results_list = []
        
        for i, result in enumerate(results):
            results_list.append(self.corpus[result])
        
        return json.dumps({"results": results_list})
    
    def calculate_reward(self, args: Dict, result: str) -> float:
        """
        Calculate reward for search action
        
        Args:
            args: Tool parameters
            result: Tool execution result
            
        Returns:
            Reward value
        """
        # # Basic reward calculation based on whether results were found
        # if "No results found" in result:
        #     return 0.1  # Small reward for trying
        
        # # Count number of results found
        # result_count = result.count("**")
        
        # # Base reward for finding results
        # reward = 0.5
        
        # # Additional reward based on number of results (diminishing returns)
        # reward += min(0.5, 0.1 * result_count)
        
        # return reward
        return 0.0