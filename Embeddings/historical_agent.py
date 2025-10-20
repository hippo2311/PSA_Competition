import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
import pandas as pd
from typing import List, Dict, Tuple, Optional
import json
import re
import requests


class HistoricalRAGSystem:
    """
    Multi-Model RAG System for SOP retrieval using 3 different embedding models.
    Falls back to next model if AI determines chunks aren't useful.
    """

    def __init__(
        self,
        csv_path: str,  # Now historical CSV
        knowledge_csv_path: str,  # NEW: knowledge base CSV
        embedding_models: List[str] = [
            'sentence-transformers/all-MiniLM-L6-v2',
            'sentence-transformers/all-mpnet-base-v2',
            'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2'
        ]
    ):
        """
        Initialize the Multi-Model RAG System.

        Args:
            csv_path: Path to the knowledge base CSV file
            embedding_models: List of 3 embedding model names
        """
        print("Initializing Multi-Model RAG System...")

        # Load data
        print(f"Loading data from {csv_path}...")
        self.df = pd.read_excel(csv_path)
        print(f"Loaded {len(self.df)} records")
        # Load knowledge base data (for SOP lookup)
        print(f"Loading knowledge base from {knowledge_csv_path}...")
        self.knowledge_df = pd.read_csv(knowledge_csv_path)
        print(f"Loaded {len(self.knowledge_df)} knowledge base records")

        # PSA API configuration
        self.psa_api_key = "89f81513c88549ad9decb34d56d88850"
        self.psa_base_url = "https://psacodesprint2025.azure-api.net/gpt-5-mini/openai"
        self.psa_deployment = "gpt-5-mini"
        self.psa_api_version = "2025-04-01-preview"
        self.psa_timeout = 120

        # Initialize embedding models
        print("Loading embedding models...")
        self.embedding_models = []
        for model_name in embedding_models:
            print(f"  Loading {model_name}...")
            model = SentenceTransformer(model_name)
            self.embedding_models.append(model)

        # Prepare text chunks (combine title and overview for better context)
        print("Preparing text chunks...")
        self.chunks = []
        self.chunk_metadata = []

        for idx, row in self.df.iterrows():
            # Use Problem Statements column for embedding
            chunk_text = str(row['Problem Statements']) if pd.notna(row['Problem Statements']) else ""
            if chunk_text:  # Skip empty problem statements
                self.chunks.append(chunk_text)
                self.chunk_metadata.append({
                    'index': idx,
                    'module': row['Module'] if pd.notna(row['Module']) else "",
                    'mode': row['Mode'] if pd.notna(row['Mode']) else "",
                    'problem_statement': chunk_text,
                    'solution': str(row['Solution']) if pd.notna(row['Solution']) else "",
                    'sop': str(row['SOP']) if pd.notna(row['SOP']) else "",
                    'timestamp': str(row['TIMESTAMP']) if pd.notna(row['TIMESTAMP']) else ""
                })

        # Create FAISS indices for each embedding model
        print("Creating FAISS indices...")
        self.faiss_indices = []

        for i, model in enumerate(self.embedding_models):
            print(f"  Creating index for model {i+1}...")
            # Embed all chunks
            embeddings = model.encode(self.chunks, show_progress_bar=True)

            # Create FAISS index
            dimension = embeddings.shape[1]
            index = faiss.IndexFlatL2(dimension)
            index.add(embeddings.astype('float32'))

            self.faiss_indices.append(index)
            print(f"  Model {i+1} index created with {index.ntotal} vectors (dim={dimension})")

        print("Initialization complete!\n")
    def _post_to_psa(self, messages: str) -> dict:
        """
        Send a chat-completions request to the PSA endpoint.
        """
        endpoint = f"{self.psa_base_url}/deployments/{self.psa_deployment}/chat/completions?api-version={self.psa_api_version}"

        headers = {
          "Content-Type": "application/json",
          "api-key": self.psa_api_key
        }

        body = {
          "messages": [
              {"role": "user", "content": messages}
          ],
          "model": self.psa_deployment
        }

        try:
            resp = requests.post(endpoint, headers=headers, json=body, timeout=self.psa_timeout)
        except Exception as e:
            return {"error": f"Network error: {e}", "endpoint": endpoint}

        if resp.status_code != 200:
            preview = resp.text[:500].replace("\n", " ")
            return {"error": f"HTTP {resp.status_code}: {preview}", "endpoint": endpoint}

        try:
            return resp.json()
        except Exception as e:
            raw_preview = resp.text[:500].replace("\n", " ")
            return {"error": f"Invalid JSON response: {e}", "raw": raw_preview}
    def retrieve_top_k(
        self,
        query: str,
        model_index: int,
        k: int = 5
    ) -> List[Dict]:
        """
        Retrieve top K chunks from FAISS using specified embedding model.

        Args:
            query: Query string to search
            model_index: Which embedding model to use (0, 1, or 2)
            k: Number of chunks to retrieve

        Returns:
            List of dictionaries containing chunk metadata and scores
        """
        # Get the embedding model and FAISS index
        model = self.embedding_models[model_index]
        index = self.faiss_indices[model_index]

        # Embed the query
        query_embedding = model.encode([query])[0]

        # Search FAISS
        distances, indices = index.search(
            query_embedding.reshape(1, -1).astype('float32'),
            k
        )

        # Prepare results
        results = []
        for i, idx in enumerate(indices[0]):
            metadata = self.chunk_metadata[idx].copy()
            metadata['score'] = float(distances[0][i])
            metadata['rank'] = i + 1
            results.append(metadata)

        return results
    def search_knowledge_base_sop(self, sop_title: str) -> Optional[Dict]:
        """
        Search for matching SOP in knowledge base by title.

        Args:
            sop_title: SOP title from historical case

        Returns:
            Dictionary with knowledge base SOP data, or None if not found
        """
        print(f"Searching knowledge base for SOP: '{sop_title}'")

        # Try exact match first
        matches = self.knowledge_df[self.knowledge_df['title'].str.strip() == sop_title.strip()]

        if not matches.empty:
            row = matches.iloc[0]
            print(f"✓ Found exact match in knowledge base")
            return {
                'module': row['module'],
                'title': row['title'],
                'overview': row['overview'],
                'rest_chunk': row['rest_chunk']
            }

        # Try partial match
        matches = self.knowledge_df[self.knowledge_df['title'].str.contains(sop_title.strip(), case=False, na=False)]

        if not matches.empty:
            row = matches.iloc[0]
            print(f"✓ Found partial match: '{row['title']}'")
            return {
                'module': row['module'],
                'title': row['title'],
                'overview': row['overview'],
                'rest_chunk': row['rest_chunk']
            }

        print(f"✗ No matching SOP found in knowledge base")
        return None
    def ai_analyze_chunks(
        self,
        original_query: str,
        chunks: List[Dict],
        model_number: int
    ) -> Dict:
        """
        Let AI decide if retrieved chunks are useful for answering the query.

        Args:
            original_query: Original user query
            chunks: List of retrieved chunks with metadata
            model_number: Which embedding model was used (1, 2, or 3)

        Returns:
            Dictionary with:
                - useful: bool - Whether chunks are useful
                - selected_chunk: Dict - The most relevant chunk if useful
                - reasoning: str - AI's reasoning
        """
        # Prepare chunks for AI analysis
        chunks_text = ""
        for i, chunk in enumerate(chunks):
            chunks_text += f"\n--- Chunk {i+1} (Score: {chunk['score']:.4f}) ---\n"  # ✅ Add header
            chunks_text += f"Module: {chunk['module']}\n"
            chunks_text += f"Problem: {chunk['problem_statement'][:200]}...\n"
            chunks_text += f"Solution: {chunk['solution'][:200]}...\n"
            if chunk['sop']:
                chunks_text += f"SOP Available: {chunk['sop']}\n"

        # Create prompt for AI
        prompt = f"""You are an expert SOP (Standard Operating Procedure) analyst. 
    You need to determine if any of the retrieved chunks can help answer the user's query.

    USER QUERY: {original_query}

    RETRIEVED CHUNKS (from embedding model {model_number}):
    {chunks_text}

    TASK:
    1. Carefully analyze each chunk to see if it contains information relevant to solving the user's problem
    2. Determine if ANY of these chunks can help answer the query
    3. If YES, select the MOST relevant chunk (by chunk number)
    4. If NO, explain why none of the chunks are relevant

    Respond in JSON format:
    {{
    "useful": true/false,
    "selected_chunk_number": <chunk number if useful, else null>,
    "reasoning": "<your detailed reasoning>"
    }}
    """

        try:
            # Use PSA API
            response = self._post_to_psa(prompt)

            # Check for errors
            if "error" in response:
                return {
                  "useful": False,
                  "selected_chunk_number": None,
                  "selected_chunk": None,
                  "reasoning": f"API error: {response['error']}"
                }

            # Extract result text from PSA response
            result_text = response['choices'][0]['message']['content'].strip()

            # Extract JSON from response
            json_match = re.search(r'\{.*\}', result_text, re.DOTALL)
            if json_match:
                result = json.loads(json_match.group())
            else:
                result = json.loads(result_text)

            # If useful, attach the full selected chunk data
            if result['useful'] and result['selected_chunk_number']:
                chunk_idx = result['selected_chunk_number'] - 1
                result['selected_chunk'] = chunks[chunk_idx]
            else:
                result['selected_chunk'] = None

            return result

        except Exception as e:
            print(f"Error in AI analysis: {e}")
            return {
                "useful": False,
                "selected_chunk_number": None,
                "selected_chunk": None,
                "reasoning": f"Error during analysis: {str(e)}"
            }

    def extract_verification_instructions(
        self,
        original_query: str,
        selected_chunk: Dict
        ) -> Dict:
        """
        Extract verification queries and admin guidance from the selected SOP chunk.

        Args:
            original_query: Original user query
            selected_chunk: The selected chunk containing SOP information

        Returns:
            Dictionary with:
                - problem_analysis: str - What issue is the SOP solving
                - verification_queries: List[Dict] - SELECT queries to verify the problem
                - admin_action_steps: List[str] - Steps for admin to manually execute
                - warnings: List[str] - Precautions and risks
                - expected_outcome: str - What should happen after fix
                - summary: str - Brief summary for admin
                - full_sop: str - Complete SOP content
                - sop_metadata: Dict - Metadata about the SOP
        """
        # Get the full SOP content
        full_sop = selected_chunk['rest_chunk']

        # Create prompt for extraction
        prompt = f"""You are an expert at analyzing SOPs and providing verification guidance for administrators.

        **IMPORTANT**: You are NOT executing any changes. Your role is to:
        1. Analyze the SOP and understand what it's trying to fix
        2. Create VERIFICATION queries (SELECT only) to check the current state
        3. Provide step-by-step guidance for the admin on what to do

        USER QUERY: {original_query}

        SELECTED SOP:
        Title: {selected_chunk['title']}
        Module: {selected_chunk['module']}
        Overview: {selected_chunk['overview']}

        FULL SOP CONTENT:
        {full_sop}

        TASK:
        1. **Understand the Problem**: What issue is this SOP trying to resolve?
        2. **Extract VERIFICATION Queries ONLY**: Extract only SELECT statements to check the current state. DO NOT include DELETE/UPDATE/INSERT queries.
        3. **Analyze the Resolution**: What does the SOP suggest doing to fix the issue?
        4. **Provide Admin Guidance**: What should the admin manually do after verification?

        **VERIFICATION QUERIES RULES**:
        - Only extract SELECT statements
        - These queries should help verify if the problem described in the query actually exists
        - Queries should check the current state of the system/database
        - Replace placeholders (like :CONTAINER_NO, :VESSEL_ID) with actual values from the user query if available

        **RESOLUTION GUIDANCE RULES**:
        - Describe what the admin should do AFTER verifying the problem
        - If the SOP contains DELETE/UPDATE/INSERT, describe it as guidance (e.g., "Admin should delete duplicate records...")
        - Explain the expected outcome
        - Warn about any risks or precautions

        Respond in JSON format:
        {{
          "problem_analysis": "Duplicate containers exist...",
          "verification_queries": [
              {{
                  "purpose": "Check for duplicates",
                  "query": "SELECT * FROM container WHERE cntr_no = 'CMAU0000020'...",
                  "expected_result": "Multiple rows = duplicates exist"
              }}
          ],
          "admin_action_steps": [
              "1. Review verification results",
              "2. Manually delete older records using: DELETE c FROM container..."
          ],
          "warnings": ["Ensure you keep the latest record"],
          "expected_outcome": "Only one container record should remain",
          "summary": "Verify duplicates exist, then manually remove older records"
        }}
        """

        try:
            # Use PSA API
            response = self._post_to_psa(prompt)

            # Check for errors
            if "error" in response:
                return {
                    "problem_analysis": "API Error occurred",
                    "verification_queries": [],
                    "admin_action_steps": [],
                    "warnings": [f"API error: {response['error']}"],
                    "expected_outcome": "Unable to extract instructions",
                    "summary": f"API error: {response['error']}",
                    "full_sop": full_sop,
                    "sop_metadata": {
                        'module': selected_chunk['module'],
                        'title': selected_chunk['title'],
                        'overview': selected_chunk['overview']
                    },
                    "original_query": original_query
                }

            # Extract result text from PSA response
            result_text = response['choices'][0]['message']['content'].strip()

            # Extract JSON from response
            json_match = re.search(r'\{.*\}', result_text, re.DOTALL)
            if json_match:
                result = json.loads(json_match.group())
            else:
                result = json.loads(result_text)

            # Add the full SOP for reference
            result['full_sop'] = full_sop
            result['sop_metadata'] = {
                'module': selected_chunk['module'],
                'title': selected_chunk['title'],
                'overview': selected_chunk['overview']
            }

            # Add original query for context
            result['original_query'] = original_query

            return result

        except Exception as e:
            print(f"Error extracting instructions: {e}")
            return {
                "problem_analysis": "Error during analysis",
                "verification_queries": [],
                "admin_action_steps": [],
                "warnings": [f"Error: {str(e)}"],
                "expected_outcome": "Unable to extract instructions",
                "summary": f"Error extracting instructions: {str(e)}",
                "full_sop": full_sop,
                "sop_metadata": {
                    'module': selected_chunk['module'],
                    'title': selected_chunk['title'],
                    'overview': selected_chunk['overview']
                },
                "original_query": original_query
            }
    def classify_and_extract(self, incident):  # ✅ Added self
        """
        Clean and extract main incident body using PSA API.
        """
        prompt = f"""
        You are an expert incident analyst. Given the following incident report (which may be an email, SMS, or call transcript)
        **Extract and return the main incident body** (strip all greetings, sign-offs, and unrelated text, but keep all IDs and technical details word for word).

        **Instructions for pruning the paragraph:**
        - If the report is an **email**:
        - Strip the subject line, all greetings (e.g., "Dear Team,"), sign-offs (e.g., "Best Regards," "Thanks," "Regards," "Yours sincerely," etc.).
        - Ignore any footer or disclaimer.
        - Return the rest of the paragraph word for word.

        - If the report is an **SMS**:
        - Ignore alert numbers, timestamps, sender/recipient metadata, and any standard notification phrases.
        - Return whatever is after the issue: body word for word.

        - If the report is a **call transcript** or call alert:
        - Ignore call reference numbers, timestamps, and any introductory or closing statements.
        - Return whatever is after the issue: and details: only the rest of the call word for word.
        - **General:**
        - Do not summarize, paraphrase, or reword. Output the relevant paragraph(s) exactly as they appear in the original report.
        - Do not include any greeting and closing.
        - Watch out for sentences that possibly contain IDs (usually numbers or nonsensical looking strings/numbers like CMAU______) of specific keys and also keep them.

        return just the cleaned text.

        Incident Report:
        {incident}
        """
        response = self._post_to_psa(prompt)  # ✅ Added self

        if isinstance(response, dict) and "choices" in response:
            content = response['choices'][0]['message']['content'].strip()  # ✅ Added .strip()
            return content
        else:
            # Fallback to original query if API fails
            print(f"Warning: PSA API failed, using original query. Error: {response.get('error', 'Unknown')}")
            return incident  # ✅ Return original instead of empty string


    def process_query(self, query: str, k: int = 5) -> Dict:
        """
        MAIN FUNCTION: Process a query through the multi-model RAG pipeline.

        Args:
          query: Raw query string
          k: Number of chunks to retrieve per model

        Returns:
          Dictionary with complete results including instructions
        """
        print(f"\n{'='*80}")
        print(f"PROCESSING QUERY: {query[:100]}...")
        print(f"{'='*80}\n")

        # Step 1: Clean the query
        print("Step 1: Cleaning query...")
        cleaned_query = self.classify_and_extract(query)
        print(f"Cleaned query: {cleaned_query}\n")

        # Step 2: Try each embedding model
        for model_idx in range(3):
            model_num = model_idx + 1
            print(f"{'='*80}")
            print(f"TRYING EMBEDDING MODEL {model_num}")
            print(f"{'='*80}\n")

            # Retrieve top k chunks
            print(f"Retrieving top {k} chunks...")
            chunks = self.retrieve_top_k(cleaned_query, model_idx, k)

            print(f"Retrieved {len(chunks)} chunks:")
            for chunk in chunks:
                print(f"  - {chunk['problem_statement'][:80]}... (score: {chunk['score']:.4f})")
            print()

            # AI analyzes chunks
            print(f"AI analyzing chunks...")
            analysis = self.ai_analyze_chunks(cleaned_query, chunks, model_num)

            print(f"Analysis result:")
            print(f"  Useful: {analysis['useful']}")
            print(f"  Reasoning: {analysis['reasoning'][:200]}...")
            print()

            if analysis['useful']:
                # Found useful historical case!
                print(f"✓ Useful historical case found with model {model_num}!")
                selected_case = analysis['selected_chunk']
                print(f"  Selected: {selected_case['problem_statement'][:100]}...\n")

                # Check if historical case has an SOP
                sop_title = selected_case['sop'].strip()

                if sop_title and sop_title.lower() not in ['', 'nan', 'none']:
                    print(f"Historical case contains SOP: '{sop_title}'")
                    print(f"Searching knowledge base for matching SOP...\n")

                    # Search knowledge base for matching SOP
                    kb_sop = self.search_knowledge_base_sop(sop_title)

                    if kb_sop:
                        # Found matching SOP in knowledge base!
                        print(f"✓ Matching SOP found in knowledge base!")
                        print(f"  Title: {kb_sop['title']}\n")

                        # Extract verification instructions from KB SOP
                        print("Extracting verification instructions...")
                        instructions = self.extract_verification_instructions(
                            cleaned_query,
                            kb_sop
                        )

                        print(f"Instructions extracted:")
                        print(f"  Problem Analysis: {instructions['problem_analysis'][:100]}...")
                        print(f"  Verification Queries: {len(instructions['verification_queries'])}")
                        print(f"  Admin Action Steps: {len(instructions['admin_action_steps'])}")
                        print(f"  Warnings: {len(instructions['warnings'])}")
                        print(f"  Summary: {instructions['summary'][:100]}...")
                        print()

                        return {
                            'success': True,
                            'source': 'historical_with_kb_sop',
                            'model_used': model_num,
                            'cleaned_query': cleaned_query,
                            'historical_case': selected_case,
                            'analysis': analysis,
                            'instructions': instructions,
                            'message': f'Similar historical case found, matched with KB SOP'
                        }
                    else:
                        # SOP not found in KB
                        print(f"✗ SOP not found in knowledge base\n")
                        return {
                            'success': True,
                            'source': 'historical_only',
                            'model_used': model_num,
                            'cleaned_query': cleaned_query,
                            'historical_case': selected_case,
                            'analysis': analysis,
                            'message': f'Historical case found but SOP not in KB',
                            'warning': f"SOP '{sop_title}' not found in knowledge base"
                        }
                else:
                    # No SOP in historical case
                    print(f"No SOP in historical case\n")
                    return {
                        'success': True,
                        'source': 'historical_only',
                        'model_used': model_num,
                        'cleaned_query': cleaned_query,
                        'historical_case': selected_case,
                        'analysis': analysis,
                        'message': f'Historical case found, no SOP available'
                    }

# Helper functions outside the class

def save_instructions_to_file(result: Dict, output_path: str = "instructions.json"):
    """Save the extracted instructions to a JSON file."""
    with open(output_path, 'w') as f:
        json.dump(result, f, indent=2)
    print(f"Instructions saved to {output_path}")

def create_rag_api_handler(rag_system: HistoricalRAGSystem):
    """Create API endpoint wrapper."""
    def handle_query(query: str) -> Dict:
        result = rag_system.process_query(query)
        if result['success']:
            return {
                'status': 'success',
                'sop_found': True,
                'model_used': result['model_used'],
                'instructions': result['instructions'],
                'sop_metadata': result['instructions']['sop_metadata']
            }
        else:
            return {
                'status': 'success',
                'sop_found': False,
                'message': result['message']
            }
    return handle_query

def batch_process_queries(rag_system: HistoricalRAGSystem, queries: List[str]) -> List[Dict]:
    """Process multiple queries at once."""
    results = []
    for i, query in enumerate(queries):
        print(f"\n\n{'#'*80}")
        print(f"BATCH QUERY {i+1}/{len(queries)}")
        print(f"{'#'*80}")
        result = rag_system.process_query(query)
        results.append(result)
    return results

# Main example runner

def historical_model(query):
    """Example usage with error handling."""

    try:
        # Initialize system
        historical_rag = HistoricalRAGSystem(
            csv_path='Knowledge/Case Log.xlsx',  # Historical data
            knowledge_csv_path='knowledge_base_chunks_combined.csv',  # KB for SOP lookup
            embedding_models=[
                'sentence-transformers/all-MiniLM-L6-v2',
                'sentence-transformers/all-mpnet-base-v2',
                'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2'
            ]
        )


        # Process query
        result = historical_rag.process_query(query, k=5)
        
        # Save and display results
        if result['success']:
            #save_instructions_to_file(result, 'output_instructions.json')

            print("\n" + "="*80)
            print("FINAL RESULTS (HISTORICAL AGENT)")
            print("="*80)
            print(f"\nHistorical Case Found: YES")
            print(f"Model Used: Model {result['model_used']}")
            print(f"Source: {result['source']}")

            # Show historical case info
            case = result['historical_case']
            print(f"\n📋 SIMILAR HISTORICAL CASE:")
            print(f"Module: {case['module']}")
            print(f"Problem: {case['problem_statement'][:300]}...")
            print(f"\nSolution Applied Previously: {case['solution'][:300]}...")
            if case['sop']:
                print(f"SOP Reference: {case['sop']}")

            # Check if KB SOP instructions are available
            if 'instructions' in result:
                print(f"\n{'='*80}")
                print("KNOWLEDGE BASE SOP INSTRUCTIONS")
                print(f"{'='*80}")

                print(f"\n📋 SUMMARY:")
                print(f"{result['instructions']['summary']}")

                print(f"\n🔍 PROBLEM ANALYSIS:")
                print(f"{result['instructions']['problem_analysis']}")

                print(f"\n🔎 VERIFICATION QUERIES ({len(result['instructions']['verification_queries'])}):")
                for i, vq in enumerate(result['instructions']['verification_queries'], 1):
                    print(f"\n  Query {i}:")
                    print(f"  Purpose: {vq['purpose']}")
                    print(f"  Query: {vq['query']}")
                    print(f"  Expected Result: {vq['expected_result']}")

                print(f"\n✅ ADMIN ACTION STEPS:")
                for i, step in enumerate(result['instructions']['admin_action_steps'], 1):
                    print(f"{i}. {step}")

                if result['instructions']['warnings']:
                    print(f"\n⚠️  WARNINGS:")
                    for i, warning in enumerate(result['instructions']['warnings'], 1):
                        print(f"{i}. {warning}")

                print(f"\n🎯 EXPECTED OUTCOME:")
                print(f"{result['instructions']['expected_outcome']}")
            else:
                # No KB SOP found, only historical solution
                print(f"\n⚠️ NOTE: {result['message']}")
                if 'warning' in result:
                    print(f"⚠️ WARNING: {result['warning']}")
                print(f"\n💡 SUGGESTED ACTION:")
                print(f"Review the historical solution above and adapt it to your current situation.")
            return {"Alert": query,
                    "Details":json.dumps(result, indent = 1)}

        else:
            print(f"\n{result['message']}")

    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()