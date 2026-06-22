## SKG MCP Server

Formal Model Context Protocol (MCP) server scaffold for scholarly knowledge graph access.

This refactors the previous kitchen-sink FastAPI approach into a standard Python `mcp` SDK server with:
- typed Pydantic request/response contracts
- formal MCP tools defined directly with `@mcp.tool`
- MCP resources defined directly with `@mcp.resource`
- abstract backend methods plus a concrete SUDO KG backend for Fuseki + Milvus
- Smithery publishing configuration

### Package Layout

- `src/skg_mcp/models.py`: shared request/response models
- `src/skg_mcp/backend.py`: abstract backend contract and `NotImplementedBackend`
- `src/skg_mcp/client.py`: reusable async MCP client wrapper for stdio/HTTP transports
- `src/skg_mcp/mock_backend.py`: LLM-connected mock backend implementation for testing
- `src/skg_mcp/sudo_backend.py`: concrete backend for the SUDO KG over Apache Jena Fuseki and Milvus
- `src/skg_mcp/server.py`: FastMCP server with explicit tool/resource decorators
- `src/server.py`: stdio runtime entrypoint (`uv run src/server.py`)
- `smithery.yaml`: Smithery deployment/start command config

### Implemented MCP Tools

- `filter_papers`
- `lexical_search` (unified concept/statement lexical search; supports `node_types`)
- `semantic_search` (unified concept/statement semantic search; supports `node_types`)
- `semantic_constraint_search` (unified constrained semantic search; supports `node_types`)
- `resolve_concept_reference`
- `expand_context` (unified context expansion for concept/statement nodes)
- `expand_neighbors` (generic neighbor traversal for any node kind with `hop_count`)
- `get_attibution` (node-level attribution metadata)
- `get_provenance` (node-level provenance records)

Each tool is wired to an abstract backend method. Use `SKG_BACKEND=sudo_kg` for the concrete SUDO KG adapter, `SKG_BACKEND=llm_mock` for the synthetic test adapter, or replace `NotImplementedBackend` with another adapter.

The filter models also support `sparql_filters`, a list of small SPARQL `WHERE` fragments with named parameters. Fragments are inserted into the relevant query and may use variables exposed by the tool query, such as `?paper`, `?node`, `?type`, and `?label`.

Example:

```json
{
  "where": "?node sudo:mentions ?concept . FILTER(?concept IN ({{concepts}}))",
  "params": {
    "concepts": [
      "https://purl.org/twc/sudo/kg/concept/05e48dc7-00a2-4d8e-a0d7-588dd553d853"
    ]
  },
  "graph": "sudo"
}
```

### MCP Resources

- `skg://metadata/server`: server metadata
- `skg://catalog/tools`: full tool catalog with input/output schemas
- `skg://catalog/tools/{tool_name}`: schema lookup for one tool

### Local Run

1. Install dependencies:

```bash
uv sync
```

2. Run MCP server over stdio:

```bash
uv run src/server.py
```

Run the MCP server over streamable HTTP:

```bash
export SKG_BACKEND=sudo_kg
export SKG_MCP_TRANSPORT=streamable-http
export SKG_MCP_HOST=127.0.0.1
export SKG_MCP_PORT=8004
export SKG_MCP_PATH=/mcp

uv run src/server.py
```

Then connect clients or notebooks to:

```text
http://127.0.0.1:8000/mcp
```

### Notebook Test

Use the Jupyter notebook:
- `notebook/mcp_client_test.ipynb` (client-driven smoke test; recommended)
- `notebook/mcp_test.ipynb` (raw JSON-RPC smoke test)
- `notebook/sudo_kg_http_server_playground.ipynb` (connect to an already-running SUDO KG MCP HTTP server)
- `notebook/sudo_kg_tool_playground.ipynb` (stdio subprocess playground)

It includes cells for:
- client initialization over stdio or streamable-http
- `tools/list`, resource reads, and typed tool calls
- smoke assertions for search/context/neighbors/attribution/provenance

### Python MCP Client

Use the built-in client wrapper:

```python
import asyncio

from skg_mcp.client import SKGMCPClient
from skg_mcp.models import LexicalSearchArgs, SearchNodeType

async def main() -> None:
    async with SKGMCPClient.connect_stdio(env={"SKG_BACKEND": "llm_mock"}) as client:
        result = await client.lexical_search(
            LexicalSearchArgs(
                query="attention",
                node_types=[SearchNodeType.CONCEPT, SearchNodeType.STATEMENT],
                limit=4,
            )
        )
        print(len(result.concepts), len(result.statements))

asyncio.run(main())
```

### LLM Mock Backend (Testing)

Run with LLM-backed mock responses:

```bash
export SKG_BACKEND=llm_mock
export SKG_LLM_MODEL=openai/gpt-oss-20b
# optional; defaults shown below
export SKG_LLM_PROVIDER=lmstudio
export SKG_LLM_BASE_URL=http://localhost:1234/v1
export SKG_LLM_API_KEY=your_api_key

uv run src/server.py
```

Environment variables:
- `SKG_BACKEND`: set to `llm_mock` (or `mock_llm`) to enable mock backend
- `SKG_LLM_MODEL`: model name used for synthetic response generation (default: `openai/gpt-oss-20b`)
- `SKG_LLM_PROVIDER`: `lmstudio` (default), `openai_endpoint`, or `nim`
- `SKG_LLM_BASE_URL`: OpenAI-compatible base URL (default `https://spark-6d47:1234/v1`)
- `SKG_LLM_API_KEY` (or `OPENAI_API_KEY`): API key for the LLM provider
- `SKG_LLM_TEMPERATURE`: generation temperature (default `0.2`)
- `SKG_LLM_TIMEOUT_SECONDS`: request timeout (default `45`)
- `SKG_LLM_STRICT`: if `true`, fail instead of fallback when LLM calls fail
- `SKG_EMBEDDER_PROVIDER`: `lmstudio` (default), `lms`, or `openai`
- `SKG_EMBEDDER_MODEL`: embedder model name (default `text-embedding-bge-base-en-v1.5`)
- `SKG_EMBEDDER_BASE_URL`: base URL for embedder provider (defaults to `SKG_LLM_BASE_URL`)
- `SKG_EMBEDDER_API_KEY`: API key for embedder provider

### SUDO KG Backend

Run against Apache Jena Fuseki and Milvus:

```bash
export SKG_BACKEND=sudo_kg
export SKG_FUSEKI_URL=http://localhost:3030
export SKG_FUSEKI_DATASET=sudo_kg
export SKG_MILVUS_URI=http://localhost:19530
export SKG_MILVUS_COLLECTION=sudo_kg
export SKG_EMBEDDER_MODEL=jina-v4-text-matching

uv run src/server.py
```

The backend also reads this checkout's `.env` automatically and accepts these aliases:
- `FUSEKI_HOST`, `FUSEKI_PORT`, `FUSEKI_DATASET_NAME`
- `MILVUS_HOST`, `MILVUS_PORT`, `MILVUS_COLLECTION_NAME`, `MILVUS_DB_NAME`
- `MODEL_PROVIDER`, `PROVIDER_URL`, `OPENAI_API_KEY`

For LM Studio-hosted Jina embeddings, use the OpenAI-compatible embedder path:

```bash
export SKG_EMBEDDER_PROVIDER=openai
export SKG_EMBEDDER_BASE_URL=http://localhost:1234/v1
export SKG_EMBEDDER_MODEL=jina-v4-text-matching
```

Environment variables:
- `SKG_BACKEND`: set to `sudo_kg` or `sudo`
- `SKG_FUSEKI_QUERY_URL`: full SPARQL endpoint URL; overrides `SKG_FUSEKI_URL` + `SKG_FUSEKI_DATASET`
- `SKG_FUSEKI_URL`: Fuseki base URL (default `http://localhost:3030`)
- `SKG_FUSEKI_DATASET`: Fuseki dataset name (default `sudo_kg`)
- `SKG_GRAPH_META`, `SKG_GRAPH_STRUCT`, `SKG_GRAPH_SUDO`, `SKG_GRAPH_PROV`, `SKG_GRAPH_CONCEPT`: named graph IRIs/names (defaults `urn:meta`, `urn:struct`, `urn:sudo`, `urn:prov`, `concept`)
- `SKG_MILVUS_URI`, `SKG_MILVUS_TOKEN`, `SKG_MILVUS_DB_NAME`: Milvus connection settings
- `SKG_MILVUS_COLLECTION`: Milvus collection name (default `sudo_kg`)
- `SKG_EMBEDDER_PROVIDER`, `SKG_EMBEDDER_MODEL`, `SKG_EMBEDDER_BASE_URL`, `SKG_EMBEDDER_API_KEY`: embedding settings for dense search

Live tool test runner:

```bash
uv run python scripts/test_sudo_kg_tools.py
```

The runner starts the MCP server over stdio with `SKG_BACKEND=sudo_kg`, translates the `.env` aliases above into `SKG_*` settings, discovers real seed paper/concept/statement IDs, and calls every MCP tool.

Milvus collection fields expected by the backend:
- `id`: KG node local name
- `text`: KG node label
- `text_sparse`: sparse text vector for lexical/BM25 search
- `type`: `artifact`, `proposition`, or `concept`
- `text_dense`: 768-dimensional dense vector
- `paper_id`: local paper ID from `prov:hadPrimarySource`

Notes:
- The mock backend now uses the repository's `src.llm` and `src.embedder` modules.
- The mock backend always returns schema-valid synthetic payloads.
- If strict mode is disabled and LLM calls fail, deterministic fallback payloads are returned.

### Wiring a Real Backend

Implement `ScholarlyKnowledgeGraphBackend` in `src/skg_mcp/backend.py` and pass your implementation into `create_mcp_server(backend=...)` or `run_stdio(backend=...)`.

### Smithery

This repository includes a Smithery-compatible `smithery.yaml` using `startCommand` + stdio.

Typical publish/install flow:

```bash
# publish from Smithery web UI after connecting your GitHub repo
# then install from client side
npx -y @smithery/cli install @<your-namespace>/skg-mcp --client claude
```

If your MCP root is a subdirectory, place `smithery.yaml` in that subdirectory and set the same base directory in Smithery.
