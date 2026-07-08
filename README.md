# Needle

Intelligent QA for businesses.

---

## Installation

Requirements:

- Python 3.12+
- Docker and Docker Compose

Create a virtual environment and install the project:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
```

The default install includes the document parsing dependencies from
`requirements-doc.txt`.

For development tools and test dependencies, install the `dev` extra:

```bash
python -m pip install -e ".[dev]"
```

Create the local environment file and update any secrets or model provider
settings you need:

```bash
cp .env.example .env
```

Start the local Elasticsearch, Kibana, and MinIO services:

```bash
docker compose up -d
```

Run the API:

```bash
python app.py
```

Check that the service is running:

```bash
curl http://127.0.0.1:11000/api/health
```

To-Do:

- [x] Add auth
- [x] Store original uploaded documents in an object storage (e.g. S3, or MinIO)
- [x] Ability to delete documents and associated data
- [x] PDF file support
- [x] Add live-file capabilities
- [x] Store object metadata
- [x] Add pipeline unit tests
- [ ] Add CI/CD for testing
- [ ] Async database and pipeline operations
- [ ] Update README
- [x] Add Kibana to the project's docker-compose
- [ ] Spin up a UI using shadcn
- [ ] Elasticsearch and Kibana authentication
- [x] Package configuration via env variables
- [ ] Improved [Persian analyzer](https://www.elastic.co/docs/reference/text-analysis/analysis-lang-analyzer#persian-analyzer) for Elasticsearch
- [ ] Configurable embeddings dimensions
- [ ] Knowledge base for document grouping
