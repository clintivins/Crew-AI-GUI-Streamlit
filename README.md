# CrewAI Studio

Welcome to CrewAI Studio! This application provides a user-friendly interface written in Streamlit for interacting with CrewAI, suitable even for those who don't want to write any code. Follow the steps below to install and run the application using Docker/docker-compose or Conda/venv.

## Features

- **Multi-platform support**: Works on Windows, Linux and MacOS.
- **No coding required**: User-friendly interface for interacting with CrewAI.
- **Conda and virtual environment support**: Choose between Conda and a Python virtual environment for installation.
- **Results history**: You can view previous results.
- **Knowledge sources**: You can add knowledge sources for your crews
- **CrewAI tools** You can use crewai tools to interact with real world. ~~Crewai studio uses a forked version of crewai-tools with some bugfixes and enhancements (https://github.com/strnad/crewAI-tools)~~ (bugfixes already merged to crewai-tools)
- **Custom Tools** Custom tools for calling APIs, writing files, enhanced code interpreter, enhanced web scraper... More will be added soon
- **LLM providers supported**: Currently OpenAI, Groq, Anthropic, ollama, Grok and LM Studio backends are supported. OpenAI key is probably still needed for embeddings in many tools. Don't forget to load an embedding model when using LM Studio.
- **Single Page app export**: Feature to export crew as simple single page streamlit app.
- **Threaded crew run**: Crews can run in background and can be stopped.
- **Current Date/Time Context Tool**: New `CurrentDateTime` tool supplies the live UTC and local timestamp so agents can ground searches in "now".
- **Graceful Optional Dependencies**: Enhanced CSV search tool (`CSVSearchToolEnhanced`) degrades cleanly if optional `embedchain` isn't installed.

## Support CrewAI Studio

Your support helps fund the development and growth of our project. Every contribution is greatly appreciated!

### Donate with Bitcoin
bc1qgsn45g02wran4lph5gsyqtk0k7t98zsg6qur0y

### Sponsor via GitHub
[![Sponsor on GitHub](https://img.shields.io/badge/Sponsor-GitHub-ff69b4?style=for-the-badge&logo=github)](https://github.com/sponsors/strnad)


## Screenshots

<img src="https://raw.githubusercontent.com/strnad/CrewAI-Studio/main/img/ss1.png" alt="crews definition" style="width:50%;"/><img src="https://raw.githubusercontent.com/strnad/CrewAI-Studio/main/img/ss2.png" alt="kickoff" style="width:50%;"/>
<img src="https://raw.githubusercontent.com/strnad/CrewAI-Studio/main/img/ss3.png" alt="kickoff" style="width:50%;"/><img src="https://raw.githubusercontent.com/strnad/CrewAI-Studio/main/img/ss4.png" alt="kickoff" style="width:50%;"/>
<img src="https://raw.githubusercontent.com/strnad/CrewAI-Studio/main/img/ss5.png" alt="kickoff" style="width:50%;"/><img src="https://raw.githubusercontent.com/strnad/CrewAI-Studio/main/img/ss6.png" alt="kickoff" style="width:50%;"/>

## Installation

### Using Virtual Environment

**For Virtual Environment**: Ensure you have Python installed. If you dont have python instaled, you can simply use the conda installer.

#### On Linux or MacOS

1. **Clone the repository (or use downloaded ZIP file)**:

   ```bash
   git clone https://github.com/strnad/CrewAI-Studio.git
   cd CrewAI-Studio
   ```

2. **Run the installation script**:

   ```bash
   ./install_venv.sh
   ```

3. **Run the application**:
   ```bash
   ./run_venv.sh
   ```

#### On Windows

1. **Clone the repository (or use downloaded ZIP file)**:

   ```powershell
   git clone https://github.com/strnad/CrewAI-Studio.git
   cd CrewAI-Studio
   ```

2. **Run the Conda installation script**:

   ```powershell
   ./install_venv.bat
   ```

3. **Run the application**:
   ```powershell
   ./run_venv.bat
   ```

### Using Conda

Conda will be installed locally in the project folder. No need for a pre-existing Conda installation.

#### On Linux

1. **Clone the repository (or use downloaded ZIP file)**:

   ```bash
   git clone https://github.com/strnad/CrewAI-Studio.git
   cd CrewAI-Studio
   ```

2. **Run the Conda installation script**:

   ```bash
   ./install_conda.sh
   ```

3. **Run the application**:
   ```bash
   ./run_conda.sh
   ```

#### On Windows

1. **Clone the repository (or use downloaded ZIP file)**:

   ```powershell
   git clone https://github.com/strnad/CrewAI-Studio.git
   cd CrewAI-Studio
   ```

2. **Run the Conda installation script**:

   ```powershell
   ./install_conda.bat
   ```

3. **Run the application**:
   ```powershell
   ./run_conda.bat
   ```

### One-Click Deployment

[![Deploy to RepoCloud](https://d16t0pc4846x52.cloudfront.net/deploylobe.svg)](https://repocloud.io/details/?app_id=318)

## Running with Docker Compose

To quickly set up and run CrewAI-Studio using Docker Compose, follow these steps:

### Prerequisites

- Ensure [Docker](https://docs.docker.com/get-docker/) and [Docker Compose](https://docs.docker.com/compose/install/) are installed on your system.

### Steps

1. Clone the repository:
```
git clone https://github.com/strnad/CrewAI-Studio.git
cd CrewAI-Studio
```

2. Create a .env file for configuration.  Edit for your own configuration:
```
cp .env_example .env
```

3. Start the application with Docker Compose:
```
docker-compose up --build
```

4. Access the application: http://localhost:8501

## Configuration

Before running the application, ensure you update the `.env` file with your API keys and other necessary configurations. An example `.env` file is provided for reference.

## Real-Time & Fresh Data Retrieval

To make the model aware of the current moment and retrieve up‑to‑date information:

1. Add the `CurrentDateTime` tool to your agent. The agent should call it first to anchor the current date (UTC + local).
2. Add at least one search / web tool (e.g. `DuckDuckGoSearchTool`, `SerperDevTool`, `EXASearchTool`, or `ScrapeWebsiteToolEnhanced`).
3. In your task prompt, explicitly instruct the agent to: (a) call `CurrentDateTime`, (b) perform targeted searches scoped by recent time intervals (e.g. "past 7 days"), (c) cite sources.
4. (Optional) Use scraping tools after search to pull full-page context before summarizing.

Example task prompt:
```
Research severe cybersecurity incidents reported in the last 10 days. First call CurrentDateTime to determine today's date, then use a search tool to find at least 5 recent credible sources (CISA, reputable news, vendor reports). Extract incident names, dates, impact, and provide a concise synthesis with source links.
```

## CurrentDateTime Tool

The `CurrentDateTime` tool returns:
- UTC ISO timestamp
- Local ISO timestamp
- Date, time, weekday
- Unix epoch (seconds)

This minimizes hallucinations around "current" events by giving the LLM an authoritative temporal context to frame subsequent searches.

## Enhanced CSV Search (Optional embedchain)

`CSVSearchToolEnhanced` attempts to use `embedchain` for semantic querying over CSV files. If `embedchain` (and its dependency `tiktoken`) cannot be installed (e.g. missing Rust compiler on some Python versions), the application will:
- Skip semantic indexing gracefully
- Return an explanatory message instead of crashing at startup

To enable full semantic CSV search:
```
pip install embedchain
```
If you encounter build errors for `tiktoken`, consider using Python 3.11/3.12 or install Rust (https://www.rust-lang.org/tools/install) before retrying.

## Recent Additions / Changes

| Date | Change |
|------|--------|
| 2025-09-25 | Added `CurrentDateTime` tool for temporal grounding |
| 2025-09-25 | Fixed duplicate Streamlit widget key collisions for agents & tasks (use IDs instead of names) |
| 2025-09-25 | Added graceful fallback for `CSVSearchToolEnhanced` when `embedchain` is absent |

## Syncing With Upstream (If You Forked)

If you forked from the original repository and want to pull future updates:
```
git remote add upstream https://github.com/strnad/CrewAI-Studio.git
git fetch upstream
git checkout main
git merge upstream/main   # or: git rebase upstream/main
git push origin main
```
Resolve any merge conflicts, then re-test the app (`./run_venv.bat` or `./run_venv.sh`).

## Suggested Prompt Pattern for Fresh News

```
You MUST first call CurrentDateTime to know today's date. Then perform multiple focused web searches with time qualifiers (e.g. "past week", specific month/year). Avoid answering until you have at least 5 distinct, dated sources. Return a summary with an ISO8601 date list and a Sources section.
```

## Troubleshooting
In case of problems:
- Delete the `venv/miniconda` folder and reinstall `crewai-studio`.
- Rename `crewai.db` (it contains your crews but sometimes new versions can break compatibility).
- Raise an issue and I will help you.

## Video tutorial
Video tutorial on CrewAI Studio made by Josh Poco

[![FREE CrewAI Studio GUI EASY AI Agent Creation!🤖 Open Source AI Agent Orchestration Self Hosted](https://img.youtube.com/vi/3Uxdggt88pY/hqdefault.jpg)](https://www.youtube.com/watch?v=3Uxdggt88pY)

## Star History

<a href="https://star-history.com/#strnad/CrewAI-Studio&Date">
 <picture>
   <source media="(prefers-color-scheme: dark)" srcset="https://api.star-history.com/svg?repos=strnad/CrewAI-Studio&type=Date&theme=dark" />
   <source media="(prefers-color-scheme: light)" srcset="https://api.star-history.com/svg?repos=strnad/CrewAI-Studio&type=Date" />
   <img alt="Star History Chart" src="https://api.star-history.com/svg?repos=strnad/CrewAI-Studio&type=Date" />
 </picture>   
</a>
