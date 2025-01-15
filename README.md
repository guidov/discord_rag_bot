# RAG Discord RAG Bot

This Discord bot utilizes the Retrieval-Augmented Generation (RAG) approach to enhance the output of a large language model. It can work with deepseek or openai. dge base outside of its training data sources.

## Prerequisites

Before running the bot, make sure you have the following dependencies installed:

- Python 3.10 or higher

## Getting Started

1. Clone the repository to your local machine:

```bash
git clone https://github.com/guidov/discord_rag_bot
cd discord_rag_bot
```

2. Create a virtual environment (optional but recommended):

```bash
conda crate -n discord-rag-bot
conda activate discord-rag-bot
```

3. Install the required dependencies:

```bash
pip install -r requirements.txt
```

5. Rename the `.env.example` file to `.env` in the project root directory and add your tokens and api keys:

```env
DISCORD_BOT_TOKEN=your_bot_token_here
OPENAI_API_KEY=your_openai_api_key
DEEPSEEK_API_KEY=your_deepseek_api_key
```

## Running the Bot

```bash
python bot.py
```

This will launch the bot, and you should see "Ready" in the console once it has successfully connected to Discord.

## Bot Usage

The bot responds to a single slash command:

### `/query`

- **Description:** Enter your query :)
- **Options:**
  - `input_text` (required): The input text for the query.

### `/updatedb`

- **Description:** Updates your information database

