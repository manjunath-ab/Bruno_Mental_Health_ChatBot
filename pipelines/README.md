## Installation

To install the required dependencies, run the following commands in your terminal:

```bash
pip install -q langchain-openai langchain playwright beautifulsoup4
playwright install

### Configuration
Set the environment variable OPENAI_API_KEY or load it from a .env file. You can use the following Python code in your script to load the environment variables:

import dotenv
dotenv.load_dotenv()
