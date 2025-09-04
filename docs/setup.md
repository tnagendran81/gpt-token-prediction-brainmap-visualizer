# Detailed Setup Guide

## Step-by-Step Installation

### 1. Python Environment Setup

Check Python version (3.8+ required)
python --version

Create virtual environment (recommended)
python -m venv token-brainmap-env

Activate virtual environment
On Windows:
token-brainmap-env\Scripts\activate

On macOS/Linux:
source token-brainmap-env/bin/activate


### 2. Get OpenAI API Key

1. Visit [OpenAI Platform](https://platform.openai.com/)
2. Sign up or log in to your account
3. Navigate to API Keys section
4. Create a new API key
5. Copy the key (starts with `sk-`)

### 3. Configure Environment Variables

Copy example environment file
cp .env.example .env

Edit .env file with your API key
OPENAI_API_KEY=sk-your-actual-api-key-here


### 4. Install Dependencies

pip install -r requirements.txt


### 5. Run the Application

python main.py


### 6. Access the Interface

Open your browser and navigate to:
- Local: `http://localhost:7860`
- Network: `http://0.0.0.0:7860` (accessible from other devices)

## Troubleshooting

### Common Issues

**Issue: "OPENAI_API_KEY not found"**
- Solution: Ensure your `.env` file is in the same directory as `app.py`
- Verify the API key format starts with `sk-`

**Issue: "Module not found"**
- Solution: Ensure virtual environment is activated
- Reinstall requirements: `pip install -r requirements.txt`

**Issue: "API rate limit exceeded"**
- Solution: Wait a few minutes, or upgrade your OpenAI plan

### Performance Tips

- Use GPT-3.5-turbo for faster responses
- Limit max tokens (5-10) for quicker analysis
- Reduce top alternatives (2-3) for cleaner visualizations

## Deployment Options

### Local Network Access

python app.py

Access from other devices on your network via your IP


### Cloud Deployment
- **Hugging Face Spaces:** Upload to HF Spaces for free hosting
- **Replit:** Deploy directly on Replit platform
- **Heroku:** Use for more permanent deployment

### Environment Variables for Production

export OPENAI_API_KEY=your-key
export GRADIO_SERVER_NAME=0.0.0.0
export GRADIO_SERVER_PORT=7860


