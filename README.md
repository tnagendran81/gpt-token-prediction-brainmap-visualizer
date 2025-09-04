# Token Prediction Brainmap Visualizer

**Interactive visualization tool that reveals how AI language models calculate probabilities for the next token prediction**
##  What This Tool Does

Ever wondered how ChatGPT, GPT-4, or Claude "decides" what word comes next? This tool pulls back the curtain on AI language models by visualizing the **exact probabilities** each model calculates for every possible next token.

**Key Features:**

- 🔍 **Real-time Probability Analysis** - See actual token probabilities from OpenAI's API
- 🧠 **Interactive Brainmap** - Hierarchical visualization showing decision trees
- 🎮 **Fully Interactive** - Scroll, zoom, pan through prediction sequences
- 📊 **Step-by-Step Breakdown** - Detailed analysis of each prediction step
- 🔄 **Model Comparison** - Compare different GPT models' prediction patterns

##  The Science Behind It

Language models don't "think" - they calculate probabilities. For every word in their vocabulary (~50,000+ words), they compute:

P(next_word | context) = mathematical_function(trained_parameters, input_context)

This tool visualizes these calculations in real-time, showing you:
- Which tokens the model considered
- Exact probability scores for each option
- Why certain words were chosen over others
- How different models make different choices
## Installation & Setup

### Prerequisites
- Python 3.8+
- OpenAI API key

### Quick Start

1. **Clone the repository**

https://github.com/tnagendran81/gpt-token-prediction-brainmap-visualizer/tree/main

2. **Install dependencies**

pip install -r requirements.txt

3. **Set up your API key**

cp .env.example .env

Edit .env and add your OpenAI API key

4. **Run the application**

python main.py

5. **Open your browser** to `http://localhost:7860`

### Detailed Setup Instructions

See [docs/SETUP.md](docs/SETUP.md) for comprehensive installation guide.

## 🎮 How to Use

### Basic Usage
1. **Enter your prompt** in the left input box
2. **Configure settings:**
- Choose your model (GPT-3.5, GPT-4, GPT-4o)
- Set max tokens (3-100)
- Select alternatives to show (1-10)
3. **Click "Generate Interactive Brainmap"**
4. **Explore the visualization:**
- 🖱️ **Scroll** to zoom in/out
- 👆 **Drag** to pan around
- 🖱️ **Double-click** to reset view
- ℹ️ **Hover** for detailed information

### Understanding the Visualization

**🔴 Red Nodes:** Selected tokens (highest probability)

**🔵 Blue Nodes:** Alternative token options 

**Thick Edges:** Selected prediction path. 
**Thin Edges:** Alternative prediction paths
**Node Size:** Proportional to token probability

### Example Analysis

For prompt: *"The future of artificial intelligence"*

Step 1: "The future of artificial intelligence"

    ├─ "will" (0.453) ← SELECTED
    ├─ "is" (0.287)
    └─ "could" (0.164)

Step 2: "The future of artificial intelligence will"

    ├─ "be" (0.521) ← SELECTED
    ├─ "revolutionize" (0.234)
    └─ "transform" (0.128)




## 🧪 Technical Details

### How It Works

1. **API Integration:** Uses OpenAI's `logprobs` parameter to get actual token probabilities
2. **Real-time Processing:** Processes each token prediction step sequentially
3. **Interactive Visualization:** Plotly-based brainmap with full zoom/pan capabilities
4. **Gradio Interface:** Clean, intuitive web interface

### Key Components

- **`OpenAITokenPredictionBrainMap`**: Core class handling API calls and probability extraction
- **`create_scrollable_brainmap()`**: Generates interactive Plotly visualization
- **`create_gradio_interface()`**: Web interface with input/output panels

### Supported Models

- GPT-3.5 Turbo
- GPT-4
- GPT-4 Turbo Preview  
- GPT-4o

## 📊 Example Use Cases

### 1. **Educational Tool**
- Understand how language models work
- Visualize the "thinking" process of AI
- Compare decision patterns across models

### 2. **Research & Analysis**
- Analyze model behavior on specific prompts
- Study probability distributions
- Compare model confidence across different contexts

### 3. **Prompt Engineering**
- See how different prompts affect predictions
- Optimize prompts for desired outcomes
- Understand model biases and preferences

## 🔬 Insights You'll Discover

**Model Differences:** GPT-4 vs GPT-3.5 make different probability calculations
**Context Matters:** Previous tokens heavily influence next token probabilities  
**Confidence Patterns:** Models are more confident about some predictions than others
**Creative Process:** "Temperature" settings affect probability distributions

## 🛡️ Privacy & Security

- ✅ Your API key stays in your environment variables
- ✅ No data is stored or logged
- ✅ Direct communication with OpenAI only
- ✅ Open source - verify the code yourself

## 🤝 Contributing

Contributions welcome! Please feel free to:

- 🐛 Report bugs
- 💡 Suggest features  
- 🔧 Submit pull requests
- 📚 Improve documentation

### Development Setup


## License

**[MIT](https://choosealicense.com/licenses/mit/)**

## 🙏 Acknowledgments

- **OpenAI** for providing the API with logprobs functionality
- **Plotly** for amazing interactive visualization capabilities
- **Gradio** for the intuitive web interface framework

## 📚 Related Reading

- [Medium Article: "Demystifying AI: How Language Models Calculate the Next Best Word"](your-medium-article-link)
- [OpenAI API Documentation](https://platform.openai.com/docs)

**⭐ If this tool helped you understand AI better, please star the repository!**

*Built with ❤️ for the AI community*
