# 🐾 AI Animal Expert System

An intelligent animal identification and analysis system that combines **Computer Vision**, **Large Language Models (LLMs)**, and **autonomous agents** to provide expert-level insights on animals from images.

## 🚀 Features

- 📸 **Animal Detection**: Upload an image and detect the most probable animal species using a fine-tuned ResNet50 model.
- 🧠 **Expert AI Analysis** (via Ollama + CrewAI agents):
  - 🍽️ Diet & Nutrition
  - 🏠 Habitat & Environment
  - 🎭 Behavior & Interaction
  - ❤️ Health & Veterinary Care
  - 🌍 Conservation Status
  - 🐕 Pet Suitability
- 📄 **Downloadable Reports** with contextual, location-aware guidance
- ⚡ Streamlit-based real-time UI with styled results

## 📚 Motivation

This project was built as part of my **self-learning journey** into:
- Computer Vision
- Generative AI & LLMs
- Agentic workflows
- Full-stack Python apps with Streamlit

I wanted to create something meaningful that combines AI and wildlife awareness.

## 🛠️ Tech Stack

| Component         | Tech Used                         |
|------------------|-----------------------------------|
| Frontend         | `Streamlit`                       |
| Image Analysis   | `PyTorch`, `TorchVision`, `ResNet50` |
| LLM Interface    | `CrewAI`, `Ollama`, `LLaMA 3 3B`  |
| Utilities        | `OpenCV`, `PIL`, `NumPy`, `requests` |
| Backend Logic    | `Python`                          |

## ⚙️ Installation

### 1. Clone the repository

```bash
git clone https://github.com/your-username/ai-animal-expert.git
cd ai-animal-expert
```

### 2. Set up a virtual environment

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Install and run Ollama

Install Ollama (if not installed):
```bash
curl -fsSL https://ollama.ai/install.sh | sh
```

Pull the required LLM:
```bash
ollama pull llama3.2:3b
```

Start the Ollama server:
```bash
ollama serve
```

### 5. Run the Streamlit app

```bash
streamlit run app.py
```

## 📷 How to Use

1. Upload an animal image
2. Click **"Identify Animal"**
3. Select an expert analysis type
4. (Optional) Add your location or any context
5. Receive detailed results powered by LLM agents
6. Download a full report

## 🧪 Sample Animals Supported

- 🐱 Cats (tabby, tiger, siamese, persian)
- 🐶 Dogs (retrievers, chihuahuas, shepherds)
- 🐘 Elephants, 🦁 Lions, 🐻 Bears, 🦍 Gorillas
- 🐦 Eagles, peacocks, ostriches
- 🦈 Sharks, 🐟 Goldfish
- 🐼 Pandas, 🐒 Primates, and more...

## 📂 Project Structure

```
.
├── app.py                 # Main Streamlit application
├── requirements.txt       # Python dependencies
├── README.md              # You're here
└── assets/                # (Optional) Images, screenshots
```

## 🤝 Contributing

Contributions are welcome!  
Whether it’s bug fixes, new agents, or better animal detection — feel free to fork, PR, or suggest improvements.

## 🧠 Credits

- [PyTorch](https://pytorch.org/)
- [Streamlit](https://streamlit.io/)
- [CrewAI](https://github.com/joaomdmoura/crewAI)
- [Ollama](https://ollama.ai/)
- [LLaMA 3](https://ai.meta.com/llama/)

## 📬 Contact

If you'd like to collaborate, connect, or ask questions, feel free to [connect on LinkedIn](https://www.linkedin.com/in/sehar-nisar-078b92260/)!

## 📄 License

MIT License — free to use and modify.
