import streamlit as st
import os
import cv2
import numpy as np
from PIL import Image
import torch
import torchvision.transforms as transforms
from torchvision.models import resnet50, ResNet50_Weights
import json
import requests
import time
import tempfile
from crewai import Agent, Task, Crew, LLM
import io
import logging
from datetime import datetime
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Page configuration
st.set_page_config(
    page_title="🐾 AI Animal Expert System",
    page_icon="🦁",
    layout="wide"
)

# Enhanced CSS styling
st.markdown("""
<style>
    .main-header {
        text-align: center;
        padding: 1rem 0;
        margin-bottom: 2rem;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 10px;
    }

    .main-header h1, .main-header p {
        color: white;
    }

    .analysis-card {
        background: rgb(148, 193, 242);
        color: white;
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
        border-left: 4px solid #667eea;
        box-shadow: 0 2px 4px rgba(64, 23, 125, 0.1);
    }

    /* All cards with custom background, but white text */
    .diet-card { 
        border-left-color: #28a745; 
        background: #28a745;
        color: white;
    }

    .habitat-card { 
        border-left-color: #17a2b8;
        background: #17a2b8;
        color: white;
    }

    .behavior-card { 
        border-left-color: #ffc107;
        background: #ffc107;
        color: white;
    }

    .health-card { 
        border-left-color: #dc3545;
        background: #dc3545;
        color: white;
    }

    .conservation-card { 
        border-left-color: #6f42c1;
        background: #6f42c1;
        color: white;
    }

    .pet-card { 
        border-left-color: #fd7e14;
        background: #fd7e14;
        color: white;
    }

    .confidence-high,
    .confidence-medium,
    .confidence-low {
        color: white;
        font-weight: bold;
    }

    .analysis-section {
        margin: 1rem 0;
        padding: 1rem;
        background: white;
        border-radius: 8px;
        border: 1px solid #0b2238;
        color: #0c263f;
    }

    .feature-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
        gap: 1rem;
        margin: 1rem 0;
    }

    .feature-card {
        background: #0c263f;
        padding: 1rem;
        border-radius: 8px;
        text-align: center;
        color: white;
        border: 1px solid #0c263f;
    }

    .stAlert > div,
    .stInfo > div,
    .stSuccess > div,
    .stWarning > div,
    .stError > div {
        color: white !important;
    }

    .stAlert > div { background-color: rgba(255, 255, 255, 0.1) !important; }
    .stInfo > div { background-color: rgba(255, 255, 255, 0.1) !important; }
    .stSuccess > div { background-color: rgba(40, 167, 69, 0.2) !important; }
    .stWarning > div { background-color: rgba(255, 193, 7, 0.2) !important; }
    .stError > div { background-color: rgba(220, 53, 69, 0.2) !important; }

    /* Sidebar styling */
    .css-1d391kg {
        background-color: #2e3440;
    }

    .analysis-results {
        color: white;
        background-color: #0c263f;
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
        border: 1px solid #0c263f;
    }

    .installation-guide {
        background: #0c263f;
        color: white;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #17a2b8;
        margin: 1rem 0;
    }

    .installation-guide code {
        background: #1b2c40;
        color: white;
        padding: 0.2rem 0.4rem;
        border-radius: 4px;
        font-family: monospace;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
def init_session_state():
    states = [
        'analysis_complete', 'analysis_results', 'llm_analysis_results',
        'selected_analysis_type', 'user_location', 'additional_context'
    ]
    for state in states:
        if state not in st.session_state:
            st.session_state[state] = None if 'results' in state else False
    if 'chat_history' not in st.session_state:
        st.session_state['chat_history'] = []

# Set up Groq API key (from environment variable or hardcoded)
GROQ_API_KEY = os.getenv('GROQ_API_KEY')
GROQ_API_BASE = 'https://api.groq.com/openai/v1'
GROQ_MODEL = 'llama-3.3-70b-versatile'

# LLM Analysis Functions
def create_specialized_agents():
    """Create all specialized agents for different analysis types using Groq API"""
    try:
        groq_llm = LLM(
            model=GROQ_MODEL,
            api_key=GROQ_API_KEY,
            base_url=GROQ_API_BASE
        )
    except Exception as e:
        groq_llm = GROQ_MODEL  # fallback, but should not be used
    agents = {
        'diet': Agent(
            role="Animal Nutritionist",
            goal="Analyze dietary requirements, feeding habits, and nutritional needs of animals",
            backstory="""Expert veterinary nutritionist with comprehensive knowledge of animal dietary patterns, \
            feeding behaviors, nutritional requirements across species, and optimal feeding protocols.""",
            llm=groq_llm,
            verbose=False,
            allow_delegation=False,
            max_iter=2
        ),
        'habitat': Agent(
            role="Wildlife Habitat Specialist", 
            goal="Determine optimal living conditions, habitat requirements, and environmental needs",
            backstory="""Conservation biologist specializing in animal habitats, environmental requirements, \
            climate preferences, and ecosystem interactions across different species worldwide.""",
            llm=groq_llm,
            verbose=False,
            allow_delegation=False,
            max_iter=2
        ),
        'behavior': Agent(
            role="Animal Behaviorist",
            goal="Predict and analyze animal behavior patterns, social needs, and interaction guidelines",
            backstory="""Certified animal behaviorist with expertise in ethology, animal psychology, \
            social structures, and safe human-animal interactions across various species.""",
            llm=groq_llm,
            verbose=False,
            allow_delegation=False,
            max_iter=2
        ),
        'health': Agent(
            role="Veterinary Health Advisor",
            goal="Provide health insights, common medical issues, and preventive care recommendations",
            backstory="""Experienced veterinarian with broad knowledge of animal health conditions, \
            preventive care protocols, breed-specific health issues, and wellness management.""",
            llm=groq_llm,
            verbose=False,
            allow_delegation=False,
            max_iter=2
        ),
        'conservation': Agent(
            role="Conservation Biologist",
            goal="Assess conservation status, ecological role, and environmental impact",
            backstory="""Wildlife conservation expert with deep knowledge of species protection, \
            ecological relationships, biodiversity conservation, and environmental impact assessment.""",
            llm=groq_llm,
            verbose=False,
            allow_delegation=False,
            max_iter=2
        ),
        'pet': Agent(
            role="Pet Care Specialist",
            goal="Evaluate pet suitability and provide comprehensive care guidance",
            backstory="""Professional pet care specialist with extensive experience in animal husbandry, \
            pet behavior management, and matching animals with appropriate care environments.""",
            llm=groq_llm,
            verbose=False,
            allow_delegation=False,
            max_iter=2
        )
    }
    return agents

def create_analysis_tasks(animal_name, analysis_type, user_location="", additional_context=""):
    """Create specific tasks based on analysis type"""
    
    tasks = {
        'diet': Task(
            description=f"""
            Provide comprehensive dietary analysis for {animal_name.replace('_', ' ')}:
            
            1. NATURAL DIET: What does this animal eat in the wild?
            2. NUTRITIONAL NEEDS: Key nutrients, vitamins, and minerals required
            3. FEEDING SCHEDULE: How often and when to feed
            4. FOOD SAFETY: Foods that are toxic or harmful to avoid
            5. PORTION GUIDELINES: Appropriate serving sizes
            6. SPECIAL CONSIDERATIONS: Age, season, or health-related dietary needs
            
            Additional context: {additional_context}
            
            Provide practical, actionable advice for proper nutrition.
            """,
            expected_output="Detailed dietary guide with feeding recommendations, safety warnings, and nutritional requirements.",
            agent=None  # Will be set when creating the crew
        ),
        
        'habitat': Task(
            description=f"""
            Analyze habitat and environmental requirements for {animal_name.replace('_', ' ')}:
            
            1. NATURAL HABITAT: Where does this animal naturally live?
            2. CLIMATE NEEDS: Temperature, humidity, and seasonal preferences
            3. SPACE REQUIREMENTS: Territory size and space needs
            4. ENVIRONMENTAL ENRICHMENT: What kind of environment setup is needed?
            5. LOCATION COMPATIBILITY: Suitability for {user_location or 'various locations'}
            6. LEGAL CONSIDERATIONS: Permits or restrictions for interaction/ownership
            
            Focus on creating optimal living conditions and environmental wellness.
            """,
            expected_output="Complete habitat analysis with environmental requirements and location-specific recommendations.",
            agent=None
        ),
        
        'behavior': Task(
            description=f"""
            Analyze behavioral characteristics and interaction guidelines for {animal_name.replace('_', ' ')}:
            
            1. NATURAL BEHAVIORS: Typical behavior patterns and social structure
            2. ACTIVITY PATTERNS: Daily routines, sleep cycles, active periods
            3. SOCIAL NEEDS: Solitary vs social, interaction with others
            4. HUMAN INTERACTION: Safe approaches and interaction guidelines
            5. STRESS INDICATORS: Signs of stress, fear, or discomfort to watch for
            6. ENRICHMENT ACTIVITIES: Mental stimulation and physical activities needed
            
            Context: {additional_context}
            
            Provide practical guidance for understanding and interacting safely with this animal.
            """,
            expected_output="Comprehensive behavioral profile with interaction safety guidelines and enrichment recommendations.",
            agent=None
        ),
        
        'health': Task(
            description=f"""
            Provide health and medical care information for {animal_name.replace('_', ' ')}:
            
            1. COMMON HEALTH ISSUES: Typical diseases, conditions, or health problems
            2. PREVENTIVE CARE: Vaccination schedules, regular check-ups, health monitoring
            3. WARNING SIGNS: Symptoms that require immediate veterinary attention
            4. LIFESPAN & AGING: Expected lifespan and age-related health considerations
            5. EXERCISE NEEDS: Physical activity requirements for optimal health
            6. GROOMING & HYGIENE: Maintenance needs for health and wellbeing
            
            Observed concerns: {additional_context}
            
            IMPORTANT: Always emphasize consulting qualified veterinarians for medical issues.
            """,
            expected_output="Health care guide with preventive measures, warning signs, and professional care recommendations.",
            agent=None
        ),
        
        'conservation': Task(
            description=f"""
            Assess conservation status and ecological impact of {animal_name.replace('_', ' ')}:
            
            1. CONSERVATION STATUS: Current protection status (IUCN, local regulations)
            2. POPULATION TRENDS: Are numbers increasing, stable, or declining?
            3. MAIN THREATS: Primary dangers to species survival
            4. ECOLOGICAL ROLE: Importance in ecosystem and environmental impact
            5. CONSERVATION EFFORTS: Current protection programs and initiatives
            6. HOW TO HELP: Practical ways individuals can support conservation
            
            Geographic focus: {user_location or 'global'}
            
            Provide actionable conservation information and ways to make a positive impact.
            """,
            expected_output="Conservation assessment with species status, threats, and actionable support recommendations.",
            agent=None
        ),
        
        'pet': Task(
            description=f"""
            Evaluate {animal_name.replace('_', ' ')} as a potential companion animal:
            
            1. PET SUITABILITY: Is this animal appropriate as a pet?
            2. EXPERIENCE LEVEL: What level of animal care experience is needed?
            3. DAILY CARE NEEDS: Time commitment and daily care requirements
            4. HOUSING REQUIREMENTS: Space, setup, and facility needs
            5. FINANCIAL COSTS: Estimated costs for proper care and maintenance
            6. LEGAL REQUIREMENTS: Permits, licenses, or legal restrictions
            7. ALTERNATIVES: If not suitable as pet, what are better options?
            
            Owner context: {additional_context}
            
            Provide honest, responsible guidance about pet ownership suitability.
            """,
            expected_output="Comprehensive pet suitability assessment with care requirements and responsible ownership guidance.",
            agent=None
        )
    }
    
    return tasks.get(analysis_type)

def run_llm_analysis(animal_name, analysis_type, user_location="", additional_context=""):
    """Run the selected LLM analysis"""
    try:
        agents = create_specialized_agents()
        agent = agents[analysis_type]
        
        task = create_analysis_tasks(animal_name, analysis_type, user_location, additional_context)
        
        # Fix: Set the agent for the task
        task.agent = agent
        
        crew = Crew(
            agents=[agent],
            tasks=[task],
            verbose=False
        )
        
        result = crew.kickoff()
        return result
        
    except Exception as e:
        logger.error(f"LLM Analysis failed: {e}")
        return f"Analysis failed: {str(e)}"

# Image Analysis Class
@st.cache_resource
def load_image_analyzer():
    try:
        return ImageAnalyzer()
    except Exception as e:
        logger.error(f"Failed to load image analyzer: {e}")
        st.error("Failed to load image analysis model.")
        return None

class ImageAnalyzer:
    def __init__(self):
        try:
            self.model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
            self.model.eval()
            
            self.transform = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                   std=[0.229, 0.224, 0.225])
            ])
            
            self.animal_classes = self._get_animal_classes()
            self.imagenet_labels = self._load_imagenet_labels()
            logger.info("ImageAnalyzer initialized successfully")
            
        except Exception as e:
            logger.error(f"Error initializing ImageAnalyzer: {e}")
            raise
    
    def _load_imagenet_labels(self):
        """Load ImageNet class labels for better animal identification"""
        try:
            imagenet_labels = {
                0: "tench", 1: "goldfish", 2: "great_white_shark", 3: "tiger_shark",
                4: "hammerhead", 5: "electric_ray", 6: "stingray", 7: "cock", 8: "hen",
                9: "ostrich", 10: "brambling", 11: "goldfinch", 12: "house_finch",
                13: "junco", 14: "indigo_bunting", 15: "robin", 16: "bulbul", 17: "jay",
                18: "magpie", 19: "chickadee", 20: "water_ouzel", 21: "kite", 22: "bald_eagle",
                23: "vulture", 24: "great_grey_owl", 80: "black_grouse", 81: "ptarmigan",
                82: "ruffed_grouse", 83: "prairie_chicken", 84: "peacock", 85: "quail",
                151: "chihuahua", 152: "japanese_spaniel", 153: "maltese_dog", 154: "pekinese",
                155: "shih_tzu", 207: "golden_retriever", 208: "labrador_retriever",
                235: "german_shepherd", 281: "tabby_cat", 282: "tiger_cat", 283: "persian_cat",
                284: "siamese_cat", 285: "egyptian_cat", 291: "lion", 292: "tiger",
                293: "cheetah", 294: "brown_bear", 340: "indian_elephant", 341: "african_elephant",
                342: "lesser_panda", 343: "giant_panda", 365: "orangutan", 366: "gorilla",
                367: "chimpanzee", 368: "gibbon"
            }
            return imagenet_labels
        except Exception as e:
            logger.warning(f"Could not load ImageNet labels: {e}")
            return {}
    
    def _get_animal_classes(self):
        """Enhanced animal class mappings"""
        return {
            # Domestic cats
            281: "tabby_cat", 282: "tiger_cat", 283: "persian_cat", 284: "siamese_cat", 285: "egyptian_cat",
            
            # Wild cats
            286: "cougar", 287: "lynx", 288: "leopard", 289: "snow_leopard", 290: "jaguar", 
            291: "lion", 292: "tiger", 293: "cheetah",
            
            # Dogs (sample)
            151: "chihuahua", 207: "golden_retriever", 208: "labrador_retriever", 235: "german_shepherd",
            
            # Birds
            7: "cock", 8: "hen", 9: "ostrich", 22: "bald_eagle", 84: "peacock",
            
            # Large mammals
            340: "indian_elephant", 341: "african_elephant", 343: "giant_panda",
            294: "brown_bear", 291: "lion", 292: "tiger",
            
            # Primates
            365: "orangutan", 366: "gorilla", 367: "chimpanzee", 368: "gibbon",
            
            # Marine life
            0: "tench", 1: "goldfish", 2: "great_white_shark", 3: "tiger_shark"
        }
    
    def analyze_image(self, image):
        try:
            if isinstance(image, np.ndarray):
                image = Image.fromarray(image)
            
            image = image.convert('RGB')
            input_tensor = self.transform(image).unsqueeze(0)
            
            with torch.no_grad():
                outputs = self.model(input_tensor)
                probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
                
            top_k = min(15, len(probabilities))
            top_prob, top_indices = torch.topk(probabilities, top_k)
            
            animal_scores = {}
            for i in range(top_k):
                class_idx = top_indices[i].item()
                confidence = top_prob[i].item()
                
                if class_idx in self.animal_classes:
                    animal = self.animal_classes[class_idx]
                    
                    if animal in animal_scores:
                        animal_scores[animal] += confidence
                    else:
                        animal_scores[animal] = confidence
            
            animals_found = []
            for animal, total_confidence in animal_scores.items():
                animals_found.append({
                    'animal': animal,
                    'confidence': min(total_confidence, 1.0)
                })
            
            animals_found.sort(key=lambda x: x['confidence'], reverse=True)
            return animals_found
            
        except Exception as e:
            logger.error(f"Error analyzing image: {e}")
            st.error(f"Error analyzing image: {str(e)}")
            return []

def get_confidence_class(confidence):
    """Get CSS class for confidence level"""
    if confidence > 0.4:
        return "confidence-high"
    elif confidence > 0.2:
        return "confidence-medium"
    else:
        return "confidence-low"

def display_results(results):
    """Display animal identification results"""
    if not results:
        st.error("No animals detected")
        return
    
    # Primary result
    primary = results[0]
    confidence_pct = primary['confidence'] * 100
    confidence_class = get_confidence_class(primary['confidence'])
    
    st.markdown(f"""
    <div class="analysis-card">
        <h3>🎯 Identified Animal: {primary['animal'].replace('_', ' ').title()}</h3>
        <p class="{confidence_class}">Confidence: {confidence_pct:.1f}%</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Progress bar
    st.progress(primary['confidence'])
    
    # All results
    if len(results) > 1:
        st.subheader("All Detections")
        for i, result in enumerate(results[:5], 1):
            conf_pct = result['confidence'] * 100
            animal_name = result['animal'].replace('_', ' ').title()
            st.write(f"{i}. **{animal_name}** - {conf_pct:.1f}%")

def display_analysis_results(analysis_type, results):
    """Display LLM analysis results with appropriate styling"""
    
    type_configs = {
        'diet': {'icon': '🍽️', 'title': 'Diet & Nutrition Analysis', 'class': 'diet-card'},
        'habitat': {'icon': '🏠', 'title': 'Habitat & Environment', 'class': 'habitat-card'},
        'behavior': {'icon': '🎭', 'title': 'Behavior & Interaction', 'class': 'behavior-card'},
        'health': {'icon': '❤️', 'title': 'Health & Veterinary Care', 'class': 'health-card'},
        'conservation': {'icon': '🌍', 'title': 'Conservation & Ecology', 'class': 'conservation-card'},
        'pet': {'icon': '🐕', 'title': 'Pet Suitability Assessment', 'class': 'pet-card'}
    }
    
    config = type_configs.get(analysis_type, {'icon': '📋', 'title': 'Analysis Results', 'class': 'analysis-card'})
    
    st.markdown(f"""
    <div class="analysis-card {config['class']}">
        <h3>{config['icon']} {config['title']}</h3>
    </div>
    """, unsafe_allow_html=True)
    
    # Display results
    if hasattr(results, 'raw'):
        content = results.raw
    else:
        content = str(results)
    
    # Display the entire result as markdown, which will correctly render headings
    st.markdown(content)

def display_installation_guide():
    """Display Ollama installation guide"""
    st.markdown("""
    <div class="installation-guide">
        <h4>🤖 Enable Expert AI Analysis</h4>
        <p><strong>To get detailed expert analysis, install Ollama:</strong></p>
        
        <p><strong>1. Install Ollama:</strong></p>
        <ul>
            <li><strong>Windows/Mac:</strong> Download from <a href="https://ollama.ai" target="_blank">ollama.ai</a></li>
            <li><strong>Linux:</strong> <code>curl -fsSL https://ollama.ai/install.sh | sh</code></li>
        </ul>
        
        <p><strong>2. Pull the required model:</strong></p>
        <code>ollama pull llama3.2:3b</code>
        
        <p><strong>3. Start Ollama service:</strong></p>
        <code>ollama serve</code>
        
        <p><strong>4. Refresh this page</strong></p>
        
        <p><em>Once installed, you'll have access to expert analysis including:</em></p>
        <ul>
            <li>🍽️ Detailed diet and nutrition guidance</li>
            <li>🏠 Habitat and environmental requirements</li>
            <li>🎭 Behavioral analysis and interaction tips</li>
            <li>❤️ Health and veterinary care advice</li>
            <li>🌍 Conservation status and ecological impact</li>
            <li>🐕 Pet suitability assessment</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

def main():
    """Enhanced main function with LLM analysis capabilities"""
    init_session_state()
    
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>🐾 AI Animal Expert System</h1>
        <p>Upload an image to identify animals and get expert analysis</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Main content layout - single column for a cleaner look
    st.header("📸 Image Upload & Analysis")
    uploaded_file = st.file_uploader(
        "Choose an animal image",
        type=['png', 'jpg', 'jpeg', 'webp'],
        help="Upload a clear image of an animal for identification"
    )
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_container_width=True)
        st.info(f"**Size:** {image.size[0]} × {image.size[1]} pixels | **Format:** {image.format}")
        if st.button("🔍 Identify Animal", type="primary"):
            with st.spinner("🔍 Analyzing image..."):
                analyzer = load_image_analyzer()
                if analyzer is None:
                    st.error("Failed to load image analyzer")
                    return
                analysis_results = analyzer.analyze_image(image)
                st.session_state.analysis_results = analysis_results
                if analysis_results:
                    st.session_state.analysis_complete = True
                    st.session_state.chat_history.append({
                        'sender': 'user',
                        'message': "I uploaded an animal image for identification",
                        'timestamp': datetime.now().strftime('%H:%M')
                    })
                else:
                    st.error("No animals detected in image")

    # Analysis Results Section - Full Width, single column
    if st.session_state.analysis_complete and st.session_state.analysis_results:
        st.markdown("---")
        st.header("🎯 Analysis Results")
        display_results(st.session_state.analysis_results)
        # Expert Analysis Section
        if GROQ_API_KEY and GROQ_API_KEY != 'YOUR_GROQ_API_KEY_HERE':
            st.markdown("---")
            st.header("🤖 Expert Analysis")
            analysis_options = {
                'diet': '🍽️ Diet & Nutrition Analysis',
                'habitat': '🏠 Habitat & Environment Suitability', 
                'behavior': '🎭 Behavior & Interaction Guide',
                'health': '❤️ Health & Veterinary Care',
                'conservation': '🌍 Conservation & Ecological Impact',
                'pet': '🐕 Pet Suitability Assessment'
            }
            selected_analysis = st.selectbox(
                "Choose analysis type:",
                options=list(analysis_options.keys()),
                format_func=lambda x: analysis_options[x]
            )
            user_location = st.text_input("Your location:", placeholder="e.g., California, USA")
            additional_context = st.text_input("Additional context:", placeholder="e.g., found injured")
            primary_animal = st.session_state.analysis_results[0]['animal']
            animal_display_name = primary_animal.replace('_', ' ').title()
            if st.button(f"🚀 Get Expert Analysis", type="primary"):
                with st.spinner(f"🤖 Analyzing {animal_display_name}..."):
                    try:
                        result = run_llm_analysis(primary_animal, selected_analysis, user_location, additional_context)
                        st.session_state.llm_analysis_results = result
                        st.session_state.selected_analysis_type = selected_analysis
                    except Exception as e:
                        st.error(f"Expert analysis failed: {str(e)}")
                        logger.error(f"LLM Analysis error: {e}")
            if st.session_state.llm_analysis_results and st.session_state.selected_analysis_type:
                st.markdown("---")
                display_analysis_results(st.session_state.selected_analysis_type, st.session_state.llm_analysis_results)
                if st.button("💾 Save Analysis"):
                    analysis_text = f"Animal: {animal_display_name}\n"
                    analysis_text += f"Analysis Type: {analysis_options[st.session_state.selected_analysis_type]}\n"
                    analysis_text += f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n"
                    if hasattr(st.session_state.llm_analysis_results, 'raw'):
                        analysis_text += st.session_state.llm_analysis_results.raw
                    else:
                        analysis_text += str(st.session_state.llm_analysis_results)
                    st.download_button(
                        label="📄 Download Report",
                        data=analysis_text,
                        file_name=f"{primary_animal}_{st.session_state.selected_analysis_type}_analysis.txt",
                        mime="text/plain"
                    )
                if st.button("🔄 Reset Analysis"):
                    for key in ['analysis_complete', 'analysis_results', 'llm_analysis_results']:
                        st.session_state[key] = None if 'results' in key else False
                    st.rerun()
        else:
            st.info("Set your Groq API key in the code or as an environment variable to enable expert analysis.")

    # Feature showcase for first-time users
    if not st.session_state.analysis_complete:
        st.markdown("---")
        st.header("🌟 Available Analysis Types")
        st.markdown("""
        ### Primary Features
        - 🍽️ **Diet & Nutrition**: Feeding guides and requirements
        - 🏠 **Habitat & Environment**: Living conditions and needs
        - 🎭 **Behavior & Interaction**: Safe handling guidelines
        ### Additional Insights
        - ❤️ **Health & Care**: Medical and wellness information
        - 🌍 **Conservation**: Species status and protection
        - 🐕 **Pet Suitability**: Companion animal assessment
        """)

if __name__ == "__main__":
    main()