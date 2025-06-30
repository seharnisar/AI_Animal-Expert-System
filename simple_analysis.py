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

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Page configuration
st.set_page_config(
    page_title="🐾 AI Animal Recognition",
    page_icon="🦁",
    layout="wide"
)

# Minimal CSS styling
st.markdown("""
<style>
    .main-header {
        text-align: center;
        padding: 1rem 0;
        margin-bottom: 2rem;
    }
    
    .main-header h1 {
        color: #667eea;
        margin: 0;
        font-size: 2.5rem;
        font-weight: 700;
    }
    
    .main-header p {
        color: #666;
        margin: 0.5rem 0 0 0;
        font-size: 1.1rem;
    }
    
    .result-box {
        background: #f8f9fa;
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
        border-left: 4px solid #667eea;
    }
    
    .confidence-high { color: #28a745; font-weight: bold; }
    .confidence-medium { color: #ffc107; font-weight: bold; }
    .confidence-low { color: #dc3545; font-weight: bold; }
    
    .status-item {
        padding: 0.5rem 0;
        border-bottom: 1px solid #eee;
    }
    
    .stButton > button {
        background: #667eea;
        color: white;
        border: none;
        border-radius: 5px;
        padding: 0.5rem 1rem;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
def init_session_state():
    if 'analysis_complete' not in st.session_state:
        st.session_state.analysis_complete = False
    if 'analysis_results' not in st.session_state:
        st.session_state.analysis_results = None
    if 'crewai_results' not in st.session_state:
        st.session_state.crewai_results = None

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
            # Use multiple models for better accuracy
            self.model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
            self.model.eval()
            
            # Standard transform - simplified to one transform for stability
            self.transform = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                   std=[0.229, 0.224, 0.225])
            ])
            
            self.animal_classes = self._get_animal_classes()
            # Load ImageNet class labels - fixed method call
            self.imagenet_labels = self._load_imagenet_labels()
            logger.info("ImageAnalyzer initialized successfully")
            
        except Exception as e:
            logger.error(f"Error initializing ImageAnalyzer: {e}")
            raise
    
    def _load_imagenet_labels(self):
        """Load ImageNet class labels for better animal identification"""
        try:
            # Basic ImageNet animal class names (subset for common animals)
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
        """Enhanced animal class mappings with more specific categories"""
        return {
            # Domestic cats (more specific)
            281: "tabby_cat", 282: "tiger_cat", 283: "persian_cat", 284: "siamese_cat", 
            285: "egyptian_cat", 
            
            # Wild cats
            286: "cougar", 287: "lynx", 288: "leopard", 289: "snow_leopard", 
            290: "jaguar", 291: "lion", 292: "tiger", 293: "cheetah",
            
            # Dogs (breed-specific for better accuracy)
            151: "chihuahua", 152: "japanese_spaniel", 153: "maltese_dog", 154: "pekinese",
            155: "shih_tzu", 156: "blenheim_spaniel", 157: "papillon", 158: "toy_terrier",
            159: "rhodesian_ridgeback", 160: "afghan_hound", 161: "basset", 162: "beagle",
            163: "bloodhound", 164: "bluetick", 165: "black_and_tan_coonhound", 166: "walker_hound",
            167: "english_foxhound", 168: "redbone", 169: "borzoi", 170: "irish_wolfhound",
            171: "italian_greyhound", 172: "whippet", 173: "ibizan_hound", 174: "norwegian_elkhound",
            175: "otterhound", 176: "saluki", 177: "scottish_deerhound", 178: "weimaraner",
            179: "staffordshire_bullterrier", 180: "american_staffordshire_terrier", 
            181: "bedlington_terrier", 182: "border_terrier", 183: "kerry_blue_terrier",
            184: "irish_terrier", 185: "norfolk_terrier", 186: "norwich_terrier",
            187: "yorkshire_terrier", 188: "wire_haired_fox_terrier", 189: "lakeland_terrier",
            190: "sealyham_terrier", 191: "airedale", 192: "cairn", 193: "australian_terrier",
            194: "dandie_dinmont", 195: "boston_bull", 196: "miniature_schnauzer",
            197: "giant_schnauzer", 198: "standard_schnauzer", 199: "scotch_terrier",
            200: "tibetan_terrier", 201: "silky_terrier", 202: "soft_coated_wheaten_terrier",
            203: "west_highland_white_terrier", 204: "lhasa", 205: "flat_coated_retriever",
            206: "curly_coated_retriever", 207: "golden_retriever", 208: "labrador_retriever",
            209: "chesapeake_bay_retriever", 210: "german_shorthaired_pointer", 
            211: "vizsla", 212: "english_setter", 213: "irish_setter", 214: "gordon_setter",
            215: "brittany_spaniel", 216: "clumber", 217: "english_springer", 218: "welsh_springer_spaniel",
            219: "cocker_spaniel", 220: "sussex_spaniel", 221: "irish_water_spaniel",
            222: "kuvasz", 223: "schipperke", 224: "groenendael", 225: "malinois",
            226: "briard", 227: "kelpie", 228: "komondor", 229: "old_english_sheepdog",
            230: "shetland_sheepdog", 231: "collie", 232: "border_collie", 233: "bouvier_des_flandres",
            234: "rottweiler", 235: "german_shepherd", 236: "doberman", 237: "miniature_pinscher",
            238: "greater_swiss_mountain_dog", 239: "bernese_mountain_dog", 240: "appenzeller",
            241: "entlebucher", 242: "boxer", 243: "bull_mastiff", 244: "tibetan_mastiff",
            245: "french_bulldog", 246: "great_dane", 247: "saint_bernard", 248: "eskimo_dog",
            249: "malamute", 250: "siberian_husky", 251: "affenpinscher", 252: "basenji",
            253: "pug", 254: "leonberg", 255: "newfoundland", 256: "great_pyrenees",
            257: "samoyed", 258: "pomeranian", 259: "chow", 260: "keeshond",
            261: "brabancon_griffon", 262: "pembroke", 263: "cardigan", 264: "toy_poodle",
            265: "miniature_poodle", 266: "standard_poodle", 267: "mexican_hairless", 268: "dingo",
            
            # Birds (more specific)
            7: "cock", 8: "hen", 9: "ostrich", 10: "brambling", 11: "goldfinch",
            12: "house_finch", 13: "junco", 14: "indigo_bunting", 15: "robin",
            16: "bulbul", 17: "jay", 18: "magpie", 19: "chickadee", 20: "water_ouzel",
            21: "kite", 22: "bald_eagle", 23: "vulture", 24: "great_grey_owl",
            80: "black_grouse", 81: "ptarmigan", 82: "ruffed_grouse", 83: "prairie_chicken",
            84: "peacock", 85: "quail", 86: "partridge", 87: "african_grey", 88: "macaw",
            89: "sulphur_crested_cockatoo", 90: "lorikeet", 91: "coucal", 92: "bee_eater",
            93: "hornbill", 94: "hummingbird", 95: "jacamar", 96: "toucan", 97: "drake",
            98: "red_breasted_merganser", 99: "goose", 100: "black_swan",
            127: "white_stork", 128: "black_stork", 129: "spoonbill", 130: "flamingo",
            131: "little_blue_heron", 132: "american_egret", 133: "bittern", 134: "crane",
            135: "limpkin", 136: "european_gallinule", 137: "american_coot", 138: "bustard",
            139: "ruddy_turnstone", 140: "red_backed_sandpiper", 141: "redshank",
            142: "dowitcher", 143: "oystercatcher", 144: "pelican", 145: "king_penguin",
            
            # Fish
            0: "tench", 1: "goldfish", 2: "great_white_shark", 3: "tiger_shark",
            4: "hammerhead", 5: "electric_ray", 6: "stingray",
            390: "eel", 391: "coho", 392: "rock_beauty", 393: "anemone_fish",
            394: "sturgeon", 395: "gar", 396: "lionfish", 397: "puffer",
            
            # Large mammals (more specific)
            340: "indian_elephant", 341: "african_elephant", 342: "lesser_panda",
            343: "giant_panda",
            
            # Wild animals
            294: "brown_bear", 295: "american_black_bear", 296: "ice_bear", 297: "sloth_bear",
            298: "mongoose", 299: "meerkat", 300: "tiger_beetle", 301: "ladybug",
            302: "ground_beetle", 303: "long_horned_beetle", 304: "leaf_beetle",
            305: "dung_beetle", 306: "rhinoceros_beetle", 307: "weevil", 308: "fly",
            309: "bee", 310: "ant", 311: "grasshopper", 312: "cricket", 313: "walking_stick",
            314: "cockroach", 315: "mantis", 316: "cicada", 317: "leafhopper",
            318: "lacewing", 319: "dragonfly", 320: "damselfly",
            
            # Primates
            365: "orangutan", 366: "gorilla", 367: "chimpanzee", 368: "gibbon",
            369: "siamang", 370: "guenon", 371: "patas", 372: "baboon",
            373: "macaque", 374: "langur",
            
            # Small mammals
            330: "hamster", 331: "porcupine", 332: "fox_squirrel", 333: "marmot",
            334: "beaver", 335: "guinea_pig", 336: "sorrel", 337: "zebra",
            338: "hog", 339: "wild_boar",
            
            # Reptiles and amphibians
            35: "wood_turtle", 36: "leatherback_turtle", 37: "mud_turtle", 38: "terrapin",
            39: "box_turtle", 40: "banded_gecko", 41: "common_iguana", 42: "american_chameleon",
            43: "whiptail", 44: "agama", 45: "frilled_lizard", 46: "alligator_lizard",
            47: "gila_monster", 48: "green_lizard", 49: "african_chameleon",
            50: "komodo_dragon", 51: "african_crocodile", 52: "american_alligator",
            53: "triceratops", 54: "thunder_snake", 55: "ringneck_snake",
            56: "hognose_snake", 57: "green_snake", 58: "king_snake",
            59: "garter_snake", 60: "water_snake", 61: "vine_snake",
            62: "night_snake", 63: "boa_constrictor",
            
            # Marine mammals
            147: "killer_whale", 148: "dugong", 149: "sea_lion", 150: "seal",
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

def test_ollama_connection():
    try:
        response = requests.get("http://localhost:11434/api/tags", timeout=5)
        if response.status_code == 200:
            models = response.json().get('models', [])
            available_models = [model['name'] for model in models]
            return 'llama3.2:3b' in available_models
        return False
    except:
        return False

def create_crewai_agents():
    try:
        ollama_llm = LLM(
            model="ollama/llama3.2:3b",
            base_url="http://localhost:11434"
        )
    except:
        ollama_llm = "ollama/llama3.2:3b"
    
    classifier_agent = Agent(
        role="Animal Identifier",
        goal="Identify animals in images accurately",
        backstory="Expert in animal identification with computer vision analysis.",
        llm=ollama_llm,
        verbose=False,
        allow_delegation=False,
        max_iter=2
    )

    description_agent = Agent(
        role="Animal Expert", 
        goal="Provide descriptions of identified animals",
        backstory="Zoologist specializing in animal characteristics and behavior.",
        llm=ollama_llm,
        verbose=False,
        allow_delegation=False,
        max_iter=2
    )
    
    return classifier_agent, description_agent

def create_crewai_tasks(analysis_results, agents):
    classifier_agent, description_agent = agents
    
    top_results = analysis_results[:2]
    
    task1 = Task(
        description=f"Identify the primary animal from these results: {json.dumps(top_results)}",
        expected_output="Clear animal identification with confidence level.",
        agent=classifier_agent
    )

    task2 = Task(
        description="Provide a brief description of the identified animal including basic facts.",
        expected_output="Brief description with key characteristics and facts and some fun facts.",
        agent=description_agent
    )
    
    return [task1, task2]

def get_confidence_class(confidence):
    if confidence > 0.4:
        return "confidence-high"
    elif confidence > 0.2:
        return "confidence-medium"
    else:
        return "confidence-low"

def display_results(results):
    if not results:
        st.error("No animals detected")
        return
    
    # Primary result
    primary = results[0]
    confidence_pct = primary['confidence'] * 100
    confidence_class = get_confidence_class(primary['confidence'])
    
    st.markdown(f"""
    <div class="result-box">
        <h3>🎯 Detected: {primary['animal'].replace('_', ' ').title()}</h3>
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

def main():
    init_session_state()
    
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>🐾 AI Animal Recognition</h1>
        <p>Upload an image to identify animals</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar - simplified
    with st.sidebar:
        st.header("System Status")
        
        # Ollama status
        ollama_status = test_ollama_connection()
        if ollama_status:
            st.success("✅ AI Connected")
        else:
            st.warning("⚠️ AI Disconnected")
        
        # PyTorch status
        try:
            st.success(f"✅ PyTorch {torch.__version__}")
        except:
            st.error("❌ PyTorch Error")
    
    # Main content
    col1, col2 = st.columns(2)
    
    with col1:
        st.header("📁 Upload Image")
        
        uploaded_file = st.file_uploader(
            "Choose an image",
            type=['png', 'jpg', 'jpeg', 'webp']
        )
        
        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Image", use_container_width=True)
            
            # Simple image info
            st.write(f"**Size:** {image.size[0]} × {image.size[1]} pixels")
            st.write(f"**Format:** {image.format}")
            
            # Analyze button
            if st.button("🔍 Analyze", type="primary"):
                with st.spinner("Analyzing..."):
                    analyzer = load_image_analyzer()
                    
                    if analyzer is None:
                        st.error("Failed to load analyzer")
                        return
                    
                    analysis_results = analyzer.analyze_image(image)
                    st.session_state.analysis_results = analysis_results
                    
                    if analysis_results:
                        st.session_state.analysis_complete = True
                        st.success("✅ Analysis completed!")
                        
                        # Try CrewAI if available
                        if ollama_status and len(analysis_results) > 0:
                            try:
                                with st.spinner("Getting detailed analysis..."):
                                    agents = create_crewai_agents()
                                    tasks = create_crewai_tasks(analysis_results, agents)
                                    crew = Crew(agents=agents, tasks=tasks, verbose=False)
                                    crewai_result = crew.kickoff()
                                    st.session_state.crewai_results = crewai_result
                            except Exception as e:
                                logger.error(f"CrewAI failed: {e}")
                    else:
                        st.error("No animals detected")
            
            # Reset button
            if st.button("🔄 Reset"):
                st.session_state.analysis_complete = False
                st.session_state.analysis_results = None
                st.session_state.crewai_results = None
                st.rerun()
    
    with col2:
        st.header("🎯 Results")
        
        if st.session_state.analysis_complete and st.session_state.analysis_results:
            display_results(st.session_state.analysis_results)
            
            # AI Analysis
            if st.session_state.crewai_results:
                st.subheader("🤖 AI Analysis")
                try:
                    if hasattr(st.session_state.crewai_results, 'raw'):
                        ai_result = st.session_state.crewai_results.raw
                    else:
                        ai_result = str(st.session_state.crewai_results)
                    st.write(ai_result)
                except:
                    st.write(st.session_state.crewai_results)
        else:
            st.info("Upload an image to start analysis")

if __name__ == "__main__":
    main()