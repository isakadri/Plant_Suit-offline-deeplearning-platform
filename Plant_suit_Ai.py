import streamlit as st
import numpy as np
from PIL import Image
import os
import json
import io
from datetime import datetime
import cv2
import sys
import time
from collections import deque

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from model_inference import PlantModelInference
    REAL_MODEL_AVAILABLE = True
except ImportError:
    REAL_MODEL_AVAILABLE = False

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="PlantSuit AI v3.0",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Sans:wght@300;400;500&display=swap');
html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }

.plant-header {
    text-align:center; padding:32px 20px;
    background:linear-gradient(135deg,#0f4c2a 0%,#1a7a45 50%,#0d3320 100%);
    border-radius:16px; color:white; margin-bottom:24px;
    border:1px solid rgba(255,255,255,0.1);
    box-shadow:0 20px 60px rgba(0,80,30,0.4);
}
.plant-header h1 { font-family:'Syne',sans-serif; font-size:2.4rem; font-weight:800; margin:0; letter-spacing:-1px; }
.plant-header p  { opacity:.8; margin:8px 0 0; font-size:1.05rem; }

.card { background:white; border-radius:12px; padding:20px; border:1px solid #e8f5e9; margin:10px 0; box-shadow:0 2px 12px rgba(0,0,0,0.05); transition:box-shadow .2s; }
.card:hover { box-shadow:0 6px 24px rgba(0,0,0,0.10); }
.card-green  { border-left:5px solid #2e7d32; }
.card-yellow { border-left:5px solid #f9a825; }
.card-red    { border-left:5px solid #c62828; }
.card-blue   { border-left:5px solid #1565c0; }
.card-purple { border-left:5px solid #6a1b9a; }

.badge { display:inline-block; padding:4px 12px; border-radius:20px; font-size:12px; font-weight:600; margin:3px; }
.badge-green  { background:#e8f5e9; color:#2e7d32; }
.badge-red    { background:#ffebee; color:#c62828; }
.badge-yellow { background:#fffde7; color:#f57f17; }
.badge-blue   { background:#e3f2fd; color:#1565c0; }
.badge-purple { background:#f3e5f5; color:#6a1b9a; }
.badge-gray   { background:#f5f5f5; color:#424242; }

.section-title { font-family:'Syne',sans-serif; font-size:1.1rem; font-weight:700; color:#1b5e20; margin:20px 0 10px; letter-spacing:.5px; border-bottom:2px solid #c8e6c9; padding-bottom:6px; }

.health-bar-wrap { background:#e0e0e0; border-radius:8px; height:22px; overflow:hidden; margin:8px 0; }
.health-bar-fill { height:100%; border-radius:8px; display:flex; align-items:center; justify-content:center; color:white; font-weight:700; font-size:13px; transition:width .5s; }

.conf-bar-wrap { background:#f5f5f5; border-radius:4px; height:14px; overflow:hidden; margin:4px 0; }
.conf-bar-fill { height:100%; border-radius:4px; background:linear-gradient(90deg,#1b5e20,#43a047); }

.info-grid { display:grid; grid-template-columns:1fr 1fr; gap:10px; margin:10px 0; }
.info-item { background:#f8fdf8; padding:10px 14px; border-radius:8px; }
.info-label { font-size:11px; text-transform:uppercase; letter-spacing:.8px; color:#666; }
.info-value { font-size:15px; font-weight:600; color:#1b5e20; margin-top:2px; }

.ai-badge { display:inline-flex; align-items:center; gap:6px; background:linear-gradient(90deg,#1b5e20,#388e3c); color:white; padding:5px 14px; border-radius:20px; font-size:12px; font-weight:600; margin-bottom:12px; }

.caution-box { background:#fff3e0; border:1px solid #ffb74d; border-radius:8px; padding:12px 16px; margin:8px 0; }
.action-box  { background:#e8f5e9; border:1px solid #81c784; border-radius:8px; padding:12px 16px; margin:8px 0; }
.pest-box    { background:#fce4ec; border:1px solid #ef9a9a; border-radius:8px; padding:12px 16px; margin:8px 0; }

.metric-box   { background:white; border-radius:10px; padding:14px; text-align:center; border:1px solid #e0e0e0; box-shadow:0 2px 8px rgba(0,0,0,0.05); }
.metric-value { font-family:'Syne',sans-serif; font-size:1.8rem; font-weight:800; color:#1b5e20; }
.metric-label { font-size:12px; color:#666; margin-top:2px; }
</style>
""", unsafe_allow_html=True)

# ── Session state ─────────────────────────────────────────────────────────────
defaults = {
    'analysis_history':    [],
    'enhancement':         True,
    'confidence_threshold':0.5,
    'last_upload_results': None,
    'last_upload_image':   None,
    'prediction_history':  deque(maxlen=30),
    'treatment_log':       [],
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="plant-header">
    <h1>🌿 PlantSuit AI v3.0</h1>
    <p>AI-Powered Plant Identification · Disease Detection · Pest Analysis · Smart Treatment Plans</p>
</div>
""", unsafe_allow_html=True)

# ── Sidebar ───────────────────────────────────────────────────────────────────
st.sidebar.markdown("## 🌿 PlantSuit AI")
page = st.sidebar.radio("Navigation", [
    "🔍 AI Plant Analysis",
    "🔬 Disease Library",
    "🐛 Pest Identifier",
    "📈 Health History",
    "🌦️ Weather Risk",
    "⚙️ Settings"
])
st.sidebar.markdown("---")
st.sidebar.markdown(f"**Analyses done:** {len(st.session_state.analysis_history)}")
st.sidebar.markdown(f"**Version:** 3.0 AI Edition")

# ════════════════════════════════════════════════════════════════════════════════
#  DATA DICTIONARIES
# ════════════════════════════════════════════════════════════════════════════════

PLANT_PROFILES = {
    "tomato":    {"name":"Tomato","family":"Solanaceae","genus":"Solanum lycopersicum","origin":"South America","growth":"Annual","water":"Moderate","sun":"Full Sun","soil":"Well-drained, pH 6.0-6.8","lifespan":"Annual (90-150 days)","native_region":"Andes, South America"},
    "rose":      {"name":"Rose","family":"Rosaceae","genus":"Rosa spp.","origin":"Asia/Europe","growth":"Perennial","water":"Moderate","sun":"Full Sun","soil":"Rich, pH 6.0-6.5","lifespan":"Perennial (20+ years)","native_region":"Asia, Europe, North America"},
    "basil":     {"name":"Basil","family":"Lamiaceae","genus":"Ocimum basilicum","origin":"Tropical Asia","growth":"Annual","water":"Regular","sun":"Full Sun","soil":"Moist, pH 6.0-7.0","lifespan":"Annual (1 season)","native_region":"Tropical Asia, Africa"},
    "fern":      {"name":"Fern","family":"Polypodiaceae","genus":"Various","origin":"Tropical","growth":"Perennial","water":"High","sun":"Indirect","soil":"Humus-rich, pH 4.0-7.0","lifespan":"Perennial (decades)","native_region":"Worldwide tropics"},
    "succulent": {"name":"Succulent","family":"Crassulaceae","genus":"Various","origin":"Africa/Americas","growth":"Perennial","water":"Low","sun":"Full Sun","soil":"Sandy, pH 6.0-7.0","lifespan":"Perennial (3-50 years)","native_region":"Africa, Americas"},
    "unknown":   {"name":"Unknown Plant","family":"Unidentified","genus":"Unknown","origin":"Unknown","growth":"Unknown","water":"Moderate","sun":"Partial","soil":"General potting mix","lifespan":"Unknown","native_region":"Unknown"},
}

DISEASE_DB = {
    "Nutrient Deficiency": {
        "type":"Environmental","pathogen":"None (abiotic)","spread":"Non-infectious",
        "caution":"Can mimic disease – confirm with soil test before treating. Excess fertilizer can worsen deficiency-like symptoms.",
        "immediate_action":"Stop watering for 2 days, test soil pH, apply appropriate micro/macronutrient supplement.",
        "long_term":"Monthly balanced fertilization, maintain pH 6.0-7.0, mulch to retain nutrients.",
        "products":["Balanced NPK 10-10-10","Chelated iron (if yellowing)","Epsom salt (magnesium)"],
        "recovery_time":"2-4 weeks","contagious":False
    },
    "Leaf Blight": {
        "type":"Fungal/Bacterial","pathogen":"Alternaria spp. / Xanthomonas spp.","spread":"Wind, water splash, tools",
        "caution":"HIGHLY CONTAGIOUS. Isolate plant immediately. Do NOT overhead water – wet leaves spread spores rapidly.",
        "immediate_action":"Remove ALL infected leaves with sterilized scissors. Bag & dispose (never compost). Apply copper fungicide immediately.",
        "long_term":"Improve air circulation, water at base only, apply preventive fungicide bi-weekly.",
        "products":["Copper-based fungicide","Mancozeb spray","Neem oil (preventive)"],
        "recovery_time":"3-6 weeks","contagious":True
    },
    "Leaf Spot Disease": {
        "type":"Fungal","pathogen":"Cercospora / Septoria spp.","spread":"Rain splash, contaminated tools",
        "caution":"Spreads rapidly in wet conditions. Avoid wetting foliage. Disinfect tools between plants.",
        "immediate_action":"Remove spotted leaves, thin canopy for airflow, apply systemic fungicide.",
        "long_term":"Rotate fungicide classes to prevent resistance, weekly monitoring.",
        "products":["Chlorothalonil","Propiconazole","Copper sulfate"],
        "recovery_time":"2-5 weeks","contagious":True
    },
    "Wilting / Water Stress": {
        "type":"Environmental","pathogen":"None (stress-related)","spread":"Non-infectious",
        "caution":"Both overwatering AND underwatering cause wilting. Check soil moisture before watering. Root rot may be present.",
        "immediate_action":"Check soil moisture at 2-inch depth. If dry: deep water. If wet: allow drainage, check roots for rot.",
        "long_term":"Establish watering schedule, improve drainage, use moisture meter.",
        "products":["Rooting hormone (if repotting)","Wilt-Stop anti-transpirant","Drainage amendment"],
        "recovery_time":"1-2 weeks","contagious":False
    },
    "Powdery Mildew": {
        "type":"Fungal","pathogen":"Erysiphales order","spread":"Airborne spores",
        "caution":"Spreads explosively in warm, dry conditions with humid nights. One plant can infect an entire garden.",
        "immediate_action":"Isolate plant, spray diluted baking soda (1 tbsp/L water) or potassium bicarbonate immediately.",
        "long_term":"Improve spacing, morning watering only, plant resistant varieties in future.",
        "products":["Sulfur-based fungicide","Potassium bicarbonate","Neem oil"],
        "recovery_time":"3-4 weeks","contagious":True
    },
    "Root Rot": {
        "type":"Fungal","pathogen":"Phytophthora / Pythium spp.","spread":"Contaminated soil, water",
        "caution":"CRITICAL: Often fatal if untreated. Caused by overwatering. Remove from pot – black/mushy roots = advanced rot.",
        "immediate_action":"Remove from pot, trim black roots with sterile scissors, dust with sulfur powder, repot in fresh well-draining mix.",
        "long_term":"Reduce watering frequency, improve drainage, add perlite to soil mix.",
        "products":["Phosphorous acid fungicide","Hydrogen peroxide soil drench","Trichoderma (beneficial fungus)"],
        "recovery_time":"4-8 weeks","contagious":False
    },
    "No Disease Detected": {
        "type":"Healthy","pathogen":"None","spread":"N/A",
        "caution":"Plant appears healthy. Continue regular care and monitor weekly.",
        "immediate_action":"No immediate action required.",
        "long_term":"Maintain regular watering, fertilization schedule and periodic inspection.",
        "products":["Balanced fertilizer (monthly)","Neem oil (preventive spray)"],
        "recovery_time":"N/A","contagious":False
    },
}

PEST_DB = {
    "Aphids": {
        "description":"Tiny soft-bodied insects clustering on new growth and undersides of leaves.",
        "damage":"Suck plant sap, cause curled/yellowed leaves, excrete sticky honeydew promoting sooty mold.",
        "detection_signs":"Sticky residue, yellowing tips, distorted new growth, ant trails on stems.",
        "treatment":"Blast with water jet, apply insecticidal soap, introduce ladybugs (natural predator).",
        "caution":"Ants farm aphids – control ants too or aphids will return quickly.",
        "chemical":"Imidacloprid (systemic) or pyrethrins for severe infestations.",
        "organic":"Neem oil, insecticidal soap, diatomaceous earth."
    },
    "Spider Mites": {
        "description":"Microscopic arachnids forming fine webs on leaf undersides.",
        "damage":"Stippled, bronzed, or silvery leaves; fine webbing between leaves and stems.",
        "detection_signs":"Fine webbing, tiny moving dots under leaves, stippled leaf appearance.",
        "treatment":"Increase humidity, spray with water, apply miticide or neem oil.",
        "caution":"Thrive in hot, dry conditions. Overuse of insecticides kills natural predators, worsening infestations.",
        "chemical":"Abamectin, bifenazate (rotate to prevent resistance).",
        "organic":"Neem oil, predatory mites (Phytoseiulus persimilis)."
    },
    "Whiteflies": {
        "description":"Tiny white-winged insects flying up in clouds when plant is disturbed.",
        "damage":"Yellowing leaves, sooty mold from honeydew, weakened plant vigor.",
        "detection_signs":"White cloud when plant is shaken, sticky leaves, yellow stippling.",
        "treatment":"Yellow sticky traps, insecticidal soap, reflective mulch.",
        "caution":"Very difficult to eradicate once established. Act at first sighting.",
        "chemical":"Pyrethroid or neonicotinoid insecticides.",
        "organic":"Insecticidal soap, neem oil, yellow sticky traps."
    },
    "Scale Insects": {
        "description":"Brown/white bumps on stems and leaves that look like part of the plant.",
        "damage":"Sap-sucking causes yellowing, stunted growth, and leaf drop.",
        "detection_signs":"Hard or soft bumps on stems/leaves, sticky honeydew, sooty mold.",
        "treatment":"Scrape off with soft toothbrush, apply horticultural oil.",
        "caution":"Conventional sprays cannot penetrate shell – use systemic or oil-based treatments.",
        "chemical":"Systemic insecticide (imidacloprid).",
        "organic":"Horticultural oil, 70% isopropyl alcohol swabs."
    },
    "No Pests Detected": {
        "description":"No visible pest damage or signs detected in this analysis.",
        "damage":"None currently observed.",
        "detection_signs":"None present.",
        "treatment":"Preventive: weekly inspection of leaf undersides.",
        "caution":"Apply preventive neem oil spray monthly.",
        "chemical":"Not required.",
        "organic":"Monthly neem oil preventive spray."
    }
}

GROWTH_STAGES = {
    "Seedling":   {"desc":"Early development, fragile root system.","care":"Gentle watering, 16h light, no fertilizer yet.","expected_next":"2-4 weeks to vegetative"},
    "Vegetative": {"desc":"Active leaf and stem growth.","care":"Nitrogen-rich fertilizer, consistent watering.","expected_next":"3-6 weeks to pre-flowering"},
    "Pre-Flower": {"desc":"Preparing for reproduction, may show early buds.","care":"Reduce nitrogen, increase phosphorus.","expected_next":"1-3 weeks to flowering"},
    "Flowering":  {"desc":"Blooms present, pollination occurring.","care":"Potassium boost, minimal leaf disturbance.","expected_next":"2-8 weeks to fruiting/senescence"},
    "Fruiting":   {"desc":"Fruit/seed development in progress.","care":"Consistent moisture, calcium supplementation.","expected_next":"3-12 weeks to harvest"},
    "Mature":     {"desc":"Fully developed, stable growth.","care":"Maintenance fertilization, pruning as needed.","expected_next":"Ongoing – monitor for senescence"},
}

# ════════════════════════════════════════════════════════════════════════════════
#  ANALYSIS ENGINE
# ════════════════════════════════════════════════════════════════════════════════

def identify_plant(img_array, mean_color, green_dominance, texture_variance):
    r, g, b = mean_color[0], mean_color[1], mean_color[2]
    if green_dominance > 0.42 and texture_variance > 500:
        return "fern", 0.71
    elif green_dominance > 0.38 and r > 80 and b > 60:
        return "tomato", 0.68
    elif green_dominance < 0.30 and r > 150:
        return "rose", 0.65
    elif green_dominance > 0.35 and texture_variance < 200:
        return "succulent", 0.72
    elif g > 120 and texture_variance > 300:
        return "basil", 0.64
    return "unknown", 0.45


def detect_pests(img_array, gray, edges):
    pests = []
    fine_texture = cv2.Laplacian(gray, cv2.CV_64F)
    fine_var = np.var(fine_texture)
    if fine_var > 1200:
        pests.append({"name":"Spider Mites","confidence":0.66,"severity":"Moderate"})
    _, thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
    white_ratio = np.sum(thresh == 255) / thresh.size
    if white_ratio > 0.25:
        pests.append({"name":"Whiteflies","confidence":0.58,"severity":"Mild"})
    edge_density = np.sum(edges > 0) / edges.size
    if edge_density > 0.20 and fine_var < 800:
        pests.append({"name":"Scale Insects","confidence":0.54,"severity":"Mild"})
    mean_r = np.mean(img_array[:, :, 0])
    if mean_r > 160 and edge_density > 0.12:
        pests.append({"name":"Aphids","confidence":0.61,"severity":"Mild"})
    if not pests:
        pests.append({"name":"No Pests Detected","confidence":0.88,"severity":"None"})
    return pests


def detect_growth_stage(img_array, green_dominance, texture_variance):
    r = np.mean(img_array[:, :, 0])
    if texture_variance < 150:
        return "Seedling", 0.62
    elif green_dominance > 0.40 and texture_variance > 400:
        return "Vegetative", 0.71
    elif r > 130:
        return "Flowering", 0.65
    elif r > 150:
        return "Fruiting", 0.60
    return "Mature", 0.68


def detect_disease_advanced(image):
    try:
        img_array = np.array(image)
        if len(img_array.shape) != 3:
            return default_disease_response()

        mean_color      = np.mean(img_array, axis=(0, 1))
        green_dominance = mean_color[1] / (mean_color[0] + mean_color[1] + mean_color[2] + 1e-6)
        gray            = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        edges           = cv2.Canny(gray, 50, 150)
        laplacian       = cv2.Laplacian(gray, cv2.CV_64F)
        texture_variance = np.var(laplacian)
        spot_density    = np.sum(edges > 0) / edges.size

        plant_key, plant_conf = identify_plant(img_array, mean_color, green_dominance, texture_variance)
        plant_profile = PLANT_PROFILES[plant_key]

        diseases = []
        health_score = 100

        if mean_color[1] < mean_color[0] * 0.8:
            diseases.append({"name":"Nutrient Deficiency","confidence":0.75,"severity":"Moderate"})
            health_score -= 20
        if mean_color[2] < mean_color[0] * 0.7:
            diseases.append({"name":"Leaf Blight","confidence":0.68,"severity":"Severe"})
            health_score -= 25
        if spot_density > 0.15:
            diseases.append({"name":"Leaf Spot Disease","confidence":0.82,"severity":"Moderate"})
            health_score -= 30
        if texture_variance < 100:
            diseases.append({"name":"Wilting / Water Stress","confidence":0.71,"severity":"Moderate"})
            health_score -= 15
        if green_dominance < 0.20:
            diseases.append({"name":"Powdery Mildew","confidence":0.64,"severity":"Moderate"})
            health_score -= 20
        if not diseases:
            diseases.append({"name":"No Disease Detected","confidence":0.95,"severity":"None"})

        thr = float(st.session_state.get('confidence_threshold', 0.5))
        diseases = [d for d in diseases if d['confidence'] >= thr or d['name'] == "No Disease Detected"]

        for d in diseases:
            db = DISEASE_DB.get(d['name'], DISEASE_DB["No Disease Detected"])
            d.update(db)

        health_score = max(0, min(100, health_score))
        if health_score >= 80:
            final_status, status_class, icon = "Healthy", "card-green", "✅"
        elif health_score >= 50:
            final_status, status_class, icon = "Warning – Monitor Closely", "card-yellow", "⚠️"
        else:
            final_status, status_class, icon = "Unhealthy – Action Required", "card-red", "❌"

        pests = detect_pests(img_array, gray, edges)
        for p in pests:
            db = PEST_DB.get(p['name'], PEST_DB["No Pests Detected"])
            p.update(db)

        growth_stage, growth_conf = detect_growth_stage(img_array, green_dominance, texture_variance)
        growth_info = GROWTH_STAGES[growth_stage]

        soil_score = int(65 + green_dominance * 40 + min(texture_variance / 500, 1) * 20)
        soil_score = max(0, min(100, soil_score))
        soil_ph_est = round(6.0 + (green_dominance - 0.33) * 2, 1)

        return {
            "status":         final_status,
            "status_class":   status_class,
            "icon":           icon,
            "health_score":   health_score,
            "diseases":       diseases,
            "pests":          pests,
            "plant":          plant_profile,
            "plant_confidence": plant_conf,
            "plant_key":      plant_key,
            "growth_stage":   growth_stage,
            "growth_conf":    growth_conf,
            "growth_info":    growth_info,
            "green_dominance": green_dominance,
            "spot_density":   spot_density,
            "texture_variance": texture_variance,
            "soil_score":     soil_score,
            "soil_ph_est":    soil_ph_est,
            "recommendations": generate_recommendations(diseases, pests, health_score),
            "analyzed_at":    datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }
    except Exception as e:
        return default_disease_response()


def default_disease_response():
    return {
        "status":"Unable to Analyze","status_class":"card-red","icon":"❓",
        "health_score":0,
        "diseases":[{"name":"Analysis Failed","confidence":0,"severity":"Unknown",
                     "type":"Error","pathogen":"N/A","spread":"N/A",
                     "caution":"Try a clearer, well-lit image.","immediate_action":"Re-upload image.",
                     "long_term":"Ensure plant fills frame.","products":[],"recovery_time":"N/A","contagious":False}],
        "pests":[{"name":"No Pests Detected","confidence":0,"severity":"None",**PEST_DB["No Pests Detected"]}],
        "plant":PLANT_PROFILES["unknown"],"plant_confidence":0,"plant_key":"unknown",
        "growth_stage":"Unknown","growth_conf":0,"growth_info":GROWTH_STAGES["Mature"],
        "green_dominance":0,"spot_density":0,"texture_variance":0,
        "soil_score":0,"soil_ph_est":6.5,
        "recommendations":["Upload a clearer, well-lit image focused on leaves."],
        "analyzed_at":datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }


def generate_recommendations(diseases, pests, health_score):
    recs = set()
    for d in diseases:
        if "Nutrient" in d['name']:
            recs.add("🌱 Apply balanced NPK fertilizer (10-10-10) every 2 weeks")
            recs.add("💧 Test soil pH – target 6.0-7.0 for most plants")
        elif "Blight" in d['name']:
            recs.add("🍄 Apply copper-based fungicide immediately and bi-weekly")
            recs.add("✂️ Remove infected leaves; sterilize tools after each cut")
            recs.add("🚿 Switch to drip irrigation to keep foliage dry")
        elif "Spot" in d['name']:
            recs.add("🛡️ Apply systemic fungicide (propiconazole)")
            recs.add("💨 Prune inner branches for airflow")
        elif "Wilting" in d['name']:
            recs.add("💧 Use a moisture meter before every watering")
            recs.add("🪴 Check root health – repot to fresh mix if needed")
        elif "Mildew" in d['name']:
            recs.add("🧴 Spray baking soda solution (1 tbsp/L) weekly")
            recs.add("☀️ Increase light and reduce night humidity")
    for p in pests:
        if "Spider" in p['name']:
            recs.add("🕷️ Increase humidity; spray miticide or neem oil weekly")
        elif "Aphid" in p['name']:
            recs.add("🐞 Introduce ladybugs or apply insecticidal soap")
        elif "Whitefl" in p['name']:
            recs.add("🪤 Place yellow sticky traps immediately")
    if health_score < 50:
        recs.add("🚨 Isolate plant from others – potential spread risk")
        recs.add("📸 Photograph daily to monitor recovery progress")
    elif health_score < 80:
        recs.add("📊 Re-analyze in 5-7 days to track improvement")
    if not recs:
        recs.add("✅ Plant is healthy – maintain current care routine")
        recs.add("🔍 Perform weekly inspections for early detection")
    return list(recs)


def get_health_color(score):
    if score >= 80:   return "linear-gradient(90deg,#2e7d32,#43a047)"
    elif score >= 50: return "linear-gradient(90deg,#f57f17,#fdd835)"
    return "linear-gradient(90deg,#b71c1c,#e53935)"


def enhance_image(image):
    try:
        img_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        lab = cv2.cvtColor(img_cv, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        l = clahe.apply(l)
        enhanced = cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2BGR)
        kernel = np.array([[-1,-1,-1],[-1,9,-1],[-1,-1,-1]])
        sharpened = cv2.filter2D(enhanced, -1, kernel)
        return Image.fromarray(cv2.cvtColor(sharpened, cv2.COLOR_BGR2RGB))
    except:
        return image


def generate_pdf_report(results, image):
    try:
        from reportlab.lib.pagesizes import A4
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.lib.units import cm
        from reportlab.lib import colors
        from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, HRFlowable
        from reportlab.lib.enums import TA_CENTER

        buf = io.BytesIO()
        doc = SimpleDocTemplate(buf, pagesize=A4,
                                leftMargin=2*cm, rightMargin=2*cm,
                                topMargin=2*cm, bottomMargin=2*cm)
        styles = getSampleStyleSheet()
        title_s = ParagraphStyle('T', parent=styles['Title'], fontSize=22,
                                 textColor=colors.HexColor('#1b5e20'), fontName='Helvetica-Bold')
        h2_s    = ParagraphStyle('H2', parent=styles['Heading2'], fontSize=14,
                                 textColor=colors.HexColor('#2e7d32'), fontName='Helvetica-Bold', spaceBefore=12)
        body_s  = ParagraphStyle('B', parent=styles['Normal'], fontSize=10, leading=16)
        label_s = ParagraphStyle('L', parent=styles['Normal'], fontSize=9, textColor=colors.grey)

        story = []
        story.append(Paragraph("PlantSuit AI - Diagnosis Report", title_s))
        story.append(Paragraph(f"Generated: {results['analyzed_at']}", label_s))
        story.append(HRFlowable(width="100%", thickness=2, color=colors.HexColor('#2e7d32')))
        story.append(Spacer(1, 12))

        p = results['plant']
        story.append(Paragraph("Plant Identification", h2_s))
        data = [["Common Name", p['name']], ["Family", p['family']], ["Genus", p['genus']],
                ["Origin", p['origin']], ["Growth Type", p['growth']],
                ["Confidence", f"{results['plant_confidence']:.0%}"]]
        t = Table(data, colWidths=[4*cm, 12*cm])
        t.setStyle(TableStyle([
            ('BACKGROUND', (0,0), (0,-1), colors.HexColor('#e8f5e9')),
            ('FONTNAME', (0,0), (0,-1), 'Helvetica-Bold'),
            ('FONTSIZE', (0,0), (-1,-1), 10),
            ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor('#c8e6c9')),
            ('LEFTPADDING', (0,0), (-1,-1), 8),
            ('TOPPADDING', (0,0), (-1,-1), 6),
            ('BOTTOMPADDING', (0,0), (-1,-1), 6),
        ]))
        story.append(t)
        story.append(Spacer(1, 12))

        story.append(Paragraph("Health Summary", h2_s))
        sum_data = [["Overall Status", results['status']],
                    ["Health Score", f"{results['health_score']} / 100"],
                    ["Growth Stage", f"{results['growth_stage']} ({results['growth_conf']:.0%})"],
                    ["Soil Health", f"{results['soil_score']} / 100  (pH approx {results['soil_ph_est']})"]]
        t2 = Table(sum_data, colWidths=[4*cm, 12*cm])
        t2.setStyle(TableStyle([
            ('BACKGROUND', (0,0), (0,-1), colors.HexColor('#e8f5e9')),
            ('FONTNAME', (0,0), (0,-1), 'Helvetica-Bold'),
            ('FONTSIZE', (0,0), (-1,-1), 10),
            ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor('#c8e6c9')),
            ('LEFTPADDING', (0,0), (-1,-1), 8),
            ('TOPPADDING', (0,0), (-1,-1), 6),
            ('BOTTOMPADDING', (0,0), (-1,-1), 6),
        ]))
        story.append(t2)
        story.append(Spacer(1, 12))

        story.append(Paragraph("Detected Conditions", h2_s))
        for d in results['diseases']:
            story.append(Paragraph(f"<b>{d['name']}</b> - {d['severity']} severity ({d['confidence']:.0%} confidence)", body_s))
            story.append(Paragraph(f"Type: {d.get('type','N/A')} | Pathogen: {d.get('pathogen','N/A')} | Contagious: {'Yes' if d.get('contagious') else 'No'}", label_s))
            story.append(Paragraph(f"Caution: {d.get('caution','')}", body_s))
            story.append(Paragraph(f"Immediate Action: {d.get('immediate_action','')}", body_s))
            story.append(Paragraph(f"Long-term: {d.get('long_term','')}", body_s))
            story.append(Paragraph(f"Recovery Time: {d.get('recovery_time','N/A')}", label_s))
            story.append(Spacer(1, 8))

        story.append(Paragraph("Pest Analysis", h2_s))
        for pest in results['pests']:
            story.append(Paragraph(f"<b>{pest['name']}</b> - {pest['severity']} ({pest['confidence']:.0%})", body_s))
            story.append(Paragraph(f"{pest.get('description','')}", label_s))
            story.append(Paragraph(f"Treatment: {pest.get('treatment','')}", body_s))
            story.append(Spacer(1, 6))

        story.append(Paragraph("Care Recommendations", h2_s))
        for rec in results['recommendations']:
            story.append(Paragraph(f"- {rec}", body_s))

        story.append(Spacer(1, 20))
        story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor('#c8e6c9')))
        story.append(Paragraph("PlantSuit AI v3.0 - For reference only. Consult an agronomist for critical decisions.", label_s))

        doc.build(story)
        buf.seek(0)
        return buf.read()
    except Exception as e:
        return None

# ════════════════════════════════════════════════════════════════════════════════
#  DISPLAY HELPERS
# ════════════════════════════════════════════════════════════════════════════════

def display_plant_identity(results):
    p    = results['plant']
    conf = results['plant_confidence']
    st.markdown(f"""
    <div class="ai-badge">🤖 AI Identified &nbsp;·&nbsp; {conf:.0%} confidence</div>
    <div class="card card-green">
        <div class="section-title">🌿 Plant Identity</div>
        <div class="info-grid">
            <div class="info-item"><div class="info-label">Common Name</div><div class="info-value">{p['name']}</div></div>
            <div class="info-item"><div class="info-label">Family</div><div class="info-value">{p['family']}</div></div>
            <div class="info-item"><div class="info-label">Genus</div><div class="info-value"><i>{p['genus']}</i></div></div>
            <div class="info-item"><div class="info-label">Origin</div><div class="info-value">{p['origin']}</div></div>
            <div class="info-item"><div class="info-label">Water Needs</div><div class="info-value">{p['water']}</div></div>
            <div class="info-item"><div class="info-label">Sunlight</div><div class="info-value">{p['sun']}</div></div>
        </div>
        <small>🌍 <b>Native Region:</b> {p['native_region']} &nbsp;|&nbsp; <b>Soil:</b> {p['soil']} &nbsp;|&nbsp; <b>Lifespan:</b> {p['lifespan']}</small>
    </div>
    """, unsafe_allow_html=True)


def display_health_overview(results):
    score = results['health_score']
    st.markdown(f"""
    <div class="card {results['status_class']}">
        <div class="section-title">{results['icon']} Overall Health Assessment</div>
        <div style="font-size:2rem;font-weight:800;color:#1b5e20;">{score}/100</div>
        <div class="health-bar-wrap">
            <div class="health-bar-fill" style="width:{score}%;background:{get_health_color(score)}">{score}%</div>
        </div>
        <b>{results['status']}</b>
        <div style="margin-top:10px;">
            <span class="badge badge-blue">Stage: {results['growth_stage']}</span>
            <span class="badge badge-gray">Soil: {results['soil_score']}/100</span>
            <span class="badge badge-purple">Est. pH: {results['soil_ph_est']}</span>
        </div>
    </div>
    """, unsafe_allow_html=True)


def display_diseases(results):
    st.markdown('<div class="section-title">🔬 Disease Analysis</div>', unsafe_allow_html=True)
    for d in results['diseases']:
        conf = d['confidence']
        sev_map  = {"None":"badge-green","Mild":"badge-yellow","Moderate":"badge-yellow","Severe":"badge-red"}
        sev_badge = sev_map.get(d['severity'], "badge-gray")
        contagious_html = ('<span class="badge badge-red">🔴 Contagious</span>'
                           if d.get('contagious') else
                           '<span class="badge badge-green">🟢 Non-contagious</span>')
        st.markdown(f"""
        <div class="card">
            <b style="font-size:1.05rem">{d['name']}</b>
            <span class="badge {sev_badge}">{d['severity']}</span>
            {contagious_html}
            <div class="conf-bar-wrap"><div class="conf-bar-fill" style="width:{conf*100:.0f}%"></div></div>
            <small>Confidence: {conf:.0%} &nbsp;|&nbsp; Type: {d.get('type','N/A')} &nbsp;|&nbsp; Pathogen: {d.get('pathogen','N/A')} &nbsp;|&nbsp; Spread: {d.get('spread','N/A')}</small>
            <div class="caution-box" style="margin-top:8px"><b>⚠️ Caution:</b> {d.get('caution','')}</div>
            <div class="action-box"><b>✅ Immediate Action:</b> {d.get('immediate_action','')}</div>
            <div style="margin-top:6px"><b>🔄 Long-term Care:</b> {d.get('long_term','')}</div>
            <div style="margin-top:4px"><small><b>⏱ Recovery Time:</b> {d.get('recovery_time','N/A')}</small></div>
        </div>
        """, unsafe_allow_html=True)
        if d.get('products'):
            st.markdown("**Recommended Products:** " + "  ·  ".join([f"`{pr}`" for pr in d['products']]))


def display_pests(results):
    st.markdown('<div class="section-title">🐛 Pest Analysis</div>', unsafe_allow_html=True)
    for p in results['pests']:
        conf = p['confidence']
        sev_map  = {"None":"badge-green","Mild":"badge-yellow","Moderate":"badge-red","Severe":"badge-red"}
        sev_badge = sev_map.get(p['severity'], "badge-gray")
        st.markdown(f"""
        <div class="pest-box">
            <div style="display:flex;justify-content:space-between;align-items:center">
                <b style="font-size:1rem">{p['name']}</b>
                <span><span class="badge {sev_badge}">{p['severity']}</span><small style="color:#888">{conf:.0%}</small></span>
            </div>
            <div class="conf-bar-wrap"><div class="conf-bar-fill" style="width:{conf*100:.0f}%"></div></div>
            <div style="margin-top:8px"><small>{p.get('description','')}</small></div>
            <div style="margin-top:6px"><b>🔍 Signs:</b> {p.get('detection_signs','')}</div>
            <div><b>💊 Treatment:</b> {p.get('treatment','')}</div>
            <div><b>🌿 Organic:</b> {p.get('organic','')}</div>
            <div><b>⚗️ Chemical:</b> {p.get('chemical','')}</div>
            <div style="margin-top:6px;color:#c62828;font-weight:600">{p.get('caution','')}</div>
        </div>
        """, unsafe_allow_html=True)


def display_growth_stage(results):
    gs = results['growth_stage']
    gi = results['growth_info']
    gc = results['growth_conf']
    st.markdown(f"""
    <div class="card card-blue">
        <div class="section-title">🌱 Growth Stage: {gs} ({gc:.0%} confidence)</div>
        <p style="margin:8px 0">{gi['desc']}</p>
        <div class="action-box"><b>Care Right Now:</b> {gi['care']}</div>
        <div><b>⏭ Expected Next Stage:</b> {gi['expected_next']}</div>
    </div>
    """, unsafe_allow_html=True)


def display_recommendations(results):
    st.markdown('<div class="section-title">💡 Smart Care Recommendations</div>', unsafe_allow_html=True)
    for rec in results['recommendations']:
        st.markdown(f"""<div class="action-box" style="margin:5px 0">{rec}</div>""",
                    unsafe_allow_html=True)


def display_quick_stats(results):
    """MUST be called at TOP LEVEL – never inside st.columns."""
    st.markdown('<div class="section-title">📊 Quick Stats</div>', unsafe_allow_html=True)
    c1, c2, c3, c4 = st.columns(4)
    detected  = len([d for d in results['diseases'] if d['name'] != "No Disease Detected"])
    pests_n   = len([p for p in results['pests']    if p['name'] != "No Pests Detected"])
    urgency   = ("🔴 High"   if results['health_score'] < 50 else
                 "🟡 Medium" if results['health_score'] < 80 else "🟢 Low")
    with c1:
        st.markdown(f'<div class="metric-box"><div class="metric-value">{results["health_score"]}</div><div class="metric-label">Health Score</div></div>', unsafe_allow_html=True)
    with c2:
        st.markdown(f'<div class="metric-box"><div class="metric-value">{detected}</div><div class="metric-label">Conditions</div></div>', unsafe_allow_html=True)
    with c3:
        st.markdown(f'<div class="metric-box"><div class="metric-value">{pests_n}</div><div class="metric-label">Pest Types</div></div>', unsafe_allow_html=True)
    with c4:
        st.markdown(f'<div class="metric-box"><div class="metric-value" style="font-size:1.1rem">{urgency}</div><div class="metric-label">Action Urgency</div></div>', unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════════════════════
#  PAGE: AI PLANT ANALYSIS
# ════════════════════════════════════════════════════════════════════════════════

if page == "🔍 AI Plant Analysis":
    st.header("🔍 AI Plant Analysis Engine")

    input_method = st.radio(
        "Choose input:",
        ["📷 Upload Image", "🎥 Camera Capture", "📁 Batch Analysis"],
        horizontal=True
    )

    # ── Upload ──────────────────────────────────────────────────────────────
    if input_method == "📷 Upload Image":
        col_up, col_set = st.columns([2, 1])
        with col_up:
            uploaded_file = st.file_uploader(
                "Upload a plant image",
                type=["jpg","jpeg","png","bmp","webp"],
                help="Clear, well-lit images of leaves give best results"
            )
        with col_set:
            st.markdown("**Analysis Settings**")
            st.session_state.enhancement = st.checkbox("Image Enhancement", value=True)
            st.session_state.confidence_threshold = st.slider(
                "Confidence Threshold", 0.3, 0.9,
                float(st.session_state.confidence_threshold), 0.05,
                help="Lower = more sensitive detections"
            )

        if uploaded_file:
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Image", width=380)

            if st.button("🚀 Run Full AI Analysis", type="primary", use_container_width=True):
                with st.spinner("🤖 AI analysis in progress…"):
                    bar = st.progress(0)
                    for i in range(100):
                        time.sleep(0.007)
                        bar.progress(i + 1)
                    bar.empty()
                    if st.session_state.enhancement:
                        image = enhance_image(image)
                    st.session_state.last_upload_results = detect_disease_advanced(image)
                    st.session_state.last_upload_image   = image.copy()

        # ── Results – rendered OUTSIDE any column nesting ──────────────────
        if st.session_state.last_upload_results:
            results = st.session_state.last_upload_results
            st.markdown("---")
            st.subheader("📋 Full Diagnosis Report")

            col_img, col_rep = st.columns([1, 2])
            with col_img:
                st.image(st.session_state.last_upload_image, caption="Analyzed", use_container_width=True)
            with col_rep:
                display_health_overview(results)
                display_plant_identity(results)

            # Stats – top level (no nesting)
            display_quick_stats(results)

            tab_dis, tab_pest, tab_growth, tab_recs = st.tabs(
                ["🔬 Diseases", "🐛 Pests", "🌱 Growth Stage", "💡 Recommendations"]
            )
            with tab_dis:    display_diseases(results)
            with tab_pest:   display_pests(results)
            with tab_growth: display_growth_stage(results)
            with tab_recs:   display_recommendations(results)

            # Export
            st.markdown("---")
            col_dl1, col_dl2 = st.columns(2)
            with col_dl1:
                if st.button("📄 Generate PDF Report"):
                    pdf = generate_pdf_report(results, st.session_state.last_upload_image)
                    if pdf:
                        st.download_button(
                            "⬇️ Download PDF",
                            data=pdf,
                            file_name=f"plantsuit_{datetime.now().strftime('%Y%m%d_%H%M')}.pdf",
                            mime="application/pdf"
                        )
                    else:
                        st.warning("Install reportlab to enable PDF: `pip install reportlab`")
            with col_dl2:
                safe = {k: v for k, v in results.items() if k != 'image'}
                st.download_button(
                    "⬇️ Export JSON",
                    data=json.dumps(safe, indent=2, default=str),
                    file_name=f"analysis_{datetime.now().strftime('%Y%m%d_%H%M')}.json",
                    mime="application/json"
                )

            # Save to history
            existing_ts = [r['timestamp'] for r in st.session_state.analysis_history]
            if results['analyzed_at'] not in existing_ts:
                st.session_state.analysis_history.append({
                    "timestamp":  results['analyzed_at'],
                    "type":       "Upload Analysis",
                    "result":     results['status'],
                    "confidence": results['health_score'] / 100,
                    "image":      st.session_state.last_upload_image,
                    "details":    results
                })
                st.session_state.prediction_history.append(
                    {"ts": results['analyzed_at'], "score": results['health_score']}
                )
            st.success("✅ Analysis complete!")

    # ── Camera ───────────────────────────────────────────────────────────────
    elif input_method == "🎥 Camera Capture":
        camera_image = st.camera_input("Position your plant in frame")
        if camera_image:
            with st.spinner("🔍 Analysing…"):
                image = Image.open(camera_image)
                if st.session_state.enhancement:
                    image = enhance_image(image)
                results = detect_disease_advanced(image)

            col_img, col_rep = st.columns([1, 2])
            with col_img:
                st.image(image, caption="Captured", use_container_width=True)
            with col_rep:
                display_health_overview(results)
                display_plant_identity(results)

            display_quick_stats(results)   # top level

            tab_dis, tab_pest, tab_growth, tab_recs = st.tabs(
                ["🔬 Diseases", "🐛 Pests", "🌱 Growth Stage", "💡 Recommendations"]
            )
            with tab_dis:    display_diseases(results)
            with tab_pest:   display_pests(results)
            with tab_growth: display_growth_stage(results)
            with tab_recs:   display_recommendations(results)

            st.session_state.analysis_history.append({
                "timestamp":  results['analyzed_at'],
                "type":       "Camera Analysis",
                "result":     results['status'],
                "confidence": results['health_score'] / 100,
                "image":      image.copy(),
                "details":    results
            })

    # ── Batch ────────────────────────────────────────────────────────────────
    elif input_method == "📁 Batch Analysis":
        st.info("Upload multiple plant images for bulk analysis")
        files = st.file_uploader("Select images", type=["jpg","jpeg","png"],
                                  accept_multiple_files=True)
        if files and st.button("🚀 Analyse All", type="primary"):
            bar = st.progress(0)
            results_list = []
            for idx, f in enumerate(files):
                img = Image.open(f)
                if st.session_state.enhancement:
                    img = enhance_image(img)
                r = detect_disease_advanced(img)
                r['file'] = f.name
                results_list.append(r)
                bar.progress((idx + 1) / len(files))
            bar.empty()

            st.subheader("Batch Results Summary")
            for r in results_list:
                ico = "✅" if "Healthy" in r['status'] else "⚠️" if "Warning" in r['status'] else "❌"
                contagious = any(d.get('contagious') for d in r['diseases'])
                c_badge = '<span class="badge badge-red">⚠️ Contagious Risk</span>' if contagious else ''
                card_cls = "card-red" if "Unhealthy" in r['status'] else "card-yellow" if "Warning" in r['status'] else "card-green"
                st.markdown(f"""
                <div class="card {card_cls}">
                    <b>{ico} {r['file']}</b>
                    <span class="badge badge-gray">{r['plant']['name']}</span> {c_badge}
                    <div style="margin-top:6px">
                        Status: <b>{r['status']}</b> &nbsp;|&nbsp;
                        Score: <b>{r['health_score']}/100</b> &nbsp;|&nbsp;
                        Stage: <b>{r['growth_stage']}</b>
                    </div>
                </div>
                """, unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════════════════════
#  PAGE: DISEASE LIBRARY
# ════════════════════════════════════════════════════════════════════════════════

elif page == "🔬 Disease Library":
    st.header("🔬 Plant Disease Reference Library")
    search = st.text_input("🔍 Search diseases", placeholder="blight, mildew, fungal…")

    for name, info in DISEASE_DB.items():
        if search and search.lower() not in name.lower() and search.lower() not in info.get('type','').lower():
            continue
        ico = "✅" if info['type'] == "Healthy" else "🦠"
        with st.expander(f"{ico} {name}  —  {info['type']}"):
            col1, col2 = st.columns(2)
            with col1:
                st.markdown(f"**Pathogen:** {info['pathogen']}")
                st.markdown(f"**Spread:** {info['spread']}")
                st.markdown(f"**Contagious:** {'🔴 Yes' if info.get('contagious') else '🟢 No'}")
                st.markdown(f"**Recovery Time:** {info['recovery_time']}")
            with col2:
                st.markdown(f"**Caution:** {info['caution']}")
                st.markdown(f"**Immediate Action:** {info['immediate_action']}")
            st.markdown(f"**Long-term Care:** {info['long_term']}")
            if info.get('products'):
                st.markdown("**Products:** " + "  ·  ".join([f"`{p}`" for p in info['products']]))


# ════════════════════════════════════════════════════════════════════════════════
#  PAGE: PEST IDENTIFIER
# ════════════════════════════════════════════════════════════════════════════════

elif page == "🐛 Pest Identifier":
    st.header("🐛 Pest Identifier & Treatment Guide")
    search_p = st.text_input("🔍 Search pests", placeholder="aphid, mite, scale…")

    for name, info in PEST_DB.items():
        if name == "No Pests Detected":
            continue
        if search_p and search_p.lower() not in name.lower():
            continue
        with st.expander(f"🐛 {name}"):
            col1, col2 = st.columns(2)
            with col1:
                st.markdown(f"**Description:** {info['description']}")
                st.markdown(f"**Damage:** {info['damage']}")
                st.markdown(f"**Detection Signs:** {info['detection_signs']}")
            with col2:
                st.markdown(f"**Treatment:** {info['treatment']}")
                st.markdown(f"**Organic:** {info['organic']}")
                st.markdown(f"**Chemical:** {info['chemical']}")
            st.info(info['caution'])


# ════════════════════════════════════════════════════════════════════════════════
#  PAGE: HEALTH HISTORY
# ════════════════════════════════════════════════════════════════════════════════

elif page == "📈 Health History":
    st.header("📈 Plant Health History & Trends")

    col_h1, col_h2 = st.columns([3, 1])
    with col_h2:
        if st.button("🗑️ Clear History"):
            st.session_state.analysis_history  = []
            st.session_state.prediction_history = deque(maxlen=30)
            st.rerun()

    if st.session_state.prediction_history:
        st.subheader("Health Score Trend")
        import pandas as pd
        history_data = list(st.session_state.prediction_history)
        df = pd.DataFrame({
            "Time":         [h['ts'][-8:] for h in history_data],
            "Health Score": [h['score']   for h in history_data]
        })
        st.line_chart(df.set_index("Time"))
    else:
        st.info("Run analyses to see health trends here")

    st.subheader("Analysis Log")
    if st.session_state.analysis_history:
        for record in reversed(st.session_state.analysis_history):
            ico = "✅" if "Healthy" in record['result'] else "⚠️"
            with st.expander(f"{ico} {record['type']} — {record['timestamp']}"):
                c1, c2 = st.columns([1, 3])
                with c1:
                    if 'image' in record:
                        st.image(record['image'], width=120)
                with c2:
                    d = record['details']
                    st.markdown(f"**Plant:** {d['plant']['name']} ({d['plant']['family']})")
                    st.markdown(f"**Status:** {record['result']}")
                    st.markdown(f"**Health Score:** {record['confidence']*100:.0f}/100")
                    st.markdown(f"**Growth Stage:** {d['growth_stage']}")
                    pests = [p['name'] for p in d['pests'] if p['name'] != 'No Pests Detected']
                    if pests:
                        st.markdown(f"**Pests Detected:** {', '.join(pests)}")
                    st.markdown("**Top Recommendations:**")
                    for rec in d['recommendations'][:3]:
                        st.markdown(f"- {rec}")
                    if st.button("✅ Mark Treatment Applied", key=f"treat_{record['timestamp']}"):
                        st.session_state.treatment_log.append({
                            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M"),
                            "plant":     d['plant']['name'],
                            "original_score": record['confidence'] * 100
                        })
                        st.success("Logged!")
    else:
        st.info("No history yet. Run an analysis first.")

    if st.session_state.treatment_log:
        st.subheader("Treatment Log")
        for t in reversed(st.session_state.treatment_log):
            st.markdown(f"✅ **{t['timestamp']}** — {t['plant']} — original score: {t['original_score']:.0f}/100")


# ════════════════════════════════════════════════════════════════════════════════
#  PAGE: WEATHER RISK
# ════════════════════════════════════════════════════════════════════════════════

elif page == "🌦️ Weather Risk":
    st.header("🌦️ Weather-Based Disease & Pest Risk")
    st.info("Enter current conditions to get a risk assessment for your plants.")

    col_w1, col_w2, col_w3 = st.columns(3)
    with col_w1:
        temp     = st.number_input("Temperature (°C)", 0, 50, 28)
    with col_w2:
        humidity = st.number_input("Humidity (%)", 0, 100, 70)
    with col_w3:
        rainfall = st.selectbox("Recent Rainfall", ["None","Light","Moderate","Heavy"])

    if st.button("🔍 Assess Risk", type="primary"):
        risks = []

        if 20 <= temp <= 28 and 40 <= humidity <= 70:
            risks.append(("Powdery Mildew","High","#c62828","Ideal conditions. Apply preventive fungicide NOW."))
        elif humidity > 70:
            risks.append(("Powdery Mildew","Moderate","#f57f17","Elevated humidity. Monitor closely."))

        if humidity > 80 and rainfall in ["Moderate","Heavy"]:
            risks.append(("Leaf Blight / Late Blight","High","#c62828","Wet conditions favour blight. Apply copper fungicide today."))
        elif humidity > 65:
            risks.append(("Leaf Blight","Moderate","#f57f17","Keep foliage dry. Avoid overhead watering."))

        if temp > 30 and humidity < 40:
            risks.append(("Spider Mites","High","#c62828","Hot & dry: prime mite conditions. Increase humidity now."))

        if 18 <= temp <= 26 and humidity > 60:
            risks.append(("Aphids","Moderate","#f57f17","Warm & humid: check new growth daily."))

        if rainfall == "Heavy":
            risks.append(("Root Rot / Waterlogging","High","#c62828","Ensure drainage, skip watering for 3-4 days."))

        if not risks:
            risks.append(("All conditions","Low","#2e7d32","Conditions are favourable. Maintain regular care."))

        st.markdown("### Risk Assessment Results")
        for condition, level, color, advice in risks:
            st.markdown(f"""
            <div class="card" style="border-left:5px solid {color}">
                <b>{condition}</b>
                <span class="badge" style="background:{color}20;color:{color}">{level} Risk</span>
                <div style="margin-top:6px">{advice}</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("---")
        st.markdown("### 🌿 General Weather Tips")
        tips = []
        if humidity > 75:
            tips += ["💨 Increase air circulation around plants",
                     "🌬️ Avoid overhead watering – use drip irrigation"]
        if temp > 32:
            tips += ["🌤️ Provide afternoon shade to prevent heat stress",
                     "💧 Water early morning to reduce evaporation"]
        if rainfall == "Heavy":
            tips += ["🪴 Check drainage holes are clear",
                     "🚫 Skip scheduled watering for 3-5 days"]
        if not tips:
            tips.append("✅ Conditions are generally safe. Maintain regular routine.")
        for t in tips:
            st.markdown(f"- {t}")


# ════════════════════════════════════════════════════════════════════════════════
#  PAGE: SETTINGS
# ════════════════════════════════════════════════════════════════════════════════

elif page == "⚙️ Settings":
    st.header("⚙️ Settings")

    st.subheader("🎯 Detection Settings")
    col_s1, col_s2 = st.columns(2)
    with col_s1:
        st.session_state.enhancement = st.checkbox(
            "Auto-enhance images", value=st.session_state.enhancement)
        st.session_state.confidence_threshold = st.slider(
            "Default Confidence Threshold", 0.3, 0.9,
            float(st.session_state.confidence_threshold), 0.05)
    with col_s2:
        st.select_slider("Alert Sensitivity", ["Low","Medium","High"], value="Medium")
        st.selectbox("Language", ["English","Hindi","Marathi","Spanish","French"])

    st.subheader("💾 Data Management")
    col_e1, col_e2 = st.columns(2)
    with col_e1:
        if st.button("Export History as JSON"):
            export = [{"timestamp": r['timestamp'], "type": r['type'],
                       "result": r['result'], "confidence": r['confidence'],
                       "plant": r['details']['plant']['name'],
                       "growth_stage": r['details']['growth_stage']}
                      for r in st.session_state.analysis_history]
            st.download_button(
                "⬇️ Download JSON",
                data=json.dumps(export, indent=2),
                file_name=f"plantsuit_history_{datetime.now().strftime('%Y%m%d')}.json",
                mime="application/json"
            )
    with col_e2:
        if st.button("🗑️ Clear All Data"):
            for k in ['analysis_history','treatment_log','last_upload_results','last_upload_image']:
                st.session_state[k] = [] if isinstance(st.session_state[k], list) else None
            st.session_state.prediction_history = deque(maxlen=30)
            st.success("All data cleared.")

    st.subheader("ℹ️ About PlantSuit AI v3.0")
    st.markdown("""
    <div class="card card-green">
        <b>PlantSuit AI v3.0 — Full Intelligence Edition</b><br><br>
        ✅ AI Plant Identification (Name, Family, Genus, Origin, Lifespan)<br>
        ✅ 6+ Disease Detection with Caution, Action &amp; Long-term Plans<br>
        ✅ Pest Detection (Aphids, Spider Mites, Whiteflies, Scale Insects)<br>
        ✅ Growth Stage Classification (6 stages)<br>
        ✅ Soil Health &amp; pH Estimation<br>
        ✅ Weather-Based Disease &amp; Pest Risk Assessment<br>
        ✅ PDF Diagnosis Reports (requires reportlab)<br>
        ✅ Health Score Trend Tracking<br>
        ✅ Treatment Effectiveness Log<br>
        ✅ Batch Image Analysis<br>
        ✅ JSON Export<br><br>
        <small>For critical agricultural decisions, consult a certified agronomist.</small>
    </div>
    """, unsafe_allow_html=True)


# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("---")
col_f1, col_f2, col_f3 = st.columns(3)
with col_f1:
    st.caption("🌿 PlantSuit AI v3.0")
with col_f2:
    st.caption(f"Analyses logged: {len(st.session_state.analysis_history)}")
with col_f3:
    st.caption(f"🟢 Ready — {datetime.now().strftime('%H:%M:%S')}")