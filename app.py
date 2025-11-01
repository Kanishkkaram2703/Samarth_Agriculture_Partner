from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import requests
import os
import re
from dotenv import load_dotenv
from openai import OpenAI
import pandas as pd
from functools import lru_cache
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

app = Flask(__name__)
CORS(app)

# Load environment variables
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
SEARCH_ENGINE_ID = os.getenv("SEARCH_ENGINE_ID")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not all([GOOGLE_API_KEY, SEARCH_ENGINE_ID, OPENAI_API_KEY]):
    logger.warning("Some API keys are missing")

client = OpenAI(api_key=OPENAI_API_KEY)

# Load datasets with error handling
def load_datasets():
    """Load CSV datasets with proper error handling"""
    try:
        base_paths = [
            Path(__file__).parent / 'datasets',
            Path('datasets'),
        ]
        
        rainfall_df = None
        agriculture_df = None
        
        for base_path in base_paths:
            rainfall_path = base_path / 'rainfall_IMD.csv'
            agriculture_path = base_path / 'agriculture_gujarat.csv'
            
            if rainfall_path.exists():
                rainfall_df = pd.read_csv(rainfall_path)
                # Clean column names
                rainfall_df.columns = rainfall_df.columns.str.strip()
                logger.info(f"✓ Loaded rainfall data: {len(rainfall_df)} rows")
                
            if agriculture_path.exists():
                agriculture_df = pd.read_csv(agriculture_path)
                # Clean column names
                agriculture_df.columns = agriculture_df.columns.str.strip()
                logger.info(f"✓ Loaded agriculture data: {len(agriculture_df)} rows")
                
            if rainfall_df is not None and agriculture_df is not None:
                break
        
        if rainfall_df is None or agriculture_df is None:
            logger.warning("Using fallback sample data")
            rainfall_df = pd.DataFrame({
                'Year': [2020, 2021, 2022, 2023, 2024],
                'Actual': [111, 100, 106, 95, 108],
                'Forecast': [102, 98, 99, 96, 106],
                'Remark': ['Outside the forecast error limit', 'Accurate', 'Outside the forecast error limit', 'Accurate', 'Accurate']
            })
            
            agriculture_df = pd.DataFrame({
                'Crops': ['Groundnut', 'Castor seed', 'Sesamum', 'Rapeseed & Mustard', 'Soyabean', 'Other Oilseeds', 'Total Oil Seeds'],
                'Area': [1452.0, 589.0, 195.0, 823.0, 765.0, 144.0, 3968.0],
                'Production': [3568.9, 1305.7, 195.3, 1421.5, 724.0, 153.2, 7368.6],
                'Yield': [2458, 2217, 1002, 1727, 946, 1064, 1857]
            })
        
        return rainfall_df, agriculture_df
        
    except Exception as e:
        logger.error(f"Error loading datasets: {e}")
        return None, None

rainfall_df, agriculture_df = load_datasets()

# Enhanced Google Search
@lru_cache(maxsize=50)
def cached_google_search(query):
    """Google search with caching"""
    if not GOOGLE_API_KEY or not SEARCH_ENGINE_ID:
        logger.warning("Google Search API not configured")
        return []
    
    try:
        url = (
            f"https://www.googleapis.com/customsearch/v1?"
            f"q={requests.utils.quote(query)}&key={GOOGLE_API_KEY}&cx={SEARCH_ENGINE_ID}"
            f"&gl=in&cr=countryIN&num=5"
        )
        response = requests.get(url, timeout=5)
        response.raise_for_status()
        
        data = response.json()
        items = data.get("items", [])
        
        results = []
        for item in items[:5]:
            results.append({
                'title': item.get('title', ''),
                'link': item.get('link', ''),
                'snippet': item.get('snippet', '')
            })
        
        logger.info(f"✓ Google Search: {len(results)} results for '{query[:50]}'")
        return results
        
    except Exception as e:
        logger.error(f"Google Search Error: {e}")
        return []

# Enhanced GPT Function
def ask_openai(question, context, max_retries=2):
    """Generate answer using OpenAI GPT with retry logic"""
    if not client:
        return "OpenAI API is not available. Please configure OPENAI_API_KEY."
    
    for attempt in range(max_retries):
        try:
            # Improved prompt
            prompt = f"""You are Samarth, an expert AI assistant for Indian agriculture and climate.

Question: {question}

Available Data/Context:
{context}

Instructions:
- Provide a clear, accurate answer in 2-4 sentences
- Use the provided data/context to support your answer
- Be specific with numbers and facts
- If data is insufficient, provide general agricultural knowledge
- Keep the tone helpful and informative

Answer:"""

            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are an expert in Indian agriculture, climate data, and sustainable farming practices. Provide accurate, concise answers."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.4,
                max_tokens=250
            )
            
            answer = response.choices[0].message.content.strip()
            logger.info(f"✓ OpenAI response generated")
            return answer
            
        except Exception as e:
            logger.error(f"OpenAI Error (attempt {attempt+1}): {e}")
            if attempt == max_retries - 1:
                return "I apologize, but I'm experiencing technical difficulties with AI processing. Please try asking your question differently."
    
    return "Unable to generate response at this time."

# Enhanced Rainfall Query Handler
def handle_rainfall_query(query):
    """Handle rainfall and monsoon related queries with better parsing"""
    if rainfall_df is None:
        return None
        
    query_lower = query.lower()
    
    if not ('rainfall' in query_lower or 'monsoon' in query_lower or 'rain' in query_lower or 'imd' in query_lower):
        return None
    
    try:
        # Extract year with improved regex
        year_pattern = r'\b(202[0-4])\b'
        year_match = re.search(year_pattern, query)
        year = int(year_match.group(1)) if year_match else None
        
        # Specific year query
        if year and year in rainfall_df['Year'].values:
            row = rainfall_df[rainfall_df['Year'] == year].iloc[0]
            
            return {
                'answer': f"In {year}, India's monsoon rainfall was {row['Actual']}% of the long period average (LPA), with a forecast of {row['Forecast']}%. Status: {row['Remark']}.",
                'sources': ['IMD Dataset (2020-2024)'],
                'data_type': 'dataset'
            }
        
        # Trend/pattern query
        if any(word in query_lower for word in ['trend', 'pattern', 'compare', 'all years', 'over time']):
            years = rainfall_df['Year'].tolist()
            actuals = rainfall_df['Actual'].tolist()
            avg = sum(actuals) / len(actuals)
            
            trend_details = ', '.join([f"{y}: {a}%" for y, a in zip(years, actuals)])
            
            return {
                'answer': f"Rainfall trends (2020-2024): {trend_details}. Average: {avg:.1f}% of LPA. Years 2020 and 2024 showed above-normal rainfall, while 2023 was slightly below average.",
                'sources': ['IMD Dataset Analysis'],
                'data_type': 'dataset'
            }
        
        # Highest/lowest queries
        if 'highest' in query_lower or 'maximum' in query_lower:
            max_row = rainfall_df.loc[rainfall_df['Actual'].idxmax()]
            return {
                'answer': f"The highest rainfall was in {int(max_row['Year'])} at {max_row['Actual']}% of LPA, which was {max_row['Remark']}.",
                'sources': ['IMD Dataset (2020-2024)'],
                'data_type': 'dataset'
            }
        
        if 'lowest' in query_lower or 'minimum' in query_lower:
            min_row = rainfall_df.loc[rainfall_df['Actual'].idxmin()]
            return {
                'answer': f"The lowest rainfall was in {int(min_row['Year'])} at {min_row['Actual']}% of LPA, which was {min_row['Remark']}.",
                'sources': ['IMD Dataset (2020-2024)'],
                'data_type': 'dataset'
            }
        
        # Forecast accuracy
        if 'forecast' in query_lower or 'accuracy' in query_lower or 'accurate' in query_lower:
            accurate_years = rainfall_df[rainfall_df['Remark'] == 'Accurate']['Year'].tolist()
            if accurate_years:
                return {
                    'answer': f"IMD forecasts were marked as 'Accurate' in years: {', '.join(map(str, accurate_years))}. These years had forecast errors within acceptable limits.",
                    'sources': ['IMD Dataset (2020-2024)'],
                    'data_type': 'dataset'
                }
        
        # Latest data fallback
        latest_year = rainfall_df['Year'].max()
        latest = rainfall_df[rainfall_df['Year'] == latest_year].iloc[0]
        return {
            'answer': f"Most recent data ({int(latest_year)}): India's monsoon rainfall was {latest['Actual']}% of LPA, forecast was {latest['Forecast']}%. Status: {latest['Remark']}.",
            'sources': ['IMD Dataset (2020-2024)'],
            'data_type': 'dataset'
        }
        
    except Exception as e:
        logger.error(f"Error in handle_rainfall_query: {e}")
        return None

# Enhanced Agriculture Query Handler
def handle_agriculture_query(query):
    """Handle crop and agriculture queries with better parsing"""
    if agriculture_df is None:
        return None
        
    query_lower = query.lower()
    
    try:
        # Highest yield
        if 'highest yield' in query_lower or 'maximum yield' in query_lower:
            max_row = agriculture_df.loc[agriculture_df['Yield'].idxmax()]
            return {
                'answer': f"The crop with the highest yield in Gujarat is {max_row['Crops']} at {max_row['Yield']} kg/ha, with production of {max_row['Production']:.1f} thousand tonnes over {max_row['Area']:.0f} thousand hectares.",
                'sources': ['Gujarat Agriculture Department Dataset'],
                'data_type': 'dataset'
            }
        
        # Rank crops by yield
        if 'rank' in query_lower and 'yield' in query_lower:
            sorted_df = agriculture_df.sort_values('Yield', ascending=False)
            rankings = ', '.join([f"{row['Crops']}: {row['Yield']} kg/ha" for _, row in sorted_df.iterrows()])
            return {
                'answer': f"Crops ranked by yield (descending): {rankings}.",
                'sources': ['Gujarat Agriculture Department Dataset'],
                'data_type': 'dataset'
            }
        
        # Total oilseeds
        if 'total oilseed' in query_lower or 'total oil seed' in query_lower:
            oil_row = agriculture_df[agriculture_df['Crops'].str.contains('Total Oil Seeds', case=False, na=False)]
            if not oil_row.empty:
                oil_row = oil_row.iloc[0]
                return {
                    'answer': f"Total oilseeds production in Gujarat is {oil_row['Production']:.2f} thousand tonnes, with average yield of {oil_row['Yield']} kg/ha across {oil_row['Area']:.0f} thousand hectares.",
                    'sources': ['Gujarat Agriculture Department Dataset'],
                    'data_type': 'dataset'
                }
        
        # Compare two crops
        if 'compare' in query_lower:
            # Extract crop names from query
            crops_found = []
            for _, row in agriculture_df.iterrows():
                crop_name = row['Crops'].lower()
                if crop_name in query_lower and 'total' not in crop_name:
                    crops_found.append(row['Crops'])
            
            if len(crops_found) >= 2:
                crop1_data = agriculture_df[agriculture_df['Crops'] == crops_found[0]].iloc[0]
                crop2_data = agriculture_df[agriculture_df['Crops'] == crops_found[1]].iloc[0]
                
                return {
                    'answer': f"Comparison: {crops_found[0]} - Area: {crop1_data['Area']:.0f}k ha, Production: {crop1_data['Production']:.1f}k tonnes, Yield: {crop1_data['Yield']} kg/ha. {crops_found[1]} - Area: {crop2_data['Area']:.0f}k ha, Production: {crop2_data['Production']:.1f}k tonnes, Yield: {crop2_data['Yield']} kg/ha.",
                    'sources': ['Gujarat Agriculture Department Dataset'],
                    'data_type': 'dataset'
                }
        
        # Specific crop query
        for _, row in agriculture_df.iterrows():
            crop_name = row['Crops'].lower()
            # Match crop name in query
            if crop_name in query_lower or any(word in query_lower for word in crop_name.split()):
                return {
                    'answer': f"{row['Crops']} in Gujarat: Area under cultivation is {row['Area']:.0f} thousand hectares, production is {row['Production']:.1f} thousand tonnes, and yield is {row['Yield']} kg/ha.",
                    'sources': ['Gujarat Agriculture Department Dataset'],
                    'data_type': 'dataset'
                }
        
        # List all crops
        if any(word in query_lower for word in ['all crop', 'list crop', 'which crop', 'major crop', 'season']):
            crops_list = [crop for crop in agriculture_df['Crops'].tolist() if 'Total' not in crop]
            return {
                'answer': f"Major oilseed crops in Gujarat: {', '.join(crops_list)}. You can ask about specific crops, yields, or production details.",
                'sources': ['Gujarat Agriculture Department Dataset'],
                'data_type': 'dataset'
            }
            
    except Exception as e:
        logger.error(f"Error in handle_agriculture_query: {e}")
        return None

# Enhanced Fallback Function
def fallback_to_google_gpt(question):
    """Fallback to Google Search + GPT for conceptual queries"""
    try:
        # Search Google
        search_results = cached_google_search(question + " India agriculture")
        
        # Build rich context
        context = ""
        sources = []
        
        if search_results:
            for result in search_results[:3]:
                context += f"Source: {result['title']}\n{result['snippet']}\n\n"
                sources.append(result['link'])
        
        # Add general knowledge context
        base_context = """
        General Indian Agriculture Knowledge:
        - India has three crop seasons: Kharif (June-Oct): Rice, Cotton, Soybean; Rabi (Oct-March): Wheat, Mustard, Gram; Zaid (March-June): Summer vegetables
        - Sustainable practices: Crop rotation, organic farming, drip irrigation, IPM, zero-tillage, bio-fertilizers
        - Gujarat is a major producer of cotton, groundnut, and oilseeds
        - Monsoon rainfall (June-September) is critical for Kharif crops
        - Precision farming uses GPS, sensors, and data analytics for efficient farming
        """
        
        full_context = context + base_context if context else base_context
        
        # Generate answer with GPT
        answer = ask_openai(question, full_context)
        
        # If GPT fails, provide manual fallback
        if "technical difficulties" in answer.lower() or "unable to generate" in answer.lower():
            answer = provide_manual_fallback(question)
        
        return {
            'answer': answer,
            'sources': sources if sources else ['Agricultural Knowledge Base', 'Ministry of Agriculture & Farmers Welfare'],
            'data_type': 'web_search'
        }
        
    except Exception as e:
        logger.error(f"Error in fallback: {e}")
        return {
            'answer': provide_manual_fallback(question),
            'sources': ['Agricultural Knowledge Base'],
            'data_type': 'knowledge'
        }

def provide_manual_fallback(question):
    """Provide manual answers for common questions when APIs fail"""
    q = question.lower()
    
    if 'sustainable' in q or 'technique' in q or 'practice' in q:
        return "Top sustainable agriculture techniques in India: 1) Crop rotation to improve soil health, 2) Organic farming with compost and bio-fertilizers, 3) Drip and sprinkler irrigation for water conservation, 4) Integrated Pest Management (IPM), 5) Zero-tillage farming to reduce soil erosion. These practices enhance productivity while protecting the environment."
    
    if 'season' in q or 'kharif' in q or 'rabi' in q:
        return "India has three main cropping seasons: Kharif (Monsoon, June-October) includes rice, maize, cotton, soybean, groundnut; Rabi (Winter, October-March) includes wheat, barley, mustard, gram, peas; Zaid (Summer, March-June) includes watermelon, cucumber, fodder crops. Each season depends on specific climate conditions."
    
    if 'precision' in q or 'ai' in q or 'iot' in q or 'technology' in q:
        return "Precision farming uses GPS, IoT sensors, drones, and AI to optimize crop management. Benefits include: efficient water use, precise fertilizer application, early pest detection, and data-driven decisions. In India, it can increase yields by 10-30% while reducing input costs."
    
    if 'climate' in q and ('data' in q or 'forecast' in q or 'analytic' in q):
        return "Data analytics in climate forecasting uses historical weather data, satellite imagery, and machine learning models to predict rainfall, temperature patterns, and extreme events. This helps farmers plan sowing dates, choose appropriate crops, and manage risks better."
    
    if 'policy' in q or 'government' in q or '2030' in q:
        return "Policy recommendations for sustainable agriculture: 1) Increase investment in irrigation infrastructure, 2) Provide subsidies for organic inputs and drip systems, 3) Strengthen crop insurance schemes, 4) Promote farmer training on modern techniques, 5) Improve market linkages and MSP implementation. These can help achieve food security and sustainability goals."
    
    return "I can provide insights on Indian agriculture, climate data, crop yields, sustainable practices, and farming techniques. Please ask specific questions about rainfall patterns, crop production, or agricultural methods."

# Routes
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/health")
def health():
    return jsonify({
        'status': 'healthy',
        'datasets_loaded': rainfall_df is not None and agriculture_df is not None,
        'apis_configured': all([GOOGLE_API_KEY, SEARCH_ENGINE_ID, OPENAI_API_KEY]),
        'rainfall_years': rainfall_df['Year'].tolist() if rainfall_df is not None else [],
        'crops_count': len(agriculture_df) if agriculture_df is not None else 0
    })

@app.route("/ask", methods=["POST"])
def ask():
    try:
        data = request.get_json()
        question = data.get("question", "").strip()
        
        if not question:
            return jsonify({
                'answer': 'Please ask a question!',
                'sources': [],
                'data_type': 'error'
            }), 400
        
        logger.info(f"📝 Query: {question}")
        
        # Try rainfall dataset first
        result = handle_rainfall_query(question)
        if result:
            logger.info("✓ Answered from rainfall dataset")
            return jsonify(result)
        
        # Try agriculture dataset
        result = handle_agriculture_query(question)
        if result:
            logger.info("✓ Answered from agriculture dataset")
            return jsonify(result)
        
        # Fallback to Google + GPT
        logger.info("→ Using Google Search + GPT fallback")
        result = fallback_to_google_gpt(question)
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"❌ Error in /ask: {e}")
        return jsonify({
            'answer': 'An error occurred. Please try rephrasing your question.',
            'sources': [],
            'data_type': 'error'
        }), 500

@app.errorhandler(404)
def not_found(e):
    return jsonify({'error': 'Endpoint not found'}), 404

@app.errorhandler(500)
def server_error(e):
    return jsonify({'error': 'Internal server error'}), 500

if __name__ == "__main__":
    print("\n" + "="*50)
    print("🌾 Project Samarth - Starting Server")
    print("="*50)
    print(f"✓ Rainfall data: {len(rainfall_df) if rainfall_df is not None else 0} rows")
    print(f"✓ Agriculture data: {len(agriculture_df) if agriculture_df is not None else 0} rows")
    print(f"✓ Google API: {'Configured' if GOOGLE_API_KEY else 'Not configured'}")
    print(f"✓ OpenAI API: {'Configured' if OPENAI_API_KEY else 'Not configured'}")
    print("="*50)
    print("🚀 Server running at: http://127.0.0.1:5000")
    print("="*50 + "\n")
    

    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
