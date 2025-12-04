import os
import uvicorn
import asyncio
import logging
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
try:
    from serpapi import GoogleSearch
except Exception as e:
    raise ImportError(
        "Could not import 'GoogleSearch' from 'serpapi'. Please install or upgrade the official SerpAPI Python package:\n"
        "    pip install --upgrade google-search-results\n"
        f"Original import error: {e}"
    )
from crewai import Agent, Task, Crew, Process, LLM
from datetime import datetime
from functools import lru_cache
from dotenv import load_dotenv

# Load .env (if present) and API Keys
load_dotenv()
GEMINI_API_KEY = os.getenv("GOOGLE_API_KEY", "your_gemini_api_key_here")
SERP_API_KEY = os.getenv("SERP_API_KEY", "your_serpapi_key_here")

# Initialize Logger
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


# ==============================================
# 🤖 Initialize Google Gemini AI (LLM)
# ==============================================
@lru_cache(maxsize=1)
def initialize_llm():
    """Initialize and cache the LLM instance to avoid repeated initializations."""
    return LLM(
        model="gemini/gemini-2.5-flash",
        provider="google",
        api_key=GEMINI_API_KEY
    )

# ==============================================
# 📝 Pydantic Models
# ==============================================
class FlightRequest(BaseModel):
    origin: str
    destination: str
    outbound_date: str
    return_date: str


class HotelRequest(BaseModel):
    location: str
    check_in_date: str
    check_out_date: str


class ItineraryRequest(BaseModel):
    destination: str
    check_in_date: str
    check_out_date: str
    flights: str
    hotels: str


class FlightInfo(BaseModel):
    airline: str
    price: str
    duration: str
    stops: str
    departure: str
    arrival: str
    travel_class: str
    return_date: str
    airline_logo: str


class HotelInfo(BaseModel):
    name: str
    price: str
    rating: float
    location: str
    link: str


class AIResponse(BaseModel):
    flights: List[FlightInfo] = []
    hotels: List[HotelInfo] = []
    ai_flight_recommendation: str = ""
    ai_hotel_recommendation: str = ""
    itinerary: str = ""


# ==============================================
# 🚀 Initialize FastAPI
# ==============================================
app = FastAPI(title="Travel Planning API", version="1.1.0")


# ==============================================
# 🛫 Fetch Data from SerpAPI
# ==============================================
async def run_search(params):
    """Generic function to run SerpAPI searches asynchronously."""
    try:
        return await asyncio.to_thread(lambda: GoogleSearch(params).get_dict())
    except Exception as e:
        logger.exception(f"SerpAPI search error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Search API error: {str(e)}")


async def search_flights(flight_request: FlightRequest):
    """Fetch real-time flight details from Google Flights using SerpAPI."""
    logger.info(f"Searching flights: {flight_request.origin} to {flight_request.destination}")

    params = {
        "api_key": SERP_API_KEY,
        "engine": "google_flights",
        "hl": "en",
        "gl": "us",
        "departure_id": flight_request.origin.strip().upper(),
        "arrival_id": flight_request.destination.strip().upper(),
        "outbound_date": flight_request.outbound_date,
        "return_date": flight_request.return_date,
        "currency": "USD"
    }

    search_results = await run_search(params)

    if "error" in search_results:
        logger.error(f"Flight search error: {search_results['error']}")
        return {"error": search_results["error"]}

    best_flights = search_results.get("best_flights", [])
    if not best_flights:
        logger.warning("No flights found in search results")
        return []

    formatted_flights = []
    for flight in best_flights:
        if not flight.get("flights") or len(flight["flights"]) == 0:
            continue

        first_leg = flight["flights"][0]
        formatted_flights.append(FlightInfo(
            airline=first_leg.get("airline", "Unknown Airline"),
            price=str(flight.get("price", "N/A")),
            duration=f"{flight.get('total_duration', 'N/A')} min",
            stops="Nonstop" if len(flight["flights"]) == 1 else f"{len(flight['flights']) - 1} stop(s)",
            departure=f"{first_leg.get('departure_airport', {}).get('name', 'Unknown')} ({first_leg.get('departure_airport', {}).get('id', '???')}) at {first_leg.get('departure_airport', {}).get('time', 'N/A')}",
            arrival=f"{first_leg.get('arrival_airport', {}).get('name', 'Unknown')} ({first_leg.get('arrival_airport', {}).get('id', '???')}) at {first_leg.get('arrival_airport', {}).get('time', 'N/A')}",
            travel_class=first_leg.get("travel_class", "Economy"),
            return_date=flight_request.return_date,
            airline_logo=first_leg.get("airline_logo", "")
        ))

    logger.info(f"Found {len(formatted_flights)} flights")
    return formatted_flights


async def search_hotels(hotel_request: HotelRequest):
    """Fetch hotel information from SerpAPI."""
    logger.info(f"Searching hotels for: {hotel_request.location}")

    params = {
        "api_key": SERP_API_KEY,
        "engine": "google_hotels",
        "q": hotel_request.location,
        "hl": "en",
        "gl": "us",
        "check_in_date": hotel_request.check_in_date,
        "check_out_date": hotel_request.check_out_date,
        "currency": "USD",
        "sort_by": 3,
        "rating": 8
    }

    search_results = await run_search(params)

    if "error" in search_results:
        logger.error(f"Hotel search error: {search_results['error']}")
        return {"error": search_results["error"]}

    hotel_properties = search_results.get("properties", [])
    if not hotel_properties:
        logger.warning("No hotels found in search results")
        return []

    formatted_hotels = []
    for hotel in hotel_properties:
        try:
            formatted_hotels.append(HotelInfo(
                name=hotel.get("name", "Unknown Hotel"),
                price=hotel.get("rate_per_night", {}).get("lowest", "N/A"),
                rating=hotel.get("overall_rating", 0.0),
                location=hotel.get("location", "N/A"),
                link=hotel.get("link", "N/A")
            ))
        except Exception as e:
            logger.warning(f"Error formatting hotel data: {str(e)}")
            # Continue with next hotel rather than failing completely

    logger.info(f"Found {len(formatted_hotels)} hotels")
    return formatted_hotels


# ==============================================
# 🔄 Format Data for AI
# ==============================================
def format_travel_data(data_type, data):
    """Generic formatter for both flight and hotel data."""
    if not data:
        return f"No {data_type} available."

    if data_type == "flights":
        formatted_text = "✈️ **Available flight options**:\n\n"
        for i, flight in enumerate(data):
            formatted_text += (
                f"**Flight {i + 1}:**\n"
                f"✈️ **Airline:** {flight.airline}\n"
                f"💰 **Price:** ${flight.price}\n"
                f"⏱️ **Duration:** {flight.duration}\n"
                f"🛑 **Stops:** {flight.stops}\n"
                f"🕔 **Departure:** {flight.departure}\n"
                f"🕖 **Arrival:** {flight.arrival}\n"
                f"💺 **Class:** {flight.travel_class}\n\n"
            )
    elif data_type == "hotels":
        formatted_text = "🏨 **Available Hotel Options**:\n\n"
        for i, hotel in enumerate(data):
            formatted_text += (
                f"**Hotel {i + 1}:**\n"
                f"🏨 **Name:** {hotel.name}\n"
                f"💰 **Price:** ${hotel.price}\n"
                f"⭐ **Rating:** {hotel.rating}\n"
                f"📍 **Location:** {hotel.location}\n"
                f"🔗 **More Info:** [Link]({hotel.link})\n\n"
            )
    else:
        return "Invalid data type."

    return formatted_text.strip()


# ==============================================
# 🧠 AI Analysis Functions
# ==============================================
async def get_ai_recommendation(data_type, formatted_data):
    """Generate an AI recommendation using Google Gemini (google-generativeai).

    Falls back to the CrewAI-based agent if the Gemini client is not available or fails.
    """
    logger.info(f"Getting {data_type} analysis from AI (Gemini)")

    # Build prompts depending on data type
    if data_type == "flights":
        prompt_intro = (
            "You are an expert travel analyst. Analyze the following flight options and recommend the best one "
            "considering price, duration, stops, and overall convenience. Provide clear reasoning for your choice.\n\n"
        )
    elif data_type == "hotels":
        prompt_intro = (
            "You are an expert hotel analyst. Analyze the following hotel options and recommend the best one "
            "considering price, rating, location, and amenities. Provide clear reasoning for your choice.\n\n"
        )
    else:
        raise ValueError("Invalid data type for AI recommendation")

    prompt = f"{prompt_intro}Data:\n{formatted_data}\n\nPlease provide a concise recommendation with bullet points for reasoning."

    # Try to use google-generativeai (Gemini) first
    try:
        # Try the newer `google.genai` package first (some installs expose this)
        try:
            import google.genai as genai
        except Exception:
            import google.generativeai as genai

        # Configure client with API key (support multiple client versions)
        try:
            # many versions expose a configure helper
            genai.configure(api_key=GEMINI_API_KEY)
        except Exception:
            try:
                genai.api_key = GEMINI_API_KEY
            except Exception:
                pass
        # Define a single call path that works across common client shapes
        def call_gemini():
            # Preferred helper if available
            # Newer clients often expose `models.generate_text` or top-level generate_text
            model_variants = [
                "models/gemini-2.5-flash",
                "models/gemini/gemini-2.5-flash",
                "gemini-2.5-flash",
            ]

            # Try top-level helper
            if hasattr(genai, "generate_text"):
                for mdl in model_variants:
                    try:
                        return genai.generate_text(model=mdl, input=prompt)
                    except TypeError:
                        try:
                            return genai.generate_text(model=mdl, prompt=prompt)
                        except Exception:
                            continue
                    except Exception:
                        continue

            # Try `models.generate_text` shape
            if hasattr(genai, "models") and hasattr(genai.models, "generate_text"):
                for mdl in model_variants:
                    try:
                        return genai.models.generate_text(model=mdl, input=prompt)
                    except TypeError:
                        try:
                            return genai.models.generate_text(model=mdl, prompt=prompt)
                        except Exception:
                            continue
                    except Exception:
                        continue

            # Fallback to older TextGeneration client
            if hasattr(genai, "TextGeneration"):
                gen = genai.TextGeneration()
                for mdl in model_variants:
                    try:
                        return gen.generate(model=mdl, input=prompt)
                    except Exception:
                        continue

            raise RuntimeError("google generative client does not expose a supported generation method or model variants failed")

        try:
            gen_result = await asyncio.to_thread(call_gemini)

            if gen_result is None:
                return "(No response from Gemini)"

            # Normalize common response shapes to plain text
            text = None

            # If object-like with `candidates`
            if isinstance(gen_result, dict):
                candidates = gen_result.get("candidates") or gen_result.get("outputs")
                if isinstance(candidates, list) and candidates:
                    first = candidates[0]
                    # candidate may be string or dict with `content`/`text`
                    if isinstance(first, str):
                        text = first
                    elif isinstance(first, dict):
                        text = first.get("content") or first.get("text") or first.get("output")
                # Fallback keys
                text = text or gen_result.get("output_text") or gen_result.get("text") or gen_result.get("response")

            # If object has attributes
            if text is None:
                # Some client libs return objects with `.output` or `.candidates`
                try:
                    if hasattr(gen_result, "output"):
                        attr = getattr(gen_result, "output")
                        if isinstance(attr, str):
                            text = attr
                        elif isinstance(attr, list) and attr:
                            first = attr[0]
                            text = getattr(first, "content", None) or getattr(first, "text", None) or str(first)
                    if text is None and hasattr(gen_result, "candidates"):
                        cand = getattr(gen_result, "candidates")
                        if isinstance(cand, list) and cand:
                            first = cand[0]
                            text = getattr(first, "content", None) or getattr(first, "text", None) or str(first)
                except Exception:
                    pass

            # Final fallback to string conversion
            if text is None:
                try:
                    text = str(gen_result)
                except Exception:
                    text = None

            return text or "(No textual content returned by Gemini)"

        except Exception as e:
            logger.warning(f"Gemini call failed: {e}")
            # Fall through to CrewAI fallback
    except Exception as e:
        logger.info(f"google-generativeai not available or failed to import: {e}")

    # --- Fallback: use a local rule-based recommender to avoid external API failures ---
    logger.info("Falling back to local rule-based recommendation")

    def simple_rule_based_recommendation(data_type: str, text: str) -> str:
        import re

        if not text or not isinstance(text, str):
            return f"No {data_type} data available for recommendation."

        if data_type == "flights":
            # Extract flights blocks and parse price, stops, duration
            flight_blocks = re.split(r"\*\*Flight \d+:", text)[1:]
            parsed = []
            for block in flight_blocks:
                price_m = re.search(r"\$?(\d{1,6})", block)
                stops_m = re.search(r"(Nonstop|\d+ stop)", block)
                duration_m = re.search(r"(\d+) min", block)
                airline_m = re.search(r"Airline:\*\*\s*(\w+)", block)
                price = int(price_m.group(1)) if price_m else float('inf')
                stops = stops_m.group(1) if stops_m else "Unknown"
                duration = int(duration_m.group(1)) if duration_m else 99999
                airline = airline_m.group(1) if airline_m else "Unknown"
                parsed.append({"price": price, "stops": stops, "duration": duration, "airline": airline, "block": block.strip()})

            if not parsed:
                return "No structured flight details available to recommend from."

            # Prefer nonstop then lowest price then shortest duration
            def score(f):
                nonstop_bonus = 0 if 'Nonstop' in f['stops'] else 1
                return (nonstop_bonus, f['price'], f['duration'])

            best = sorted(parsed, key=score)[0]
            reasoning = [
                f"Selected flight with airline {best['airline']}",
                f"Price: ${best['price']}",
                f"Stops: {best['stops']}",
                f"Duration: {best['duration']} min",
            ]
            return "Recommended Flight:\n- " + "\n- ".join(reasoning) + "\n\n(Recommendation generated by a local rule-based fallback.)"

        elif data_type == "hotels":
            # Parse hotels for rating and price
            hotel_blocks = re.split(r"\*\*Hotel \d+:", text)[1:]
            parsed = []
            for block in hotel_blocks:
                price_m = re.search(r"\$?(\d{1,6})", block)
                rating_m = re.search(r"(\d(?:\.\d)?)", block)
                name_m = re.search(r"Name:\*\*\s*(.*?)\n", block)
                price = int(price_m.group(1)) if price_m else float('inf')
                rating = float(rating_m.group(1)) if rating_m else 0.0
                name = name_m.group(1).strip() if name_m else "Unknown"
                parsed.append({"price": price, "rating": rating, "name": name, "block": block.strip()})

            if not parsed:
                return "No structured hotel details available to recommend from."

            # Prefer highest rating then lowest price
            best = sorted(parsed, key=lambda h: (-h['rating'], h['price']))[0]
            reasoning = [
                f"Selected hotel: {best['name']}",
                f"Rating: {best['rating']}",
                f"Price per night: ${best['price']}",
            ]
            return "Recommended Hotel:\n- " + "\n- ".join(reasoning) + "\n\n(Recommendation generated by a local rule-based fallback.)"

        else:
            return "Invalid data type for recommendation."

    try:
        return simple_rule_based_recommendation(data_type, formatted_data)
    except Exception as e:
        logger.exception(f"Local fallback recommender failed: {e}")
        return f"Unable to generate {data_type} recommendation due to an error."


async def generate_itinerary(destination, flights_text, hotels_text, check_in_date, check_out_date):
    """Generate a detailed travel itinerary based on flight and hotel information."""
    try:
        # Convert the string dates to datetime objects
        check_in = datetime.strptime(check_in_date, "%Y-%m-%d")
        check_out = datetime.strptime(check_out_date, "%Y-%m-%d")

        # Calculate the difference in days
        days = (check_out - check_in).days

        llm_model = initialize_llm()

        analyze_agent = Agent(
            role="AI Travel Planner",
            goal="Create a detailed itinerary for the user based on flight and hotel information",
            backstory="AI travel expert generating a day-by-day itinerary including flight details, hotel stays, and must-visit locations in the destination.",
            llm=llm_model,
            verbose=False
        )

        analyze_task = Task(
            description=f"""
            Based on the following details, create a {days}-day itinerary for the user:

            **Flight Details**:
            {flights_text}

            **Hotel Details**:
            {hotels_text}

            **Destination**: {destination}

            **Travel Dates**: {check_in_date} to {check_out_date} ({days} days)

            The itinerary should include:
            - Flight arrival and departure information
            - Hotel check-in and check-out details
            - Day-by-day breakdown of activities
            - Must-visit attractions and estimated visit times
            - Restaurant recommendations for meals
            - Tips for local transportation

            📝 **Format Requirements**:
            - Use markdown formatting with clear headings (# for main headings, ## for days, ### for sections)
            - Include emojis for different types of activities (🏛️ for landmarks, 🍽️ for restaurants, etc.)
            - Use bullet points for listing activities
            - Include estimated timings for each activity
            - Format the itinerary to be visually appealing and easy to read
            """,
            agent=analyze_agent,
            expected_output="A well-structured, visually appealing itinerary in markdown format, including flight, hotel, and day-wise breakdown with emojis, headers, and bullet points."
        )

        itinerary_planner_crew = Crew(
            agents=[analyze_agent],
            tasks=[analyze_task],
            process=Process.sequential,
            verbose=False
        )

        crew_results = await asyncio.to_thread(itinerary_planner_crew.kickoff)

        # Handle different possible return types from CrewAI
        if hasattr(crew_results, 'outputs') and crew_results.outputs:
            return crew_results.outputs[0]
        elif hasattr(crew_results, 'get'):
            return crew_results.get("AI Travel Planner", "No itinerary available.")
        else:
            return str(crew_results)

    except Exception as e:
        logger.exception(f"Error generating itinerary: {str(e)}")
        return "Unable to generate itinerary due to an error. Please try again later."


# ==============================================
# 🚀 API Endpoints
# ==============================================
@app.post("/search_flights/", response_model=AIResponse)
async def get_flight_recommendations(flight_request: FlightRequest):
    """Search flights and get AI recommendation."""
    try:
        # Search for flights
        flights = await search_flights(flight_request)

        # Handle errors
        if isinstance(flights, dict) and "error" in flights:
            raise HTTPException(status_code=400, detail=flights["error"])

        if not flights:
            raise HTTPException(status_code=404, detail="No flights found")

        # Format flight data for AI
        flights_text = format_travel_data("flights", flights)

        # Get AI recommendation
        ai_recommendation = await get_ai_recommendation("flights", flights_text)

        # Return response
        return AIResponse(
            flights=flights,
            ai_flight_recommendation=ai_recommendation
        )
    except HTTPException:
        # Re-raise HTTP exceptions to preserve status codes
        raise
    except Exception as e:
        logger.exception(f"Flight search endpoint error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Flight search error: {str(e)}")


@app.post("/search_hotels/", response_model=AIResponse)
async def get_hotel_recommendations(hotel_request: HotelRequest):
    """Search hotels and get AI recommendation."""
    try:
        # Fetch hotel data
        hotels = await search_hotels(hotel_request)

        # Handle errors
        if isinstance(hotels, dict) and "error" in hotels:
            raise HTTPException(status_code=400, detail=hotels["error"])

        if not hotels:
            raise HTTPException(status_code=404, detail="No hotels found")

        # Format hotel data for AI
        hotels_text = format_travel_data("hotels", hotels)

        # Get AI recommendation
        ai_recommendation = await get_ai_recommendation("hotels", hotels_text)

        # Return response
        return AIResponse(
            hotels=hotels,
            ai_hotel_recommendation=ai_recommendation
        )
    except HTTPException:
        # Re-raise HTTP exceptions to preserve status codes
        raise
    except Exception as e:
        logger.exception(f"Hotel search endpoint error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Hotel search error: {str(e)}")


@app.post("/complete_search/", response_model=AIResponse)
async def complete_travel_search(flight_request: FlightRequest, hotel_request: Optional[HotelRequest] = None):
    """Search for flights and hotels concurrently and get AI recommendations for both."""
    try:
        # If hotel request is not provided, create one from flight request
        if hotel_request is None:
            hotel_request = HotelRequest(
                location=flight_request.destination,
                check_in_date=flight_request.outbound_date,
                check_out_date=flight_request.return_date
            )

        # Run flight and hotel searches concurrently
        flight_task = asyncio.create_task(get_flight_recommendations(flight_request))
        hotel_task = asyncio.create_task(get_hotel_recommendations(hotel_request))

        # Wait for both tasks to complete
        flight_results, hotel_results = await asyncio.gather(flight_task, hotel_task, return_exceptions=True)

        # Check for exceptions
        if isinstance(flight_results, Exception):
            logger.error(f"Flight search failed: {str(flight_results)}")
            flight_results = AIResponse(flights=[], ai_flight_recommendation="Could not retrieve flights.")

        if isinstance(hotel_results, Exception):
            logger.error(f"Hotel search failed: {str(hotel_results)}")
            hotel_results = AIResponse(hotels=[], ai_hotel_recommendation="Could not retrieve hotels.")

        # Format data for itinerary generation
        flights_text = format_travel_data("flights", flight_results.flights)
        hotels_text = format_travel_data("hotels", hotel_results.hotels)

        # Generate itinerary if both searches were successful
        itinerary = ""
        if flight_results.flights and hotel_results.hotels:
            itinerary = await generate_itinerary(
                destination=flight_request.destination,
                flights_text=flights_text,
                hotels_text=hotels_text,
                check_in_date=flight_request.outbound_date,
                check_out_date=flight_request.return_date
            )

        # Combine results
        return AIResponse(
            flights=flight_results.flights,
            hotels=hotel_results.hotels,
            ai_flight_recommendation=flight_results.ai_flight_recommendation,
            ai_hotel_recommendation=hotel_results.ai_hotel_recommendation,
            itinerary=itinerary
        )
    except Exception as e:
        logger.exception(f"Complete travel search error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Travel search error: {str(e)}")


@app.post("/generate_itinerary/", response_model=AIResponse)
async def get_itinerary(itinerary_request: ItineraryRequest):
    """Generate an itinerary based on provided flight and hotel information."""
    try:
        itinerary = await generate_itinerary(
            destination=itinerary_request.destination,
            flights_text=itinerary_request.flights,
            hotels_text=itinerary_request.hotels,
            check_in_date=itinerary_request.check_in_date,
            check_out_date=itinerary_request.check_out_date
        )

        return AIResponse(itinerary=itinerary)
    except Exception as e:
        logger.exception(f"Itinerary generation error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Itinerary generation error: {str(e)}")


# ==============================================
# 🌐 Run FastAPI Server
# ==============================================
if __name__ == "__main__":
    logger.info("Starting Travel Planning API server")
    uvicorn.run(app, host="0.0.0.0", port=8000)