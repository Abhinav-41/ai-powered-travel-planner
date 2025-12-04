import streamlit as st
import asyncio
from datetime import datetime, timedelta

# Import backend functions directly so the app runs as a single Streamlit server
from gemini2_travel_v2 import (
    search_flights,
    search_hotels,
    get_ai_recommendation,
    generate_itinerary,
    format_travel_data,
    FlightRequest,
    HotelRequest,
)

# This Streamlit app calls backend functions directly (no separate FastAPI server required)

# Page configuration
st.set_page_config(
    page_title="✈️ AI-Powered Travel Planner",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Sidebar for additional options
with st.sidebar:
    st.title("⚙️ Options")
    search_mode = st.radio(
        "Search Mode",
        ["Complete (Flights + Hotels + Itinerary)", "Flights Only", "Hotels Only"]
    )

    st.markdown("---")
    st.caption("AI-Powered Travel Planner v2.0")
    st.caption("© 2025 Travel AI Solutions")

# Main header
st.title("✈️ AI-Powered Travel Planner")
st.markdown("""
    **Find flights, hotels, and get personalized recommendations with AI! Create your perfect travel itinerary in seconds.**
""")

# Travel search form
with st.form(key="travel_search_form"):
    cols = st.columns([1, 1])

    with cols[0]:
        st.subheader("🛫 Flight Details")
        origin = st.text_input("Departure Airport (IATA code)", "ATL")
        destination = st.text_input("Arrival Airport (IATA code)", "LAX")

        # Set default dates (departure tomorrow, return in 7 days)
        tomorrow = datetime.now() + timedelta(days=1)
        next_week = tomorrow + timedelta(days=7)

        outbound_date = st.date_input("Departure Date", tomorrow)
        return_date = st.date_input("Return Date", next_week)

    with cols[1]:
        st.subheader("🏨 Hotel Details")
        use_flight_destination = st.checkbox("Use flight destination for hotel", value=True)

        if use_flight_destination:
            location = destination
            st.info(f"Using flight destination ({destination}) for hotel search")
        else:
            location = st.text_input("Hotel Location", "")

        check_in_date = st.date_input("Check-In Date", outbound_date)
        check_out_date = st.date_input("Check-Out Date", return_date)

    # Submit button
    submit_col1, submit_col2 = st.columns([3, 1])
    with submit_col2:
        submit_button = st.form_submit_button("🔍 Search", use_container_width=True)

# Handle form submission
if submit_button:
    # Validate inputs
    if not origin or not destination:
        st.error("Please provide both origin and destination airports.")
    elif outbound_date >= return_date:
        st.error("Return date must be after departure date.")
    elif check_in_date >= check_out_date:
        st.error("Check-out date must be after check-in date.")
    else:
        # Prepare request data
        flight_data = {
            "origin": origin,
            "destination": destination,
            "outbound_date": str(outbound_date),
            "return_date": str(return_date)
        }

        hotel_data = {
            "location": location,
            "check_in_date": str(check_in_date),
            "check_out_date": str(check_out_date)
        }

        # Show loading spinner and call backend functions directly
        with st.spinner("Searching for the perfect travel options for you..."):
            try:
                # Prepare Pydantic request objects
                flight_req = FlightRequest(**flight_data)
                hotel_req = HotelRequest(**hotel_data)

                # Flights + Hotels + Itinerary
                if search_mode == "Complete (Flights + Hotels + Itinerary)":
                    flights_raw = asyncio.run(search_flights(flight_req))
                    hotels_raw = asyncio.run(search_hotels(hotel_req))

                    # Keep original models for AI formatting, and convert to dicts for display
                    flights_models = flights_raw if isinstance(flights_raw, list) else []
                    hotels_models = hotels_raw if isinstance(hotels_raw, list) else []

                    flights = [f.dict() if hasattr(f, 'dict') else f for f in flights_models]
                    hotels = [h.dict() if hasattr(h, 'dict') else h for h in hotels_models]

                    # AI recommendations (use models for formatting)
                    ai_flight_recommendation = asyncio.run(get_ai_recommendation("flights", format_travel_data("flights", flights_models)))
                    ai_hotel_recommendation = asyncio.run(get_ai_recommendation("hotels", format_travel_data("hotels", hotels_models)))

                    # Itinerary (pass formatted text from models)
                    itinerary = asyncio.run(generate_itinerary(destination, format_travel_data("flights", flights_models), format_travel_data("hotels", hotels_models), str(check_in_date), str(check_out_date)))

                # Flights Only
                elif search_mode == "Flights Only":
                    flights_raw = asyncio.run(search_flights(flight_req))
                    flights_models = flights_raw if isinstance(flights_raw, list) else []
                    flights = [f.dict() if hasattr(f, 'dict') else f for f in flights_models]

                    ai_flight_recommendation = asyncio.run(get_ai_recommendation("flights", format_travel_data("flights", flights_models)))
                    hotels = []
                    ai_hotel_recommendation = ""
                    itinerary = ""

                # Hotels Only
                elif search_mode == "Hotels Only":
                    hotels_raw = asyncio.run(search_hotels(hotel_req))
                    hotels_models = hotels_raw if isinstance(hotels_raw, list) else []
                    hotels = [h.dict() if hasattr(h, 'dict') else h for h in hotels_models]

                    ai_hotel_recommendation = asyncio.run(get_ai_recommendation("hotels", format_travel_data("hotels", hotels_models)))
                    flights = []
                    ai_flight_recommendation = ""
                    itinerary = ""

            except Exception as e:
                st.error(f"An error occurred: {str(e)}")
                st.stop()

        # Display results in tabs
        if search_mode == "Flights Only":
            tabs = st.tabs(["✈️ Flights", "🏆 AI Recommendation"])
        elif search_mode == "Hotels Only":
            tabs = st.tabs(["🏨 Hotels", "🏆 AI Recommendation"])
        else:
            tabs = st.tabs(["✈️ Flights", "🏨 Hotels", "🏆 AI Recommendations", "📅 Itinerary"])

        # Flights tab
        if search_mode != "Hotels Only":
            with tabs[0]:
                st.subheader(f"✈️ Available Flights from {origin} to {destination}")

                if flights:
                    # Create two columns for flight cards
                    flight_cols = st.columns(2)

                    for i, flight in enumerate(flights):
                        col_idx = i % 2
                        with flight_cols[col_idx]:
                            with st.container(border=True):
                                st.markdown(f"""
                                ### ✈️ {flight['airline']} - {flight['stops']} Flight

                                🕒 **Departure**: {flight['departure']}  
                                🕘 **Arrival**: {flight['arrival']}  
                                ⏱️ **Duration**: {flight['duration']}  
                                💰 **Price**: **${flight['price']}**  
                                💺 **Class**: {flight['travel_class']}
                                """)
                                st.button(f"🔖 Select This Flight", key=f"flight_{i}")
                else:
                    st.info("No flights found for your search criteria.")

        # Hotels tab
        if search_mode != "Flights Only":
            with tabs[1 if search_mode == "Hotels Only" else 1]:
                st.subheader(f"🏨 Available Hotels in {location}")

                if hotels:
                    # Create columns for hotel cards
                    hotel_cols = st.columns(3)

                    for i, hotel in enumerate(hotels):
                        col_idx = i % 3
                        with hotel_cols[col_idx]:
                            with st.container(border=True):
                                st.markdown(f"""
                                ### 🏨 {hotel['name']}

                                💰 **Price**: ${hotel['price']} per night  
                                ⭐ **Rating**: {hotel['rating']}  
                                📍 **Location**: {hotel['location']}
                                """)
                                cols = st.columns([1, 1])
                                with cols[0]:
                                    st.button(f"🔖 Select", key=f"hotel_{i}")
                                with cols[1]:
                                    st.link_button("🔗 Details", hotel['link'])
                else:
                    st.info("No hotels found for your search criteria.")

        # AI Recommendations tab
        recommendation_tab_index = 1 if search_mode in ["Flights Only", "Hotels Only"] else 2
        with tabs[recommendation_tab_index]:
            if search_mode != "Hotels Only" and ai_flight_recommendation:
                st.subheader("✈️ AI Flight Recommendation")
                with st.container(border=True):
                    st.markdown(ai_flight_recommendation)

            if search_mode != "Flights Only" and ai_hotel_recommendation:
                st.subheader("🏨 AI Hotel Recommendation")
                with st.container(border=True):
                    st.markdown(ai_hotel_recommendation)

        # Itinerary tab
        if search_mode == "Complete (Flights + Hotels + Itinerary)" and itinerary:
            with tabs[3]:
                st.subheader("📅 Your Travel Itinerary")
                with st.container(border=True):
                    st.markdown(itinerary)

                # Download button for itinerary
                st.download_button(
                    label="📥 Download Itinerary",
                    data=itinerary,
                    file_name=f"travel_itinerary_{destination}_{outbound_date}.md",
                    mime="text/markdown"
                )

