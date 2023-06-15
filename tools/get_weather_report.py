import argparse
import datetime
import urllib.parse

import requests

# Create the parser
parser = argparse.ArgumentParser(description="Fetch weather forecast data")

# Add the arguments
parser.add_argument(
    "--API_KEY",
    type=str,
    default="e1689f8fed76f26be49befd1fabae8a1",
    help="Your OpenWeatherMap API Key",
)
parser.add_argument(
    "--location", type=str, required=True, help="Location for weather forecast"
)
parser.add_argument(
    "--date",
    type=lambda s: datetime.datetime.strptime(s, "%Y-%m-%d").date(),
    required=True,
    help="Target date for weather forecast (format: YYYY-MM-DD)",
)

# Parse the arguments
args = parser.parse_args()

params = {"q": args.location, "appid": args.API_KEY}

url = (
    f"http://api.openweathermap.org/data/2.5/forecast?{urllib.parse.urlencode(params)}"
)

# Send a GET request to the OpenWeatherMap API
response = requests.get(url)

# Raise an exception if the GET request was unsuccessful
response.raise_for_status()

# Convert the response to JSON
data = response.json()

# Iterate over the forecast data
for forecast in data["list"]:
    # Convert the forecast timestamp to a date
    forecast_date = datetime.datetime.fromtimestamp(forecast["dt"]).date()

    # If this forecast is for the target date, print it
    if forecast_date == args.date:
        print(f"Date & Time: {datetime.datetime.fromtimestamp(forecast['dt'])}")
        print(f"Weather Description: {forecast['weather'][0]['description']}")
        print(f"Temperature: {forecast['main']['temp']} K")
        print(f"Feels Like: {forecast['main']['feels_like']} K")
        print(f"Humidity: {forecast['main']['humidity']} %")
        print(f"Wind Speed: {forecast['wind']['speed']} m/s")
        print("\n")
