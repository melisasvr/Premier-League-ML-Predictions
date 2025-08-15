# Premier League ML Predictions
- This project provides a machine learning-based system for predicting Premier League match outcomes for the 2025-26 season.
- It includes scripts to collect fixture and team data, train a predictive model, and a web-based dashboard to visualize predictions.
- The system uses historical and current team statistics to forecast match results with confidence scores and probability distributions.

## Key features:
- Data Collection: Fetches Premier League fixtures and team stats using APIs (with a fallback to sample data).
- Prediction Model: Utilizes ensemble machine learning (Random Forest and Gradient Boosting) to predict match outcomes (Home Win, Draw, Away Win).
- Dashboard: Displays predictions, confidence levels, and a bar chart of prediction confidences for the current matchday.
- The project is updated with fixtures for Matchday 1 (August 15-16, 2025), including Liverpool vs. Bournemouth and five matches on August 16: Aston Villa vs. Newcastle, Brighton vs. Fulham, Sunderland vs. West Ham, Tottenham vs. Burnley, and Wolves vs. Man City.

## Project Structure
- premiere_league_data_collection.py: Collects Premier League fixtures and team statistics, with a fallback to sample data for demonstration.
- premier_league_predictor.py: Implements the machine learning model to predict match outcomes based on team stats and historical data.
- index.html: A web-based dashboard displaying match predictions, confidence levels, and a bar chart using Chart.js.
- README.md: This file provides a project overview and instructions.

## Prerequisites
- Python 3.8+
- Python Libraries:
- requests
- pandas
- numpy
- scikit-learn

- Web Browser: For viewing the dashboard (index.html).
- Optional: API key from football-data.org for real-time fixture data.

- Install Python dependencies using:
- `pip install requests pandas numpy scikit-learn`

## Setup
- Clone the Repository (if hosted on GitHub or similar):
-` git clone <repository-url>`
- `cd premier-league-ml-predictions`
- Configure API Key (optional):
- Register at football-data.org for a free API key.
- In premiere_league_data_collection.py, replace 'YOUR_API_KEY_HERE' with your API key in the get_current_fixtures method.

## Prepare the Dashboard:
- The index.html file is standalone and requires no server setup. Open it directly in a web browser to view predictions.
- Ensure an internet connection for Chart.js (loaded via CDN).

## Usage
- Running Predictions
- Ensure premiere_league_data_collection.py and premier_league_predictor.py are in the same directory.
- Run the prediction script:
- python premiere_league_data_collection.py
- This will:
- Fetch current fixtures (or use sample fixtures if the API fails).
- Train the ML model using sample historical data.
- Output predictions for the current matchday, including confidence scores and probabilities.

## Sample Output:
```🔄 Collecting Premier League Data...
📅 Found 6 fixtures this week:
date                home_team  away_team
2025-08-15T20:00:00Z  Liverpool  Bournemouth
2025-08-16T15:00:00Z  Aston Villa  Newcastle
...

⚽ MATCH PREDICTIONS
============================================================
🏟️  Liverpool vs Bournemouth
📅 2025-08-15T20:00:00Z
🎯 Prediction: Home Win
📊 Confidence: 82.0%
📈 Probabilities:
   Home Win: 82.0%
   Draw: 12.0%
   Away Win: 6.0%
...

```

## Viewing the Dashboard
```Open index.html in a web browser.
The dashboard displays:
A stats grid with model accuracy, number of matches predicted, high-confidence predictions (>70%), and current matchday.
Match cards showing predicted outcomes, confidence scores, and probability bars for each fixture.
A bar chart visualizing prediction confidence for each match.
A model information section with feature importance.
Click the "Refresh Predictions" button to simulate updated predictions with slight variations.
Sample Fixtures (Matchday 1, 2025-26 Season)
August 15, 2025:
Liverpool vs. Bournemouth (20:00 BST)
August 16, 2025:
Aston Villa vs. Newcastle (15:00 BST)
Brighton vs. Fulham (15:00 BST)
Sunderland vs. West Ham (15:00 BST)
Tottenham vs. Burnley (15:00 BST)
Wolves vs. Man City (15:00 BST)
```
## License
- This project is for educational purposes and uses sample data. Ensure compliance with API terms (e.g., football-data.org) when using real data.
