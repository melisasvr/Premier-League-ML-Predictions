import requests
import pandas as pd
from datetime import datetime, timedelta
import json
import time

class PLDataCollector:
    """
    Collect Premier League data from various free APIs
    """
    
    def __init__(self):
        # Free API endpoints (no API key required)
        self.base_urls = {
            'football_data': 'https://api.football-data.org/v4/',
            'api_sports': 'https://v3.football.api-sports.io/',
            'free_football': 'https://www.thesportsdb.com/api/v1/json/'
        }
        
    def get_current_fixtures(self):
        """
        Get this week's Premier League fixtures
        """
        try:
            # Using football-data.org API (free tier available)
            # You'll need to register for a free API key at football-data.org
            headers = {
                'X-Auth-Token': 'YOUR_API_KEY_HERE'  # Replace with actual API key
            }
            
            # Get Premier League ID (2021 is Premier League in football-data.org)
            url = f"{self.base_urls['football_data']}competitions/2021/matches"
            
            # Get matches for current matchday
            params = {
                'status': 'SCHEDULED',
                'dateFrom': datetime.now().strftime('%Y-%m-%d'),
                'dateTo': (datetime.now() + timedelta(days=7)).strftime('%Y-%m-%d')
            }
            
            response = requests.get(url, headers=headers, params=params)
            
            if response.status_code == 200:
                data = response.json()
                fixtures = []
                
                for match in data['matches']:
                    fixtures.append({
                        'date': match['utcDate'],
                        'home_team': match['homeTeam']['name'],
                        'away_team': match['awayTeam']['name'],
                        'matchday': match['matchday']
                    })
                
                return pd.DataFrame(fixtures)
            
        except Exception as e:
            print(f"API Error: {e}")
            
        # Fallback: Return sample fixtures for demonstration
        return self.get_sample_fixtures()
    
    def get_sample_fixtures(self):
        """
        Sample fixtures for demonstration (updated with correct matches)
        """
        sample_fixtures = [
            {
                'date': '2025-08-15T20:00:00Z',
                'home_team': 'Liverpool',
                'away_team': 'Bournemouth',
                'matchday': 1
            },
            {
                'date': '2025-08-16T15:00:00Z',
                'home_team': 'Aston Villa',
                'away_team': 'Newcastle',
                'matchday': 1
            },
            {
                'date': '2025-08-16T15:00:00Z',
                'home_team': 'Brighton',
                'away_team': 'Fulham',
                'matchday': 1
            },
            {
                'date': '2025-08-16T15:00:00Z',
                'home_team': 'Sunderland',
                'away_team': 'West Ham',
                'matchday': 1
            },
            {
                'date': '2025-08-16T15:00:00Z',
                'home_team': 'Tottenham',
                'away_team': 'Burnley',
                'matchday': 1
            },
            {
                'date': '2025-08-16T15:00:00Z',
                'home_team': 'Wolves',
                'away_team': 'Man City',
                'matchday': 1
            }
        ]
        
        return pd.DataFrame(sample_fixtures)
    
    def get_team_stats(self, team_name):
        """
        Get team statistics for the current season
        In practice, you'd collect this from APIs or databases
        """
        # Sample team stats - replace with real data collection
        team_stats_db = {
            'Arsenal': {
                'strength': 82,
                'recent_form': 0.8,
                'goals_scored_avg': 2.1,
                'goals_conceded_avg': 1.0,
                'win_percentage': 0.65,
                'injury_count': 2
            },
            'Liverpool': {
                'strength': 80,
                'recent_form': 0.6,
                'goals_scored_avg': 1.9,
                'goals_conceded_avg': 1.1,
                'win_percentage': 0.60,
                'injury_count': 1
            },
            'Man City': {
                'strength': 85,
                'recent_form': 1.2,
                'goals_scored_avg': 2.3,
                'goals_conceded_avg': 0.9,
                'win_percentage': 0.75,
                'injury_count': 1
            },
            'Chelsea': {
                'strength': 75,
                'recent_form': 0.2,
                'goals_scored_avg': 1.6,
                'goals_conceded_avg': 1.3,
                'win_percentage': 0.50,
                'injury_count': 3
            },
            'Man United': {
                'strength': 68,
                'recent_form': 0.4,
                'goals_scored_avg': 1.5,
                'goals_conceded_avg': 1.4,
                'win_percentage': 0.45,
                'injury_count': 2
            },
            'Brighton': {
                'strength': 65,
                'recent_form': 0.9,
                'goals_scored_avg': 1.7,
                'goals_conceded_avg': 1.2,
                'win_percentage': 0.55,
                'injury_count': 1
            },
            'Newcastle': {
                'strength': 70,
                'recent_form': 0.7,
                'goals_scored_avg': 1.8,
                'goals_conceded_avg': 1.1,
                'win_percentage': 0.58,
                'injury_count': 0
            },
            'Wolves': {
                'strength': 50,
                'recent_form': -0.2,
                'goals_scored_avg': 1.2,
                'goals_conceded_avg': 1.6,
                'win_percentage': 0.35,
                'injury_count': 3
            },
            'Southampton': {
                'strength': 45,
                'recent_form': -0.5,
                'goals_scored_avg': 1.1,
                'goals_conceded_avg': 1.8,
                'win_percentage': 0.30,
                'injury_count': 2
            },
            'Ipswich': {
                'strength': 40,
                'recent_form': -0.3,
                'goals_scored_avg': 1.0,
                'goals_conceded_avg': 1.9,
                'win_percentage': 0.25,
                'injury_count': 1
            },
            'Aston Villa': {
                'strength': 63,
                'recent_form': 0.5,
                'goals_scored_avg': 1.7,
                'goals_conceded_avg': 1.3,
                'win_percentage': 0.50,
                'injury_count': 2
            },
            'West Ham': {
                'strength': 60,
                'recent_form': 0.3,
                'goals_scored_avg': 1.6,
                'goals_conceded_avg': 1.4,
                'win_percentage': 0.45,
                'injury_count': 1
            },
            'Fulham': {
                'strength': 53,
                'recent_form': 0.4,
                'goals_scored_avg': 1.5,
                'goals_conceded_avg': 1.3,
                'win_percentage': 0.40,
                'injury_count': 2
            },
            'Sunderland': {
                'strength': 48,
                'recent_form': 0.1,
                'goals_scored_avg': 1.4,
                'goals_conceded_avg': 1.5,
                'win_percentage': 0.35,
                'injury_count': 1
            },
            'Tottenham': {
                'strength': 72,
                'recent_form': 0.6,
                'goals_scored_avg': 1.9,
                'goals_conceded_avg': 1.2,
                'win_percentage': 0.55,
                'injury_count': 1
            },
            'Burnley': {
                'strength': 33,
                'recent_form': -0.4,
                'goals_scored_avg': 1.0,
                'goals_conceded_avg': 1.7,
                'win_percentage': 0.30,
                'injury_count': 2
            },
            'Bournemouth': {
                'strength': 40,
                'recent_form': 0.0,
                'goals_scored_avg': 1.3,
                'goals_conceded_avg': 1.5,
                'win_percentage': 0.40,
                'injury_count': 2
            }
        }
        
        return team_stats_db.get(team_name, {
            'strength': 50,
            'recent_form': 0.0,
            'goals_scored_avg': 1.3,
            'goals_conceded_avg': 1.5,
            'win_percentage': 0.40,
            'injury_count': 2
        })
    
    def get_head_to_head(self, home_team, away_team):
        """
        Get head-to-head record between two teams
        """
        # Sample H2H data - in practice, query historical database
        h2h_data = {
            ('Liverpool', 'Bournemouth'): {'home_wins': 7, 'away_wins': 2},
            ('Aston Villa', 'Newcastle'): {'home_wins': 4, 'away_wins': 3},
            ('Brighton', 'Fulham'): {'home_wins': 3, 'away_wins': 2},
            ('Sunderland', 'West Ham'): {'home_wins': 4, 'away_wins': 4},
            ('Tottenham', 'Burnley'): {'home_wins': 6, 'away_wins': 2},
            ('Wolves', 'Man City'): {'home_wins': 2, 'away_wins': 7}
        }
        
        key = (home_team, away_team)
        return h2h_data.get(key, {'home_wins': 5, 'away_wins': 5})
    
    def prepare_match_data(self, home_team, away_team):
        """
        Prepare all data needed for a match prediction
        """
        home_stats = self.get_team_stats(home_team)
        away_stats = self.get_team_stats(away_team)
        h2h = self.get_head_to_head(home_team, away_team)
        
        match_data = {
            'home_team': home_team,
            'away_team': away_team,
            'home_strength': home_stats['strength'],
            'away_strength': away_stats['strength'],
            'home_recent_form': home_stats['recent_form'],
            'away_recent_form': away_stats['recent_form'],
            'home_goals_scored_avg': home_stats['goals_scored_avg'],
            'away_goals_scored_avg': away_stats['goals_scored_avg'],
            'home_goals_conceded_avg': home_stats['goals_conceded_avg'],
            'away_goals_conceded_avg': away_stats['goals_conceded_avg'],
            'home_win_percentage': home_stats['win_percentage'],
            'away_win_percentage': away_stats['win_percentage'],
            'head_to_head_home': h2h['home_wins'],
            'head_to_head_away': h2h['away_wins'],
            'home_injury_count': home_stats['injury_count'],
            'away_injury_count': away_stats['injury_count'],
            'days_since_last_match_home': 7,  # Assuming weekly matches
            'days_since_last_match_away': 7
        }
        
        return match_data

# Integration script
def run_weekly_predictions():
    """
    Complete workflow: collect data and make predictions
    """
    print("🔄 Collecting Premier League Data...")
    
    # Initialize components
    collector = PLDataCollector()
    
    # Get this week's fixtures
    fixtures = collector.get_current_fixtures()
    
    print(f"📅 Found {len(fixtures)} fixtures this week:")
    print(fixtures[['date', 'home_team', 'away_team']].to_string(index=False))
    
    # Initialize and train predictor (using previous code)
    from premier_league_predictor import PremierLeaguePredictor
    
    predictor = PremierLeaguePredictor()
    
    # Load historical data and train
    print("\n🤖 Training prediction model...")
    historical_data = predictor.create_sample_historical_data()
    X, y = predictor.prepare_data(historical_data)
    predictor.train_model(X, y)
    
    # Make predictions for each fixture
    print("\n⚽ MATCH PREDICTIONS")
    print("=" * 60)
    
    predictions_summary = []
    
    for _, fixture in fixtures.iterrows():
        home_team = fixture['home_team']
        away_team = fixture['away_team']
        match_date = fixture['date']
        
        # Collect match data
        match_data = collector.prepare_match_data(home_team, away_team)
        
        # Make prediction
        prediction = predictor.predict_match(home_team, away_team, match_data)
        
        # Display result
        print(f"\n🏟️  {home_team} vs {away_team}")
        print(f"📅 {match_date}")
        print(f"🎯 Prediction: {prediction['predicted_outcome']}")
        print(f"📊 Confidence: {prediction['confidence']:.1%}")
        print("📈 Probabilities:")
        for outcome, prob in prediction['probabilities'].items():
            print(f"   {outcome}: {prob:.1%}")
        
        predictions_summary.append({
            'match': f"{home_team} vs {away_team}",
            'prediction': prediction['predicted_outcome'],
            'confidence': f"{prediction['confidence']:.1%}",
            **prediction['probabilities']
        })
    
    # Summary table
    print(f"\n📋 PREDICTIONS SUMMARY")
    print("=" * 60)
    summary_df = pd.DataFrame(predictions_summary)
    print(summary_df.to_string(index=False))
    
    return predictions_summary

if __name__ == "__main__":
    # Run the complete prediction workflow
    predictions = run_weekly_predictions()