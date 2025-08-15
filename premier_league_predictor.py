import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import StandardScaler
import requests
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

class PremierLeaguePredictor:
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.feature_columns = []
        
    def create_sample_historical_data(self):
        """
        Create sample historical data for demonstration.
        In practice, you'd load this from APIs or CSV files.
        """
        np.random.seed(42)
        
        # Sample team data (last 3 seasons)
        teams = ['Arsenal', 'Man City', 'Liverpool', 'Chelsea', 'Tottenham', 
                'Newcastle', 'Man United', 'Brighton', 'Aston Villa', 'West Ham',
                'Crystal Palace', 'Fulham', 'Wolves', 'Everton', 'Brentford',
                'Nottingham Forest', 'Luton', 'Burnley', 'Sheffield United', 'Bournemouth']
        
        # Generate sample match data
        matches = []
        for i in range(1000):  # 1000 historical matches
            home_team = np.random.choice(teams)
            away_team = np.random.choice([t for t in teams if t != home_team])
            
            # Simulate team strengths (higher = better)
            team_strength = {
                'Man City': 85, 'Arsenal': 82, 'Liverpool': 80, 'Chelsea': 75,
                'Tottenham': 72, 'Newcastle': 70, 'Man United': 68, 'Brighton': 65,
                'Aston Villa': 63, 'West Ham': 60, 'Crystal Palace': 55, 'Fulham': 53,
                'Wolves': 50, 'Everton': 48, 'Brentford': 45, 'Nottingham Forest': 43,
                'Bournemouth': 40, 'Luton': 35, 'Burnley': 33, 'Sheffield United': 30
            }
            
            home_strength = team_strength.get(home_team, 50)
            away_strength = team_strength.get(away_team, 50)
            
            # Home advantage
            home_advantage = 3
            
            # Calculate outcome probabilities
            strength_diff = (home_strength + home_advantage) - away_strength
            
            # Add some randomness
            form_factor = np.random.normal(0, 5)  # Recent form
            injury_factor = np.random.normal(0, 3)  # Injury impact
            motivation_factor = np.random.normal(0, 2)  # Motivation/pressure
            
            total_advantage = strength_diff + form_factor + injury_factor + motivation_factor
            
            # Convert to probabilities
            if total_advantage > 8:
                outcome = 'H'  # Home win
            elif total_advantage < -5:
                outcome = 'A'  # Away win
            else:
                outcome = np.random.choice(['H', 'D', 'A'], p=[0.4, 0.3, 0.3])
            
            matches.append({
                'home_team': home_team,
                'away_team': away_team,
                'home_strength': home_strength,
                'away_strength': away_strength,
                'home_recent_form': np.random.normal(0, 1),  # Last 5 matches form
                'away_recent_form': np.random.normal(0, 1),
                'home_goals_scored_avg': np.random.normal(1.5, 0.5),
                'away_goals_scored_avg': np.random.normal(1.3, 0.5),
                'home_goals_conceded_avg': np.random.normal(1.2, 0.4),
                'away_goals_conceded_avg': np.random.normal(1.4, 0.4),
                'home_win_percentage': np.random.uniform(0.3, 0.7),
                'away_win_percentage': np.random.uniform(0.2, 0.6),
                'head_to_head_home': np.random.randint(0, 10),  # Home wins in last 10 H2H
                'head_to_head_away': np.random.randint(0, 10),  # Away wins in last 10 H2H
                'home_injury_count': np.random.randint(0, 5),
                'away_injury_count': np.random.randint(0, 5),
                'days_since_last_match_home': np.random.randint(3, 14),
                'days_since_last_match_away': np.random.randint(3, 14),
                'outcome': outcome
            })
        
        return pd.DataFrame(matches)
    
    def engineer_features(self, df):
        """
        Create additional features for better predictions
        """
        # Strength difference
        df['strength_difference'] = df['home_strength'] - df['away_strength']
        
        # Form difference
        df['form_difference'] = df['home_recent_form'] - df['away_recent_form']
        
        # Attack vs Defense matchups
        df['home_attack_vs_away_defense'] = df['home_goals_scored_avg'] - df['away_goals_conceded_avg']
        df['away_attack_vs_home_defense'] = df['away_goals_scored_avg'] - df['home_goals_conceded_avg']
        
        # Head-to-head advantage
        df['h2h_advantage'] = df['head_to_head_home'] - df['head_to_head_away']
        
        # Injury impact
        df['injury_difference'] = df['away_injury_count'] - df['home_injury_count']  # Positive = advantage to home
        
        # Rest advantage
        df['rest_advantage'] = df['days_since_last_match_away'] - df['days_since_last_match_home']
        
        return df
    
    def prepare_data(self, df):
        """
        Prepare data for machine learning
        """
        # Engineer features
        df = self.engineer_features(df)
        
        # Select features for model
        feature_columns = [
            'home_strength', 'away_strength', 'strength_difference',
            'home_recent_form', 'away_recent_form', 'form_difference',
            'home_goals_scored_avg', 'away_goals_scored_avg',
            'home_goals_conceded_avg', 'away_goals_conceded_avg',
            'home_attack_vs_away_defense', 'away_attack_vs_home_defense',
            'home_win_percentage', 'away_win_percentage',
            'h2h_advantage', 'home_injury_count', 'away_injury_count',
            'injury_difference', 'rest_advantage'
        ]
        
        X = df[feature_columns]
        y = df['outcome']
        
        self.feature_columns = feature_columns
        
        return X, y
    
    def train_model(self, X, y):
        """
        Train the prediction model
        """
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train ensemble model
        rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
        gb_model = GradientBoostingClassifier(n_estimators=100, random_state=42)
        
        rf_model.fit(X_train_scaled, y_train)
        gb_model.fit(X_train_scaled, y_train)
        
        # Evaluate models
        rf_score = cross_val_score(rf_model, X_train_scaled, y_train, cv=5).mean()
        gb_score = cross_val_score(gb_model, X_train_scaled, y_train, cv=5).mean()
        
        # Choose best model
        if rf_score > gb_score:
            self.model = rf_model
            print(f"Random Forest selected with CV score: {rf_score:.3f}")
        else:
            self.model = gb_model
            print(f"Gradient Boosting selected with CV score: {gb_score:.3f}")
        
        # Test performance
        y_pred = self.model.predict(X_test_scaled)
        print(f"Test Accuracy: {accuracy_score(y_test, y_pred):.3f}")
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))
        
        return self.model
    
    def predict_match(self, home_team, away_team, match_data):
        """
        Predict outcome for a single match
        """
        if self.model is None:
            raise ValueError("Model not trained yet!")
        
        # Prepare match data
        match_df = pd.DataFrame([match_data])
        match_df = self.engineer_features(match_df)
        
        X_match = match_df[self.feature_columns]
        X_match_scaled = self.scaler.transform(X_match)
        
        # Get predictions
        prediction = self.model.predict(X_match_scaled)[0]
        probabilities = self.model.predict_proba(X_match_scaled)[0]
        
        # Map to readable format
        outcome_map = {'H': 'Home Win', 'D': 'Draw', 'A': 'Away Win'}
        classes = self.model.classes_
        
        result = {
            'predicted_outcome': outcome_map[prediction],
            'confidence': max(probabilities),
            'probabilities': {
                outcome_map[classes[i]]: prob for i, prob in enumerate(probabilities)
            }
        }
        
        return result
    
    def get_feature_importance(self):
        """
        Get feature importance from the trained model
        """
        if self.model is None:
            return None
        
        importance_df = pd.DataFrame({
            'feature': self.feature_columns,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        return importance_df

# Example usage and this week's predictions
def main():
    print("🏆 Premier League ML Prediction System")
    print("=" * 50)
    
    # Initialize predictor
    predictor = PremierLeaguePredictor()
    
    # Create and load historical data
    print("📊 Loading historical data...")
    historical_data = predictor.create_sample_historical_data()
    
    # Prepare data and train model
    print("🤖 Training ML model...")
    X, y = predictor.prepare_data(historical_data)
    predictor.train_model(X, y)
    
    # Show feature importance
    print("\n📈 Most Important Features:")
    importance_df = predictor.get_feature_importance()
    print(importance_df.head(10))
    
    # Example predictions for this week's matches
    print("\n⚽ This Week's Match Predictions:")
    print("-" * 40)
    
    # Sample matches (you'd replace with actual fixture data)
    this_weeks_matches = [
        {
            'home_team': 'Arsenal',
            'away_team': 'Liverpool',
            'home_strength': 82,
            'away_strength': 80,
            'home_recent_form': 0.8,
            'away_recent_form': 0.6,
            'home_goals_scored_avg': 2.1,
            'away_goals_scored_avg': 1.9,
            'home_goals_conceded_avg': 1.0,
            'away_goals_conceded_avg': 1.1,
            'home_win_percentage': 0.65,
            'away_win_percentage': 0.60,
            'head_to_head_home': 4,
            'head_to_head_away': 6,
            'home_injury_count': 2,
            'away_injury_count': 1,
            'days_since_last_match_home': 7,
            'days_since_last_match_away': 7
        },
        {
            'home_team': 'Man City',
            'away_team': 'Chelsea',
            'home_strength': 85,
            'away_strength': 75,
            'home_recent_form': 1.2,
            'away_recent_form': 0.2,
            'home_goals_scored_avg': 2.3,
            'away_goals_scored_avg': 1.6,
            'home_goals_conceded_avg': 0.9,
            'away_goals_conceded_avg': 1.3,
            'home_win_percentage': 0.75,
            'away_win_percentage': 0.50,
            'head_to_head_home': 7,
            'head_to_head_away': 3,
            'home_injury_count': 1,
            'away_injury_count': 3,
            'days_since_last_match_home': 7,
            'days_since_last_match_away': 7
        }
    ]
    
    for i, match in enumerate(this_weeks_matches, 1):
        home_team = match['home_team']
        away_team = match['away_team']
        
        prediction = predictor.predict_match(home_team, away_team, match)
        
        print(f"{i}. {home_team} vs {away_team}")
        print(f"   Prediction: {prediction['predicted_outcome']}")
        print(f"   Confidence: {prediction['confidence']:.1%}")
        print(f"   Probabilities:")
        for outcome, prob in prediction['probabilities'].items():
            print(f"     {outcome}: {prob:.1%}")
        print()

if __name__ == "__main__":
    main()