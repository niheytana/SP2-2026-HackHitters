import pandas as pd
import numpy as np

class MyModel:
    def __init__(self):
        self.global_avg = 50
        self.venue_avg = {}
        self.team_avg = {}
        self.player_mapping = {}  

    def fit(self, deliveries_df, players_df=None, matches_df=None):
       
        if players_df is not None and not players_df.empty:
            players_df.columns=players_df.columns.str.strip()

            id_col = None
            name_col = None

            for col in players_df.columns:
                if col.lower()=="id":
                    id_col=col
                if "name" in col.lower():
                    name_col=col
            
        
            if id_col and name_col:
                self.player_mapping = dict(
                    zip(players_df[id_col].astype(str),
                        players_df[name_col])
                )

       
        if deliveries_df is None:
            return self

        df = deliveries_df.copy()
        df.columns=df.columns.str.strip().str.lower()
        

        if matches_df is not None and not matches_df.empty:
            matches_df=matches_df.copy()
            matches_df.columns=matches_df.columns.str.strip().str.lower() 

            df = df.merge(
                matches_df[["matchid", "venue"]],
                on="matchid",
                how="left"
            )
        
        df["venue"] = df["venue"].str.strip().str.lower()
        df["batting_team"] = df["batting_team"].str.strip().str.lower()
       
        pp_df = df[df["over"] < 6].copy()

        
        pp_df["total_runs"] = pp_df["batsman_runs"] + pp_df["extras"]

        
        inning_totals = pp_df.groupby(
            ["matchid", "inning", "venue", "batting_team"]
        )["total_runs"].sum().reset_index()

        
        self.global_avg = int(inning_totals["total_runs"].mean())

        
        self.venue_avg = inning_totals.groupby(
            ["venue", "inning"]
        )["total_runs"].mean().to_dict()

        

        
        self.team_avg = inning_totals.groupby(
            ["batting_team", "inning"]
        )["total_runs"].mean().to_dict()

        self.batsman_avg = pp_df.groupby("batsman")["batsman_runs"].mean().to_dict()
        self.bowler_avg = pp_df.groupby("bowler")["total_runs"].mean().to_dict()

        

        return self

    def predict(self, test_df):
        
        test_df=test_df.copy()
        test_df.columns=test_df.columns.str.strip().str.lower()
        test_df["venue"]=test_df["venue"].str.strip().str.lower()
        test_df["batting_team"]=test_df["batting_team"].str.strip().str.lower()

        predictions = []

        for _, row in test_df.iterrows():

            venue = row.get("venue")
            team = row.get("batting_team")
            inning = row.get("innings")

           
            
            batsman_ids = [s.strip() for s in str(row.get("batsman's player id", "")).split(",")]
            bowler_ids = [s.strip() for s in str(row.get("bowler's player id (opponent)", "")).split(",")]

            

            batsman_scores = [self.batsman_avg.get(self.player_mapping.get(bid, ""), self.global_avg) for bid in batsman_ids]
            bowler_scores = [self.bowler_avg.get(self.player_mapping.get(bid, ""), self.global_avg) for bid in bowler_ids]


            batsman_score = sum(batsman_scores) / len(batsman_scores)
            bowler_score = sum(bowler_scores) / len(bowler_scores)
            
            venue_score = self.venue_avg.get((venue, inning),
                                             self.global_avg)

            team_score = self.team_avg.get((team, inning),
                                           self.global_avg)

            predicted_score = int(
                0.5 * venue_score +
                0.3 * team_score +
                0.3 * batsman_score +
                0.1 * bowler_score
            )

            predictions.append({
                "id": row["id"],
                "predicted_score": predicted_score
            })

        return pd.DataFrame(predictions)