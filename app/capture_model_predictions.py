import pandas as pd

from car_price_model.predict import make_prediction
from config.config import TARGET
from app.config import DATASET_PATH


def capture_predictions() -> None:
    """Save the test data predictions to a CSV."""
    
    # load test data 
    test_dat = pd.read_csv(DATASET_PATH)
    
    # input data 
    input_data =  test_dat.drop(columns=[TARGET])
    input_data = input_data.astype("object").where(pd.notna(input_data),None)
    
    
    # prediction 
    results = make_prediction(input_data= input_data)
    
    if results["errors"] is not None :
        raise ValueError(results["errors"])
    
    # convert predictions to DataFrame 
    predictions_df = pd.DataFrame(results)
    
    # Define Save Path 
    save_path = DATASET_PATH.parent / "test_data_predictions.csv"
    
    # convert Dataframe to csv file 
    predictions_df.to_csv(save_path , index= False)   # Baseline = Reference  
    
    
if __name__ == "__main__" : 
    capture_predictions()   