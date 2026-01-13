import pandas as pd
from etl.transform import transform

if __name__ == "__main__":
    # Run extraction when executed directly
    churn_dt, metadata = extract()
    print("Extraction complete. DataFrame shape:", churn_dt.shape)
    print("Metadata:", json.dumps(metadata, indent=2))
    # Run transformation inspection
    print("\nInspecting extracted data schema:")
    cleaned_df, schema_info = transform(churn_dt)   
    print("Transformation complete. Cleaned DataFrame shape:", cleaned_df.shape)
    print("Schema Info:", json.dumps(schema_info, indent=2)) 
    cleaned_df.head()
    
