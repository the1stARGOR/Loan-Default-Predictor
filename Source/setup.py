import pandas as pd

def read_excel_data(file_path, sheet_names):
    data_frames = []
    
    # Read each sheet from the Excel file and store it in a list of DataFrames
    for sheet_name in sheet_names:
        data_frame = pd.read_excel(file_path, sheet_name=sheet_name, engine='openpyxl')
        data_frames.append(data_frame)
    
    return data_frames





