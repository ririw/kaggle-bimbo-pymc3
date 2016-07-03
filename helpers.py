"""
This script is intended to translate the data in `csv` format to `sqlite db` while fixing the field type and renaming the columns.
The idea is to define some directives and let `odo` takes care of the rest ;-). The script will register the `train` and `test` data
into tables with the same respective name.
"""
import dask.dataframe as dd
from dask.diagnostics import ProgressBar
from odo import odo, dshape

def prepare_database():
    # Train set
    df = dd.read_csv("./train.csv")
    
    col_map = dict(zip(df.columns, ["week_num","agency_id","channel_id","route_id","client_id","product_id",
                                    "sales_unit","sales_peso","returns_unit","returns_peso","adjusted_demand"]))
    
    ds = dshape("var * {week_num:int64,agency_id:int64,channel_id:int64,route_id:int64,client_id:int64,product_id:int64,\
                sales_unit:int64,sales_peso:float64,returns_unit:int64,returns_peso:float64, adjusted_demand:int64}")
    
    df = df.rename(columns=col_map)
    
    print("translating the train set...")
    with ProgressBar():
        odo(df, "sqlite:///data.sqlite3::train", dshape=ds) # the dirty part
    
    # Test set
    df = dd.read_csv("./test.csv", usecols=range(1,7)) # discard the `id` (first) column
    
    col_map = dict(zip(df.columns, ["week_num","agency_id","channel_id","route_id","client_id","product_id"]))
    
    ds = dshape("var * {week_num:int64,agency_id:int64,channel_id:int64,route_id:int64,client_id:int64,product_id:int64}")
    
    df = df.rename(columns=col_map)
    
    print("translating the test set...")
    with ProgressBar():
        odo(df, "sqlite:///data.sqlite3::test", dshape=ds)

prepare_database()