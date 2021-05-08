import webbrowser

import numpy as np
import pandas as pd
pd.set_option("display.max_columns", None)

def show_in_browser(dataframe, file_name="result.html"):
    dataframe.to_html(file_name)
    webbrowser.open(file_name)

pd.DataFrame.show_in_browser = show_in_browser

DATASET_PATH = "./LUMEN_DS.csv"
ENCODING = "UTF-16"
SEPARATOR = "|"
NA_VALUES = "NaN"
ROWS_LIMIT = None  

manufacturing_region = "manufacturing_region"
manufacturing_location_code = "manufacturing_location_code"
intercompany = "intercompany"
customer_id = "customer_id"
customer_industry = "customer_industry"
customer_region = "customer_region"
customer_first_invoice_date = "customer_first_invoice_date"
top_customer_group = "top_customer_group"
item_code = "item_code"
product_family = "product_family"
product_group = "product_group"
price_last_modified_date_in_the_erp = "price_last_modified_date_in_the_erp"
born_on_date = "born_on_date"
make_vs_buy = "make_vs_buy"
sales_channel_internal = "sales_channel_internal"
sales_channel_external = "sales_channel_external"
sales_channel_grouping = "sales_channel_grouping"
invoice_date = "invoice_date"
invoice_num = "invoice_num"
invoice_line_num = "invoice_line_num"
order_date = "order_date"
order_num = "order_num"
order_line_num = "order_line_num"
invoiced_qty_shipped = "invoiced_qty_shipped"
ordered_qty = "ordered_qty"
invoiced_price = "invoiced_price"
invoiced_price_tx = "invoiced_price_tx"
cost_of_part = "cost_of_part"
material_cost_of_part = "material_cost_of_part"
labor_cost_of_part = "labor_cost_of_part"
overhead_cost_of_part = "overhead_cost_of_part"
gm = "gm"
num_of_unique_products_on_a_quote = "num_of_unique_products_on_a_quote"

STRING_COLUMNS = [
    manufacturing_region,
    manufacturing_location_code,
    intercompany,
    customer_id,
    customer_industry,
    customer_region,
    top_customer_group,
    item_code,
    product_family,
    product_group,
    make_vs_buy,
    sales_channel_internal,
    sales_channel_external,
    sales_channel_grouping,
    invoice_num,
    invoice_line_num,
    order_num,
    order_line_num
]

DATE_ONLY_COLUMNS = [
    born_on_date,
    invoice_date,
    order_date
]

DATETIME_COLUMNS = [
    price_last_modified_date_in_the_erp,
    customer_first_invoice_date
]

INT_COLUMNS = [
    num_of_unique_products_on_a_quote
]

FLOAT_COLUMNS = [
    invoiced_qty_shipped,
    ordered_qty,
    invoiced_price,
    invoiced_price_tx,
    cost_of_part,
    material_cost_of_part,
    labor_cost_of_part,
    overhead_cost_of_part,
    gm
]

DATE_COLUMNS = DATE_ONLY_COLUMNS + DATETIME_COLUMNS
NUMERIC_COLUMNS = INT_COLUMNS + FLOAT_COLUMNS
ALL_COLUMNS = STRING_COLUMNS + DATE_COLUMNS + NUMERIC_COLUMNS

df = pd.read_csv(DATASET_PATH, sep=SEPARATOR, encoding=ENCODING, nrows=ROWS_LIMIT)

for col in STRING_COLUMNS:
    df[col] = df[col].astype("str")
    df[col] = df[col].replace("nan", np.nan, regex=True)

for col in DATE_COLUMNS:
    df[col] = pd.to_datetime(df[col], errors="coerce")

del df[invoiced_price_tx]
price_over_cost = "price_over_cost"
df[price_over_cost] = df[invoiced_price] / df[cost_of_part]

sales_channel = "sales_channel"
df[sales_channel] = df[sales_channel_internal]
del df[sales_channel_internal]
del df[sales_channel_external]
del df[sales_channel_grouping]

preconditions = ((df.cost_of_part > 0) & 
                 (df.invoiced_price > 0) & 
                 (df.gm <= 1) & 
                 (df.gm >= 0) & 
                 (df.ordered_qty > 0) &
                 (df.invoiced_qty_shipped > 0))
df = df[preconditions].reset_index(drop=True)
