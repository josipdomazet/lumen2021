import numpy as np
import pandas as pd
pd.set_option("display.max_columns", None)

DATASET_PATH = "../dataset/LUMEN_DS.csv"
ENCODING = "UTF-16"
SEPARATOR = "|"
NA_VALUES = "NaN"
ROWS_LIMIT = None
IMPUTE_COST_OF_PART = True

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
    order_line_num,
]

DATE_ONLY_COLUMNS = [born_on_date, invoice_date, order_date]

DATETIME_COLUMNS = [price_last_modified_date_in_the_erp, customer_first_invoice_date]

INT_COLUMNS = [num_of_unique_products_on_a_quote]

FLOAT_COLUMNS = [
    invoiced_qty_shipped,
    ordered_qty,
    invoiced_price,
    invoiced_price_tx,
    cost_of_part,
    material_cost_of_part,
    labor_cost_of_part,
    overhead_cost_of_part,
    gm,
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


if IMPUTE_COST_OF_PART:
    df[cost_of_part][df[cost_of_part] == 0] = None 
    df.sort_values(by=[item_code, invoice_date], inplace=True)
    df[cost_of_part] = df.groupby(by=[item_code]).cost_of_part.fillna(method='ffill')
    df[cost_of_part][df[cost_of_part].isna()] = 0
    df[gm] = (df[invoiced_price] - df[cost_of_part]) / df[invoiced_price]


df_manufacturing = df[
    ~df.manufacturing_region.isna() & ~df.manufacturing_location_code.isna()
]

manufacturing_dict = dict(
    zip(
        df_manufacturing.manufacturing_location_code,
        df_manufacturing.manufacturing_region,
    )
)

df[manufacturing_region] = df.apply(
    lambda x: manufacturing_dict[x[manufacturing_location_code]]
    if x[manufacturing_location_code] in manufacturing_dict.keys()
    else x[manufacturing_region],
    axis=1,
)


preconditions = (
    (df.cost_of_part > 0)
    & (df.invoiced_price > 0)
    & (df.gm < 1)
    & (df.gm > 0)
    & (df.ordered_qty > 0)
    & (df.invoiced_qty_shipped > 0)
    & (df.intercompany == "NO")
    & (df.customer_id != "-99")
    & (df.make_vs_buy != "RAW MATERIAL")
    & (df.make_vs_buy != "BUY - CUST. SUPPLIED")
    & (df.make_vs_buy != "BUY - INTERPLNT TRNS")
    & (df.make_vs_buy != "PURCHASED (RAW)")
)

df = df[preconditions].reset_index(drop=True)


make_cols = ["MANUFACTURED", "RAW MATERIAL", "FINISHED GOODS"]
buy_cols = [
    "BUY",
    "BUY - IMPORTED",
    "BUY - LOCAL",
    "BUY - CUST. SUPPLIED",
    "BUY - INTERPLNT TRNS",
    "PURCHASED",
    "PURCHASED (RAW)",
]

def relabel_make_vs_buy(x):
    if x in make_cols:
        return "MAKE"
    elif x in buy_cols:
        return "BUY"
    else:
        return x

df[make_vs_buy] = df.make_vs_buy.apply(lambda x: relabel_make_vs_buy(x))


def relabel_customer_region(region, group):
    if group == "STAR":
        return "STAR"
    else:
        return region

df[customer_region] = df.apply(
    lambda x: relabel_customer_region(x[customer_region], x[top_customer_group]), axis=1
)


new_old_customer = "new_old_customer"
df[new_old_customer] = df.customer_first_invoice_date.apply(lambda x: "NEW" if x.year >= 2015 else "OLD")


ordered_qty_bucket = "ordered_qty_bucket"
bounds = [0, 10, 100, 1_000, 10_000, float("+inf")]
labels = ['[1, 10]', '(10, 100]', '(100, 1000]', '(1000, 10000]', '(10000, inf)']
df[ordered_qty_bucket] = pd.cut(df[ordered_qty], bounds, labels=labels)


cols_to_remove = [
    customer_id,
    manufacturing_location_code,
    customer_first_invoice_date,
    price_last_modified_date_in_the_erp,
    born_on_date,
    invoice_date,
    order_date,
    invoiced_price_tx,
    invoiced_price,
    cost_of_part,
    sales_channel_internal,
    sales_channel_external,
    sales_channel_grouping,
    top_customer_group,
    invoice_num,
    invoice_line_num,
    order_num,
    order_line_num,
    material_cost_of_part,
    labor_cost_of_part,
    overhead_cost_of_part,
    item_code,
    intercompany,
    product_group,
    ordered_qty,
    invoiced_qty_shipped,
    num_of_unique_products_on_a_quote,
]

df = df.drop(cols_to_remove, axis=1)
