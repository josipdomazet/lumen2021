import pandas as pd
import numpy as np

manufacturing_dict = {'N7': 'North America',
                      'N13': 'Asia',
                      'N12': 'Asia',
                      'N1': 'North America',
                      'B6': 'North America',
                      'B1': 'North America',
                      'B70': 'Asia',
                      'B77': 'Asia',
                      'B3': 'North America',
                      'N8': 'Europe',
                      'K1': 'Asia',
                      'N4': 'North America',
                      'N2': 'North America',
                      'L1': 'North America',
                      'B20': 'North America',
                      'B21': 'North America',
                      'B76': 'Asia',
                      'L2': 'North America',
                      'N15': 'North America',
                      'C3': 'Asia',
                      'T1': 'Asia',
                      'N10': 'Asia',
                      'N6': 'Europe',
                      'L4': 'North America',
                      'K2': 'Asia',
                      'C5': 'Asia',
                      'B8': 'North America',
                      'S1': 'Europe'}

product_dict = {'PC026': 'PF002',
                'SF002': 'PF002',
                'PC013': 'PF002',
                'PC019': 'PF002',
                'PC002': 'PF002',
                'PC014': 'PF002',
                'PC029': 'PF002',
                'PC004': 'PF002',
                'PC023': 'PF001',
                'PC021': 'PF001',
                'PC009': 'PF001',
                'PC010': 'PF001',
                'PC001': 'PF001',
                'PC017': 'PF002',
                'PC003': 'PF002',
                'PC012': 'PF002',
                'PC005': 'PF001',
                'PC022': 'PF002',
                'PC016': 'PF001',
                'PC011': 'PF001',
                'PC025': 'PF001',
                'PC015': 'PF001',
                'PC008': 'PF001',
                'PC007': 'PF002',
                'PC018': 'PF001',
                'PC030': 'PF002',
                'PC020': 'PF001',
                'PC006': 'PF001',
                'PC028': 'PF002',
                'SF001': 'PF002',
                'PC024': 'PF002'}

make_cols = ["MANUFACTURED", "RAW MATERIAL", "FINISHED GOODS"]
not_allowed_make_vs_buy = ["RAW MATERIAL", "BUY - CUST. SUPPLIED", "BUY - INTERPLNT TRNS", "PURCHASED (RAW)"]


def determine_bucket(ordered_qty):
    labels = ["[1, 10]", "(10, 100]", "(100, 1000]", "(1000, 10000]", "(10000, inf)"]
    if 1 <= ordered_qty <= 10:
        return labels[0]
    elif 10 < ordered_qty <= 100:
        return labels[1]
    elif 100 < ordered_qty <= 1000:
        return labels[2]
    elif 1000 < ordered_qty <= 10000:
        return labels[3]
    elif ordered_qty > 10000:
        return labels[4]
    else:
        return np.nan


def check_is_missing(variable, type="numerical"):
    if variable == "" or variable == "NA" or variable == "NAN" or variable == "NaN":
        return np.nan
    else:
        if type == "numerical":
            return float(variable)
        else:
            return variable


def check_preconditions(cost_of_part, invoiced_price, invoiced_qty_shipped, ordered_qty, intercompany, customer_id,
                        make_vs_buy, gm):
    if cost_of_part <= 0.0:
        return True, "Cost of part cannot be <= 0!"
    if invoiced_price <= 0:
        return True, "Invoiced price cannot be <= 0!"
    if invoiced_qty_shipped <= 0:
        return True, "Invoiced quantity shipped cannot be <= 0!"
    if ordered_qty <= 0:
        return True, "Ordered quantity cannot be <= 0!"
    if intercompany == "YES":
        return True, "Intercompany cannot be 'YES'!"
    if customer_id < 0:
        return True, "Customer ID cannot be negative!"
    if make_vs_buy in not_allowed_make_vs_buy:
        return True, make_vs_buy + " value not allowed!"
    if gm >= 1.0 or gm <= 0.0:
        return True, "Gross margin can only be defined in range (0,1)!"
    return False, ""


def create_features(json_payload):
    manufacturing_region = check_is_missing(json_payload["manufacturing_region"], type="other")
    manufacturing_location_code = check_is_missing(json_payload["manufacturing_location_code"], type="other")
    customer_industry = check_is_missing(json_payload["customer_industry"], type="other")
    product_family = check_is_missing(json_payload["product_family"], type="other")
    product_group = check_is_missing(json_payload["product_group"], type="other")
    make_vs_buy = check_is_missing(json_payload["make_vs_buy"], type="other")
    top_customer_group = check_is_missing(json_payload["top_customer_group"], type="other")
    customer_region = check_is_missing(json_payload["customer_region"], type="other")
    customer_first_invoice_date = check_is_missing(json_payload["customer_first_invoice_date"], type="other")
    ordered_qty = check_is_missing(json_payload["ordered_qty"])
    gm = check_is_missing(json_payload["gm"])
    cost_of_part = check_is_missing(json_payload["cost_of_part"])
    invoiced_price = check_is_missing(json_payload["invoiced_price"])
    invoiced_qty_shipped = check_is_missing(json_payload["invoiced_qty_shipped"])
    intercompany = check_is_missing(json_payload["intercompany"], type="other")
    customer_id = check_is_missing(json_payload["customer_id"])

    fail, message = check_preconditions(cost_of_part, invoiced_price, invoiced_qty_shipped, ordered_qty, intercompany,
                                        customer_id, make_vs_buy, gm)

    if fail:
        return None, message

    if manufacturing_region is np.nan and manufacturing_location_code is not np.nan:
        if manufacturing_location_code in manufacturing_dict.keys():
            manufacturing_region = manufacturing_dict[manufacturing_location_code]

    if product_family is np.nan and product_group is not np.nan:
        if product_group in product_dict.keys():
            product_family = product_dict[product_group]

    if make_vs_buy is not np.nan:
        make_vs_buy = "MAKE" if make_vs_buy in make_cols else "BUY"

    if top_customer_group == "STAR":
        customer_region = "STAR"

    if customer_first_invoice_date is not np.nan:
        new_old_customer = "NEW" if int(customer_first_invoice_date.split("-")[0]) >= 2015 else "OLD"
    else:
        new_old_customer = np.nan

    ordered_qty_bucket = determine_bucket(ordered_qty)

    prediction_row = {
        "manufacturing_region": manufacturing_region,
        "customer_region": customer_region,
        "customer_industry": customer_industry,
        "product_family": product_family,
        "make_vs_buy": make_vs_buy,
        "new_old_customer": new_old_customer,
        "ordered_qty_bucket": ordered_qty_bucket,
        "gm": gm,
    }

    return pd.DataFrame(prediction_row, index=[0]).loc[0], "success"


# Test
if __name__ == "__main__":
    json_payload = {"manufacturing_region": "Asia", "manufacturing_location_code": "N13", "intercompany": "NO",
                    "customer_id": "224307", "customer_industry": "IC000", "customer_region": "Asia",
                    "customer_first_invoice_date": "2011-05-27 00:00:00", "top_customer_group": "OTHER",
                    "item_code": "343372", "product_family": "PF002", "product_group": "SF002",
                    "price_last_modified_date_in_the_erp": "", "born_on_date": "2014-12-09",
                    "make_vs_buy": "MANUFACTURED", "sales_channel_internal": "230", "sales_channel_external": "230",
                    "sales_channel_grouping": "", "invoice_date": "2018-09-29", "invoice_num": "16964668",
                    "invoice_line_num": "18581670", "order_date": "2018-07-06", "order_num": "513470",
                    "order_line_num": "15736930", "invoiced_qty_shipped": "4000.000000", "ordered_qty": "4000.000000",
                    "invoiced_price": "2.7800", "invoiced_price_tx": "2.7800", "cost_of_part": "1.0000",
                    "material_cost_of_part": ".0000", "labor_cost_of_part": ".0000", "overhead_cost_of_part": ".0000",
                    "gm": "0.25", "num_of_unique_products_on_a_quote": "6"}

    df = create_features(json_payload)[0]
    message = create_features(json_payload)[1]
    print(df)
    print(message)