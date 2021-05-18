import pandas as pd

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
        return None


def create_features(json_payload):
    manufacturing_region = json_payload["manufacturing_region"]
    manufacturing_location_code = json_payload["manufacturing_location_code"]
    customer_industry = json_payload["customer_industry"]
    product_family = json_payload["product_family"]
    product_group = json_payload["product_group"]
    make_vs_buy = json_payload["make_vs_buy"]
    top_customer_group = json_payload["top_customer_group"]
    customer_region = json_payload["customer_region"]
    customer_first_invoice_date = json_payload["customer_first_invoice_date"]
    ordered_qty = float(json_payload["ordered_qty"])
    gm = float(json_payload["gm"])

    if manufacturing_region is None and manufacturing_location_code is not None:
        if manufacturing_location_code in manufacturing_dict.keys():
            manufacturing_region = manufacturing_dict[manufacturing_location_code]

    if product_family is None and product_group is not None:
        if product_group in product_dict.keys():
            product_family = product_dict[product_group]

    if make_vs_buy is not None:
        make_vs_buy = "MAKE" if make_vs_buy in make_cols else "BUY"

    if top_customer_group == "STAR":
        customer_region = "STAR"

    new_old_customer = "NEW" if int(customer_first_invoice_date.split("-")[0]) >= 2015 else "OLD"

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

    return pd.DataFrame(prediction_row, index=[0])


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
                    "invoiced_price": "2.7800", "invoiced_price_tx": "2.7800", "cost_of_part": ".0000",
                    "material_cost_of_part": ".0000", "labor_cost_of_part": ".0000", "overhead_cost_of_part": ".0000",
                    "gm": "1.00000000000", "num_of_unique_products_on_a_quote": "6"}

    df = create_features(json_payload)
    print(df)
