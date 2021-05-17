import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from dataframe import *
from dataprep.eda import create_report

PLOTS_PATH = "./item-code-in-manufacturing-region--gm"
if not os.path.exists(PLOTS_PATH):
    os.mkdir(PLOTS_PATH)

# del df[sales_channel_grouping]
# report = create_report(df, title="Lumen 2021 dataset report")
# report.save(filename="Lumen-2021-dataset-report", to="./")

# threshold = 0.1
# not_zero = df[np.abs(df[invoiced_price]) > threshold]
# diff = (not_zero[invoiced_price] - not_zero[cost_of_part]) / not_zero[invoiced_price] - not_zero[gm]
# not_matching = np.abs(diff) > threshold
# result = not_zero[not_matching]

# for threshold in [1e-7, 0.001, 0.01, 0.1, 1, 2, 5, 10, 100]:
#    diff = df[cost_of_part] - (df[material_cost_of_part] + df[labor_cost_of_part] + df[overhead_cost_of_part])
#    result = df[np.abs(diff) > threshold]
#    print(f"threshold = {threshold}, count = {result[cost_of_part].count()}")


def plot_group_by(group_by_column, value_column, outlier_min=float("+inf")):
    result = (df[abs(df[value_column]) < outlier_min]
              [[group_by_column, value_column]]
              .groupby(group_by_column)
              .filter(lambda g: len(g) > 10)
              .groupby(group_by_column))

    df2 = pd.DataFrame({col: vals[value_column] for col, vals in result})
    medians = df2.median()
    medians.sort_values(ascending=False, inplace=True)
    df2 = df2[medians.index]

    title = f"Boxplot {value_column.replace('_', ' ')} by {group_by_column.replace('_', ' ')}"
    df2.boxplot(rot=90, showfliers=False)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(f"{PLOTS_PATH}/{title.lower().replace(' ', '-')}.png")
    plt.close()


value_column = gm
groups = (df[abs(df[gm]) < 1.5]
          [[manufacturing_region, item_code, value_column]]
          .groupby([manufacturing_region, item_code])
          .mean()
          .groupby(item_code))

for name, result in groups:
    result = result.groupby(manufacturing_region)
    if len(result) < 3: continue
    result = result.mean().sort_values(by=value_column, ascending=False)

    result.plot.bar(rot=90)
    plt.title(name + " (mean)")
    plt.tight_layout()
    plt.savefig(f"{PLOTS_PATH}/{name}.png")
    plt.close()

exit()



def plot_group_by_fast(group_by_column, value_column, outlier_min=float("+inf")):
    result = (df[abs(df[value_column]) < outlier_min]
              [[group_by_column, value_column]]
              .groupby(group_by_column)
              .filter(lambda g: len(g) > 10)
              .groupby(group_by_column)
              .median()
              .sort_values(by=value_column, ascending=False)
              .reset_index()
              .head(n=10))

    title = f"Bar {value_column.replace('_', ' ')} by {group_by_column.replace('_', ' ')}"
    result.plot.bar(x=group_by_column, rot=90)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(f"{PLOTS_PATH}/{title.lower().replace(' ', '-')}.png")
    plt.close()


diff_key = "diff_invoice_order_date"
df[diff_key] = (df.invoice_date - df.price_last_modified_date_in_the_erp).dt.days
df = df[(abs(df[diff_key]) < 5000) & (abs(df[diff_key]) > 1000)]
df = df[abs(df[gm]) < 2]

plt.scatter(df[diff_key], df[gm], facecolors='none', edgecolors='r')
plt.title("invoice-last date diff ~ GM")
plt.savefig(f"{PLOTS_PATH}/gm-invoice-last-diff.png")
exit()

df.info()
after = (df[invoice_date] > df[price_last_modified_date_in_the_erp]).sum()
on = (df[invoice_date] == df[price_last_modified_date_in_the_erp]).sum()
before = (df[invoice_date] < df[price_last_modified_date_in_the_erp]).sum()
x = ["after", "on", "before"]
y = [after, on, before]
color = ["green", "orange", "red"]

plt.bar(x, y, color=color)
plt.title("Invoice date after/on/before last date modified price")
plt.savefig(f"{PLOTS_PATH}/last-modified.png")

print(f"Invoice date after change date: {after}")
print(f"Invoice date on change date: {on}")
print(f"Invoice date before change date: {before}")
exit()

show_in_browser(df[df.item_code == "000009"])
exit()
uniques_by_group = df.groupby(item_code)[[price_last_modified_date_in_the_erp]].nunique()
uniques_by_group = uniques_by_group[uniques_by_group[price_last_modified_date_in_the_erp] > 1]
show_in_browser(uniques_by_group)
exit()


GROUP_BY_COLUMNS = [
    manufacturing_region,
    manufacturing_location_code,
    top_customer_group,
    product_family,
    product_group,
    intercompany,
    customer_region,
    customer_industry,
    make_vs_buy
]

VALUE_COLUMNS_OUTLIER = [
    (gm, 1),
    (invoiced_price, 5000),
    (cost_of_part, 5000)
]

for value_column, outlier_min in VALUE_COLUMNS_OUTLIER:

    total = len(df)
    gm_not_nan = df[value_column].notna().sum()
    gm_not_outlier = (abs(df[value_column]) < outlier_min).sum()
    plt.bar(["Total", "Not NaN", "Not outlier"], [total, gm_not_nan, gm_not_outlier])
    plt.title(f"{value_column} counts (outlier min = {outlier_min})")
    plt.tight_layout()
    plt.savefig(f"{PLOTS_PATH}/{value_column}-counts.png")
    plt.close()

    for group_by_column in GROUP_BY_COLUMNS:
        plot_group_by(group_by_column, value_column, outlier_min)
