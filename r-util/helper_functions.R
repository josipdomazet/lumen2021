plot_one_factor <- function(tbl, var) {
  tbl %>%
    count(var = fct_explicit_na({{ var }})) %>% 
    mutate(var = fct_reorder(var, n)) %>% 
    ggplot(aes(n, var, fill = var)) + 
    geom_col(show.legend = FALSE) +
    scale_x_continuous(labels = scales::comma) 
}

plot_outer_inner_factor <- function(tbl, outer_var, inner_var) {
  tbl %>% 
    rename(outer_var = {{ outer_var }}, inner_var = {{ inner_var }}) %>% 
    group_by(outer_var, inner_var) %>% 
    tally() %>% 
    ungroup() %>% 
    mutate(outer_var = fct_reorder(outer_var, -n)) %>% 
    mutate(inner_var = reorder_within(inner_var,
                                      n,
                                      outer_var)) %>% 
    ggplot(aes(n, inner_var, fill = outer_var)) +
    geom_col() +
    facet_wrap(~ outer_var, scales = "free_y") +
    scale_y_reordered() +
    guides(fill = FALSE)
}

clean <- function(tbl) {
  tbl %>% filter(cost_of_part > 0 &
                   invoiced_price > 0 &
                   gm <= 1 &
                   ordered_qty > 0 &
                   invoiced_qty_shipped > 0) 
  
}

filter_var_percent <- function(tbl, var, percent) {
  tbl %>% filter(between({{ var }},
                         quantile({{ var }}, percent, na.rm = TRUE), 
                         quantile({{ var  }}, 1 - percent, na.rm =  TRUE)))
}


