# ====================================================================================
# Conditional Acceptability in Large Language Models and Humans – Data Analysis
# ====================================================================================



# ----------------------
# Package Loading
# ----------------------

# Data handling
library(dplyr)        # For filtering, mutating or grouping data
library(tidyr)        # For reshaping dataframes (wide -> long)

# Modelling
library(lme4)         # Main package for linear mixed-effects models
library(lmerTest)     # For p-values using Kenward-Roger
library(car)          # For Type III Anova
library(emmeans)      # For estimated marginal means (EMMs) and slope estimation
library(psych)        # For descriptive statistics (describe, describeBy)

# Plotting
library(effects)      # For visualizing model effects
library(lattice)      # Base plotting system (used by effects)
library(latticeExtra) # Enhancements to lattice
library(ggplot2)      # For Plotting


# ================================================
# Overview of this script
# ================================================


# This script is is organized into three main parts:


# 1. Per-Model Analysis
#   - Reads individual CSV files for each respondent (e.g., human or LLM)
#   - Prepares and centers relevant variables
#   - Fits a linear mixed-effects model (LMM) per respondent
#   - Saves model results as .rda files for reuse


# 2. Combined Comparison
#   - Loads the saved .rda model results
#   - Implements a combined linear mixed-effects model to examine conditional probability estimates
#   - Implements the main combined linear mixed-effects model to compare behavior across respondents


# 3. Plotting:
#    - Visualizes selected results from Part 1 and Part 2




# ================================================
# Part 1: Per-Model Analysis
# ================================================



# -- 1.1: Utility Functions --

# Convert dataframes from wide to long format
#   Collapse if-statement probability and acceptability ratings into one column, and
#   Add another column with readable labels to signal the metric (P(If A, then B) vs. A(If A, then B))
#   -> This makes sure that both judgments types can be analyzed jointly as well as separately
reshape_data <- function(df) {
  df %>%
    pivot_longer(cols = c("if_prob", "if_acc"),            # select these two columns
                 names_to = "judgement_type",              # store their original column names (if_prob, if_acc) in a new column
                 values_to = "judgement") %>%              # store their values (e,g., 100, 98, 55) in another new column
    mutate(
      metric = case_when(
        judgement_type == "if_prob" ~ "P(If A, then B)",   # map judgment types to more readable labels for use in plots and output tables
        judgement_type == "if_acc" ~ "A(If A, then B)"
      )
    )
}


# Convert selected variables to factors (categorical variables)
#   This ensures they are treated as groupings in mixed models
#   Without doing this, R may assume that e.g., scenarios 1-12 pertain to numerical values 1-12
#   -> safeguard to make sure instances are treated the way we want them to be
ensure_factoring <- function(df) {
  df$instance_id <- as.factor(df$instance_id)           # factor instance ID, scenario number, relation type, and metric
  df$scenario_number <- as.factor(df$scenario_number)
  df$relation_type <- as.factor(df$relation_type)
  df$metric <- as.factor(df$metric)
  return(df)
}


# Center conditional probability and if-statement judgment values
#   0 -> -0.5, 50 -> 0, 100 -> 0.5
#   In modeling, this improves interpretability and convergence
#   And it is also required by some modeling functions (e.g., emmeans)
center_data <- function(dw) {
  dw %>%
    mutate(
      c_prob = (c_prob - 50) / 100,
      judgement = (judgement - 50) / 100
    )
}



# -- 1.2: Run Individual Model Analysis --

# Main function for per-model analysis
#   Reads data, processes it, fits a linear mixed model, and saves results to .rda file


# Arguments:
#   data_filename: path to input .csv file
#   source_label: label for this data source (respondent, e.g., "Human", "Llama 8B (vanilla)")
#   output_filename: path to save the model results as .rda

run_individual_analysis <- function(data_filename, source_label, output_filename) {
  
  # Only run the analysis if the file does not exist yet
  # Because running the analysis can be quite computationally expensive,
  # We don't want to risk unnecessary re-computing
  if (!file.exists(output_filename)) {
    message(paste("Running analysis for: ", source_label))
    
    # === Preprocessing ===
    # Read csv files and call all utility functions from above (reshaping, factoring, and centering)
    df <- read.csv(data_filename, sep = ";", fileEncoding = "UTF-8") %>%
      reshape_data() %>%
      ensure_factoring() %>%
      center_data()
    
    df$source <- source_label     # Add source column to the dataframe
    
    
    # === Mixed-Effects Model ===
    # Fit the mixed-effects model:
    #   Fixed effects: conditional probability judgments (Assume A, how probable is B) & relation type (pos, neg, irr) & judgment metric (prob/acc)
    #   Random effects: instance id (participant ID/model prompt cycle) & scenario number
    mixed_effects_model <- lmer(judgement ~ c_prob*relation_type*metric +
                                  (1 | instance_id) + # (relation_type*c_prob | instance_id)
                                  (1 | scenario_number),
                                data = df, control = lmerControl(optimizer = "bobyqa"))   # Bobyqa is a pretty stable optimizer for complex models
    
    # Perform a singularity check
    if (isSingular(mixed_effects_model)) message("Model is singular for: ", source_label)
    
    
    # === Post-hoc Analysis ===
    
    # Type III ANOVA: effect of the fixed effects on judgments as well as their interactions
    anova_type3 <- car::Anova(mixed_effects_model, type = "III", test = "F")
    
    # Slopes (trends) and estimated marginal means (EMMs)
    interaction_trends <- emtrends(mixed_effects_model, "relation_type", var = "c_prob")
    relation_type_main_effect <- emmeans(mixed_effects_model, "relation_type", by = "c_prob", at = list(c_prob = c(-0.5,  0, 0.5)))
    
    # Save everything for future reuse
    save(df, mixed_effects_model, anova_type3, interaction_trends, relation_type_main_effect, source_label,
         file = output_filename)
    
    # If the model file already exists, load it
  } else {
    load(output_filename)
    message(paste("Loaded saved analysis for: ", source_label))
  }
  
  # Return analysis objects for inspection
  list(data = df, model = mixed_effects_model, anova = anova_type3,
       trends = interaction_trends,
       emmeans = relation_type_main_effect)
  
}


# Call the individual analysis function for all dataframes (humans and LLM variants)
# And save the function output in variables for reuse
#   This either instantiates the analysis files (which might take some time and need substantial RAM)
#   Or loads them if they already exist

human_analysis <- run_individual_analysis("dataframe_human.csv", "Human", "lmm_Human.rda")

llama3_vanilla_analysis <- run_individual_analysis("dataframe_llama3_context_vanilla.csv", "Llama 8B (vanilla)", "lmm_Llama 8B (vanilla).rda")
llama3_fewshot_analysis <- run_individual_analysis("dataframe_llama3_context_fewshot.csv", "Llama 8B (few-shot)", "lmm_Llama 8B (few-shot).rda")

qwen2_vanilla_analysis <- run_individual_analysis("dataframe_qwen2_context_vanilla.csv", "Qwen 7B (vanilla)", "lmm_Qwen 7B (vanilla).rda")
qwen2_fewshot_analysis <- run_individual_analysis("dataframe_qwen2_context_fewshot.csv", "Qwen 7B (few-shot)", "lmm_Qwen 7B (few-shot).rda")



llama70b_vanilla_analysis <- run_individual_analysis("dataframe_llama70b_context_vanilla.csv", "Llama 70B (vanilla)", "lmm_Llama 70B (vanilla).rda")
llama70b_fewshot_analysis <- run_individual_analysis("dataframe_llama70b_context_fewshot.csv", "Llama 70B (few-shot)", "lmm_Llama 70B (few-shot).rda")
llama70b_cot_analysis <- run_individual_analysis("dataframe_llama70b_context_cot.csv", "Llama 70B (CoT)", "lmm_Llama 70B (CoT).rda")

qwen72b_vanilla_analysis <- run_individual_analysis("dataframe_qwen72b_context_vanilla.csv", "Qwen 72B (vanilla)", "lmm_Qwen 72B (vanilla).rda")
qwen72b_fewshot_analysis <- run_individual_analysis("dataframe_qwen72b_context_fewshot.csv", "Qwen 72B (few-shot)", "lmm_Qwen 72B (few-shot).rda")
qwen72b_cot_analysis <- run_individual_analysis("dataframe_qwen72b_context_cot.csv", "Qwen 72B (CoT)", "lmm_Qwen 72B (CoT).rda")




# Add others ...
  # Format:
  # Input filenames: .csv
  # Output filenames: ("lmm_", source_label, ".rda")  -> this is important because later analysis expects this format



# -- 1.3: Inspect Model Output --

# Function to inspect model results for one data source (respondent)
# Prints descriptive statistics, ANOVA, slopes, EMMs, and pairwise contrasts

# Argument: analysis, which stores the dataframe, linear mixed-effects model, ANOVA, slopes, and EMMs from 1.2

inspect_model_output <- function(analysis) {
  
  # Unpack the analysis object
  df <- analysis$data
  mixed_effects_model <- analysis$model
  anova_type3 <- analysis$anova
  interaction_trends <- analysis$trends
  relation_type_main_effect <- analysis$emmeans
  
  # Descriptive statistics grouped by relation type and metric
  cat("Descriptive statistics:\n")
  print(describeBy(df$judgement, group = list(df$relation_type, df$metric), mat = TRUE))
  
  # Descriptive statistics grouped by relation type and metric
  cat("Descriptive statistics (not grouped by metric):\n")
  print(describeBy(df$judgement, group = list(df$relation_type), mat = TRUE))
  
  # Response count per scenario
  cat("\nScenario-level response counts:\n")
  counts <- describeBy(df$judgement, group = list(df$scenario_number, df$relation_type, df$metric), mat = TRUE)
  print(describe(counts$n))
  
  # Type III ANOVA (main effects and interactions)
  cat("\nType III ANOVA:\n")
  print(anova_type3)
  
  # Print fixed effect estimates of conditional probability, relation type and metric
  
  cat("\nFixed effect of conditional probability:\n")
  print(fixef(mixed_effects_model)["c_prob"])
  
  cat("\nFixed effects of relation type:\n")
  print(fixef(mixed_effects_model)[grepl("^relation_type", names(fixef(mixed_effects_model)))])
  
  cat("\nFixed effects of metric:\n")
  print(fixef(mixed_effects_model)[grepl("^metric", names(fixef(mixed_effects_model)))])
  
  
  # Trends (slopes of conditional probability by relation type) and pairwise comparisons between them
  cat("\nInteraction trends (slope of c_prob by relation type):\n")
  print(interaction_trends)
  
  cat("\nPairwise comparisons of interaction slopes:\n")
  print(pairs(interaction_trends, adjust = "holm"))
  
  
  # Estimated marginal means by conditional probability and relation type as well as pairwise comparisons between them
  cat("\nEstimated marginal means:\n")
  print(relation_type_main_effect)
  
  cat("\nMarginal means summary (collapsed across c_prob):\n")
  model_emm_relation_type <- summary(update(relation_type_main_effect, by = NULL))
  print(model_emm_relation_type)
  
  cat("\nPairwise comparisons of relation types:\n")
  model_effect_relation_type <- pairs(relation_type_main_effect)
  model_effect_relation_type <- summary(update(model_effect_relation_type, by = NULL, adjust = "holm"))
  print(model_effect_relation_type)
}


# Call the function for all respondents
cat("\n Vanilla: \n")
inspect_model_output(llama3_vanilla_analysis)
inspect_model_output(qwen2_vanilla_analysis)
inspect_model_output(llama70b_vanilla_analysis)
inspect_model_output(qwen72b_vanilla_analysis)

 
cat("\n Fewshot: \n")
inspect_model_output(llama3_fewshot_analysis)
inspect_model_output(qwen2_fewshot_analysis)
inspect_model_output(llama70b_fewshot_analysis)
inspect_model_output(qwen72b_fewshot_analysis)

cat("\n CoT: \n")
inspect_model_output(llama70b_cot_analysis)
inspect_model_output(qwen72b_cot_analysis)


cat("\n Human: \n")
inspect_model_output(human_analysis)



# ================================================
# Part 2: Combined Comparison of All Respondents
# ================================================


# Load saved model data from Part 1 and bind into one dataframe
# This allows direct statistical comparison of data sources (respondents)

sources <- c("Human", "Llama 8B (vanilla)", "Llama 70B (vanilla)", "Qwen 7B (vanilla)", "Qwen 72B (vanilla)", "Llama 8B (few-shot)", "Llama 70B (few-shot)", "Qwen 7B (few-shot)", "Qwen 72B (few-shot)", "Llama 70B (CoT)", "Qwen 72B (CoT)")

filenames <- paste0("lmm_", sources, ".rda")

all_data <- list()
for (i in seq_along(sources)) {   # For each source in sources
  load(filenames[i])              # Load their file into environment
  df$source <- sources[i]         # Tag their dataframe rows with source name
  all_data[[i]] <- df             # Add the dataframe to list
}

# Combine all dataframes that are stored in the list into one
combined_data <- bind_rows(all_data)
# Order the sources (to make sure that vanilla always appears before few-shot in summary tables and plots)
combined_data$source <- factor(combined_data$source, levels = c("Human", "Llama 8B (vanilla)", "Llama 70B (vanilla)", "Qwen 7B (vanilla)", "Qwen 72B (vanilla)", "Llama 8B (few-shot)", "Llama 70B (few-shot)", "Qwen 7B (few-shot)", "Qwen 72B (few-shot)", "Llama 70B (CoT)", "Qwen 72B (CoT)"))


# -- 2.1: Compare conditional probability estimates across data sources --

# Do different respondents assign different baseline probabilities?

# === Mixed-Effects Model ===
# Fit the mixed-effects model:
#   Fixed effects: source (LLMs and humans) & relation type
#   Random effects: instance id (participant ID/model prompt cycle) & scenario number
c_prob_comparison_model <- lmer(c_prob ~ source * relation_type +
                                  (1 | instance_id) +
                                  (1 | scenario_number),
                                combined_data,
                                control = lmerControl(optimizer = "bobyqa"))

summary(c_prob_comparison_model)

# Check differences in conditional probability by source within each relation type via EMMs
emm_cprob <- emmeans(c_prob_comparison_model, ~ source | relation_type)
summary(pairs(emm_cprob, adjust = "holm"))



# -- 2.2: Comparison of judgments across data sources --

# Only compute this if the file does not exist yet
# As this model is quite big and effects are complex, this might take a while
# And we don't want to recompute unnecessarily
if (!file.exists("lmm_comparison_results.rda")) {
  message("Running analysis")

  # Center conditional probability across the combined dataset
  combined_data$c_prob <- scale(combined_data$c_prob, center = TRUE, scale = FALSE)

  
  # === Comparison Mixed-Effects Model ===
  #   Fixed effects: conditional probability judgments & relation type & source (LLMs and humans)
  #   Random effects: instance ID (participant ID/model prompt cycle) & scenario number
  comparison_model <- lmer(judgement ~ c_prob * relation_type * source +
                           (1 | instance_id) +
                           (1 | scenario_number),
                         combined_data,
                         control = lmerControl(optimizer = "bobyqa"))
  
  
  # Estimated marginal means: comparisons at specific conditional probability levels (-0.5, 0.0, 0.5)
  emm <- emmeans(comparison_model, ~ source | c_prob * relation_type, at = list(c_prob = c(-0.5, 0.0, 0.5)), infer = c(TRUE, TRUE))
  
  # Save the dataset, linear mixed-effects model, and EMMs
  save(combined_data, comparison_model, emm, file = "lmm_comparison_results.rda")

} else {
  load("lmm_comparison_results.rda")
  message("Loaded saved analysis")}

# Print pairwise contrasts (judgment comparisons at each conditional probability × relation_type × source level)
pairs (emm, adjust = "holm")

emm_df <- as.data.frame(emm)

emm_df



# ================================================
# Part 3: Plotting
# ================================================


# -- 3.1: Preparation --

# Set custom colors for each respondent
model_colors <- c("Llama 8B (vanilla)" = "#F8766D", "Llama 70B (vanilla)" = "#53B400",
                  "Qwen 7B (vanilla)" = "#00ABFD", "Qwen 72B (vanilla)" = "#A58AFF",
                  "Llama 8B (few-shot)" = "#F37B59", "Llama 70B (few-shot)" = "#39B600",
                  "Qwen 7B (few-shot)" = "#00A5FF", "Qwen 72B (few-shot)" = "#BF80FF",
                  "Llama 70B (CoT)" = "#39B600", "Qwen 72B (CoT)" = "#BF80FF",
                  "Human" = "#FF63B6") #"#FB61D7"


# Define different subsets (to use in plotting functions)
llm_subset <- c("Llama 8B (vanilla)", "Llama 70B (vanilla)", "Qwen 7B (vanilla)", "Qwen 72B (vanilla)", "Llama 8B (few-shot)", "Llama 70B (few-shot)", "Qwen 7B (few-shot)", "Qwen 72B (few-shot)", "Llama 70B (CoT)", "Qwen 72B (CoT)")

vanilla_subset <- c("Llama 8B (vanilla)", "Llama 70B (vanilla)", "Qwen 7B (vanilla)", "Qwen 72B (vanilla)")

fewshot_subset <- c("Llama 8B (few-shot)", "Llama 70B (few-shot)", "Qwen 7B (few-shot)", "Qwen 72B (few-shot)")

cot_subset <- c("Llama 70B (CoT)", "Qwen 72B (CoT)")





# -- 3.2: Individual Plots --


# Scatterplot with fixed-effect lines for individual models

plot_individual_scatterplot <- function(analysis, source, png_filename) {
  # Unpack the analysis variable
  df <- analysis$data
  mixed_effects_model <- analysis$model
  df$source <- source
  
  # Compute fixed effect predictions over a range of conditional probability values
  fe_trends <- Effect(c("c_prob", "relation_type", "metric"), mixed_effects_model,
                      xlevels = list(c_prob=seq(-0.51, 0.51, length.out = 6)), KR = TRUE)
  fe_trends_df <- as.data.frame(fe_trends)
  
  # Add source column to trend data
  fe_trends_df$source <- source
  
  
  # Plot raw points and model-predicted trends with confidence intervals
  p <- ggplot() +
    geom_point(data = df, aes(x = c_prob, y = judgement), alpha = 0.2, size = 1) +      # Plot datapoints
    geom_ribbon(data = fe_trends_df, aes(x = c_prob, ymin = lower, ymax = upper, fill = source), alpha = 0.2) +    # Plot confidence ribbon (-> uncertainty)
    geom_line(data = fe_trends_df, aes(x = c_prob, y = fit, colour = source), linewidth = 1) +    # Plot fixed-effect lines
    facet_grid(relation_type ~ metric) +                                                          # Faceting: split the data by conditions of interest
    scale_colour_manual(values = model_colors) +    # Apply custom source colours
    scale_fill_manual(values = model_colors) +
    coord_cartesian(ylim = c(-0.5, 0.5)) +      # Set fixed y-axis range, so that all plots always show the full range (even if their datapoints may not spread as far)
    labs (x = "P(B|A)", y = "Judgment") +   # Set x- and y-axis names
    theme_minimal(base_size = 14) +         # Apply minimal theme (applied to all subsequent plots, makes them look more modern than the default)
    theme(legend.position = "none")         # Suppress the legend, as axes are already labeled
  
  # Save plot to file
  ggsave(png_filename, p, width = 19, height = 16, units = "cm", dpi = 1000)
  message("Done with plotting: ", png_filename)
}

# Run scatterplot function for all sources
# plot_individual_scatterplot(human_analysis, "Human", "scatterplot_individual_Human.png")

# plot_individual_scatterplot(llama3_vanilla_analysis, "Llama 8B (vanilla)", "scatterplot_individual_Llama_vanilla.png")
# plot_individual_scatterplot(llama3_fewshot_analysis, "Llama 8B (few-shot)", "scatterplot_individual_Llama_few-shot.png")

# plot_individual_scatterplot(qwen2_vanilla_analysis, "Qwen 7B (vanilla)", "scatterplot_individual_Qwen_vanilla.png")
# plot_individual_scatterplot(qwen2_fewshot_analysis, "Qwen 7B (few-shot)", "scatterplot_individual_Qwen_few-shot.png")


# plot_individual_scatterplot(llama70b_vanilla_analysis, "Llama 70B (vanilla)", "scatterplot_individual_Llama70b_vanilla.png")
# plot_individual_scatterplot(llama70b_fewshot_analysis, "Llama 70B (few-shot)", "scatterplot_individual_Llama70b_few-shot.png")
# plot_individual_scatterplot(llama70b_cot_analysis, "Llama 70B (CoT)", "scatterplot_individual_Llama70b_cot.png")

plot_individual_scatterplot(qwen72b_vanilla_analysis, "Qwen 72B (vanilla)", "scatterplot_individual_Qwen72b_vanilla.png")
# plot_individual_scatterplot(qwen72b_fewshot_analysis, "Qwen 72B (few-shot)", "scatterplot_individual_Qwen72b_few-shot.png")
# plot_individual_scatterplot(qwen72b_cot_analysis, "Qwen 72B (CoT)", "scatterplot_individual_Qwen72b_cot.png")



# Histograms comparing acceptability vs. probability judgments

plot_metric_histogram <- function(analysis, png_filename) {
  df <- analysis$data
  metric_colors <- c("A(If A, then B)" = "#FFA07A", "P(If A, then B)" = "#A3BFD9") # Set colors that are unlikely to be confused with the source colours
  
  p <- ggplot(data = df, aes(x = judgement, fill = metric)) +         # Plot the distribution of judgements differentiated by metric
    geom_histogram(position = "identity", alpha = 0.6, bins = 10) +   # Plot histogram with 10 bins
    facet_wrap(~ relation_type) +            # Split the histograms by relation type
    scale_fill_manual(values = metric_colors) +     # Apply fixed colours
    coord_cartesian(xlim = c(-0.6, 0.6), ylim = c(0, 175)) +  # Set x- and y-axis
    labs (x = "Judgment", y = "Count", fill = "Metric") +
    theme_minimal(base_size = 14) +
    theme (
      legend.position = c(0.99, 0.99),                   # Legend inside the plot, at the top right corner, without a title
      legend.justification = c("right", "top"),
      legend.title = element_blank())
  
  ggsave(png_filename, p, width = 19, height = 16, units = "cm", dpi = 1000)
  message("Done with plotting: ", png_filename)
}

# Run metric histogram for each model and human data
# plot_metric_histogram(human_analysis, "histogram_metric_Human.png")

# plot_metric_histogram(llama3_vanilla_analysis, "histogram_metric_Llama_vanilla.png")
# plot_metric_histogram(llama3_fewshot_analysis, "histogram_metric_Llama_few-shot.png")

# plot_metric_histogram(qwen2_vanilla_analysis, "histogram_metric_Qwen_vanilla.png")
# plot_metric_histogram(qwen2_fewshot_analysis, "histogram_metric_Qwen_few-shot.png")




# Fixed effect estimates as grouped bar plot
# Arguments est1-est6: numerical values that are then matched to the individual fixed effects

plot_fixed_effects_barplot <- function(est1, est2, est3, est4, est5, est6, source, png_filename) {
  fe_df <- data.frame(
    Effect = c(                          # Set fixed effects names (to be displayed in the plot)
      "Conditional probability",
      "Relation: Positive (vs. Irrelevant)",
      "Relation: Negative (vs. Irrelevant)",
      "Metric: Probability (vs. Acceptability)",
      "Relation × Metric: Positive",
      "Relation × Metric: Negative"
    ),
    Estimate = c(est1, est2, est3, est4, est5, est6),    # Store numerical estimate values
    Type = c("Main Effect", "Main Effect", "Main Effect", "Main Effect", "Interaction", "Interaction")     # Set type of fixed effect so that they can be displayed in different colours
  )
  
  
  p <- ggplot(fe_df, aes(x = Effect, y = Estimate, fill = Type)) +            # x-axis displays the different effects on top of each other, y-axis their length, colour (fill) their type
    geom_bar(stat = "identity", position = position_dodge(width = 0.8)) +
    geom_text(aes(y = Estimate / 2, label = Effect)) +    # Display the effect labels inside the bars
    coord_flip() +   # Flip axes for better readability
    ylim(-0.5, 0.5) +
    theme_minimal(base_size = 14) +
    scale_fill_manual(values = c("Main Effect" = "#A3BFD9", "Interaction" = "#FFA07A")) +     # Set fixed colours to not be confused with source colours
    ylab(paste("Fixed Effect Estimate (", source, ")", sep = "")) +
    xlab("") +
    theme(
      legend.position = "top",          # Display the legend at the top; suppress title, side labels, and ticks
      legend.title = element_blank(),
      axis.text.y = element_blank(),
      axis.ticks.y = element_blank(),
    )

  ggsave(png_filename, p, width = 19, height = 16, units = "cm", dpi = 1000)
  message("Done with plotting: ", png_filename)
}

# Run fixed-effect barplot for each model and humans
# plot_fixed_effects_barplot(0.39, 0.35, 0.10, 0.03, -0.05, -0.04, "Human", "barplot_fixedeffects_Human.png")

# plot_fixed_effects_barplot(0.35, 0.30, 0.10, -0.06, -0.05, 0.08, "Llama 8B (vanilla)", "barplot_fixedeffects_Llama_vanilla.png")
# plot_fixed_effects_barplot(0.30, 0.33, 0.23, 0.12, -0.15, -0.14, "Llama 8B (few-shot)", "barplot_fixedeffects_Llama_few-shot.png")


# plot_fixed_effects_barplot(0.38, 0.26, 0.06, 0.05, -0.03, 0.02, "Qwen 7B (vanilla)", "barplot_fixedeffects_Qwen_vanilla.png")
# plot_fixed_effects_barplot(0.49, 0.23, 0.13, 0.05, 0.01, 0.10, "Qwen 7B (few-shot)", "barplot_fixedeffects_Qwen_few-shot.png")




# -- 3.3: Comparison Plots --


# Histogram of judgments from either only LLMs or humans (data overview)
plot_overview_histogram <- function(sources, filename) {
  
  # Filter data to include only the specified subset
  plot_data <- subset(combined_data, source %in% sources)
  plot_data$source <- factor(plot_data$source, levels = sources)
  
  p <- ggplot(data = plot_data, aes(x = judgement, fill = source)) +        # Plot the distribution of judgements
    geom_histogram(position = "identity", alpha = 0.6, binwidth = 0.05) +   # Plot histogram
    facet_wrap(~ relation_type) +            # Split the histograms by relation type  # comment this to have the overall distribution
    scale_fill_manual(values = model_colors) +
    scale_x_continuous(breaks = seq(-0.5, 0.5, by = 0.5)) +            # only needed for reltypes
    coord_cartesian(xlim = c(-0.6, 0.6), ylim = c(0, 400)) +                    # 400 for reltypes, 900 for overall
    labs(x = "Judgment", y = "Count", fill = "Source") +
    theme_minimal(base_size = 25) +
    theme(
      legend.position = c(0.99, 0.99),    # Legend inside the plot (top right corner), without a title
      legend.justification = c("right", "top"),
      legend.title = element_blank()
    )
  
  ggsave(filename, p, width = 19, height = 16, units = "cm", dpi = 1000)
  message("Done with plotting: ", filename)
}

# Call the function on the LLM subsets and on human data
# plot_overview_histogram(llm_subset, "histogram_overview_llms.png")
plot_overview_histogram(c("Human"), "histogram_overview_human_reltypes.png")    # add: overall or reltypes
plot_overview_histogram(c("Llama 70B (vanilla)"), "histogram_overview_llama70bvanilla_reltypes.png")
plot_overview_histogram(c("Qwen 72B (vanilla)"), "histogram_overview_qwen72bvanilla_reltypes.png")

# plot_overview_histogram(c("Llama 8B (few-shot)", "Qwen 7B (few-shot)"), "histogram_overview_fewshot.png")
# plot_overview_histogram(c("Llama 70B (vanilla)", "Qwen 72B (vanilla)"), "histogram_overview_vanilla.png")

# plot_overview_histogram(vanilla_subset, "histogram_overview_vanilla.png")




# Mean and standard deviation: point-and-errorbar plot

plot_mean_sd <- function(sources, filename) {
  # Compute summary stats by source and relation
  stat_df <- describeBy(combined_data$judgement, 
                        group = list(combined_data$source, combined_data$relation_type), 
                        mat = TRUE)
  
  # Rename columns for clarity (group1, group2 are the R defaults)
  stat_df$source <- stat_df$group1
  stat_df$relation_type <- stat_df$group2
  
  # Filter data to only include the specified subset
  stat_df <- subset(stat_df, source %in% sources)
  stat_df$source <- factor(stat_df$source, levels = sources)
  
  # Positioning (control spacing between the relation types)
  pd = position_dodge(width = 0.2)
  
  # Plot
  p <- ggplot(data = stat_df, aes(x = source, y = mean, color = source)) +
    geom_point(shape = 15, size = 4, position = pd) +
    geom_errorbar(aes(ymin = mean - sd, ymax = mean + sd), width = 0.2, size = 0.7, position = pd) +    # Calculate errorbars by subtracting/adding the standard deviation from/to the mean
    facet_wrap(~ relation_type, strip.position = "bottom") +    # Separate by relation type and position the names (POS, NEG, IRR) at the bottom
    scale_color_manual(values = model_colors) +
    theme_minimal(base_size = 20) +
    theme(
      legend.title = element_blank(),         # Increase space between facets; suppress x-axis title, text and ticks; show legend on top
      legend.text = element_text(size = 10),
      panel.spacing = unit(2, "lines"),
      axis.title.x = element_blank(),
      axis.text.x = element_blank(),
      axis.ticks.x = element_blank(),
      legend.position = "top"
    ) +
    ylab("Mean judgment") +
    xlab(NULL)
  
  # Save
  ggsave(filename, p, width = 19, height = 16, units = "cm", dpi = 1000)
  message("Done with plotting: ", filename)
}

# Call the function on the LLM subset and on human data
plot_mean_sd(vanilla_subset, "mean_sd_overview_vanilla_llms.png")

# plot_mean_sd(llm_subset, "mean_sd_overview_llms.png")
# plot_mean_sd(c("Human"), "mean_sd_overview_human.png")

# plot_mean_sd(c("Llama 8B (vanilla)", "Llama 8B (few-shot)", "Qwen 7B (vanilla)", "Qwen 7B (few-shot)", "Human"), "mean_sd_overview_all.png")




# Comparison of conditional probability across sources: boxplot that shows data spread and mean

p <- ggplot(combined_data, aes(x = source, y = c_prob, fill = source)) +    # Coloured boxes, next to each other; y-axis: conditional probability
  geom_boxplot() +         # Boxplot
  theme_minimal(base_size = 14) +
  labs(x = "Source", y = "Conditional Probability") + #theme(legend.position = "none") +   # No legend
  scale_color_manual(values = model_colors) +
  scale_fill_manual(values = model_colors) +
  theme(axis.title.x = element_blank())   # No title

ggsave("CondProb_comparison.png", p, width = 19, height = 16, units = "cm", dpi = 1000)
message("Done with plotting CondProb_comparison")



# Datapoints and trendlines: scatterplot
plot_scatter <- function(sources, filename) {
  # Filter data to only include vanilla LLMs (this is for the data points)
  plot_data <- subset(combined_data, source %in% sources)
  plot_data$source <- factor(plot_data$source, levels = sources)
  
  # Get fixed-effect trends from the linear model and convert results to dataframe (this is for the trendlines and confidence ribbons)
  eff <- Effect(c("c_prob", "relation_type", "source"), comparison_model,
                xlevels = list(c_prob = seq(-0.51, 0.51, length.out = 6)), KR = TRUE)
  eff_df <- as.data.frame(eff)
  eff_df <- subset(eff_df, source %in% sources)
  eff_df$source <- factor(eff_df$source, levels = sources)
  
  # Plot raw data and predicted trends
  p <- ggplot() +
    geom_point(data = plot_data, aes(x = c_prob, y = judgement), alpha = 0.15, size = 0.8) +      # Plot data points
    geom_line(data = eff_df, aes(x = c_prob, y = fit, group = interaction(source), colour = source), size = 1) +    # Plot fixed-effect lines
    geom_ribbon(data = eff_df, aes(x = c_prob, ymin = lower, ymax = upper, fill = source), alpha = 0.2) +   # Plot confidence ribbon
    facet_grid(relation_type ~ source) +              # Split plots by relation type and source facet_grid(relation_type ~ source)
    labs (x = "P(B|A)", y = "Judgment", colour = "Source", fill = "Source") +
    scale_color_manual(values = model_colors) +
    scale_fill_manual(values = model_colors) +
    scale_x_continuous(breaks = seq(-0.5, 0.5, by = 0.5)) +
    scale_y_continuous(breaks = seq(-0.5, 0.5, by = 0.5)) +
    theme_minimal(base_size = 14) +
    theme (legend.position = "none")
  
  ggsave(filename, p, width = 19, height = 16, units = "cm", dpi = 1000)
  message("Done with plotting", filename)
}


plot_scatter(vanilla_subset, "Scatterplot_Vanilla.png")
plot_scatter(c("Llama 70B (vanilla)", "Qwen 72B (vanilla)"), "Scatterplot_Qwen_Llama_big_vanilla.png")
#plot_scatter()





# ---------------------


# All vanilla LLMs in one: scatterplot
# Filter data (this is for the data points)
plot_data <- subset(combined_data, source %in% vanilla_subset)
plot_data$source <- factor(plot_data$source, levels = vanilla_subset)

# Get fixed-effect trends from the linear model and convert results to dataframe (this is for the trendlines and confidence ribbons)
eff <- Effect(c("c_prob", "relation_type", "source"), comparison_model,
              xlevels = list(c_prob = seq(-0.51, 0.51, length.out = 6)), KR = TRUE)
eff_df <- as.data.frame(eff)
eff_df <- subset(eff_df, source %in% vanilla_subset)
eff_df$source <- factor(eff_df$source, levels = vanilla_subset)

# Plot raw data and predicted trends
p <- ggplot() +
  geom_point(data = plot_data, aes(x = c_prob, y = judgement), alpha = 0.15, size = 0.8) +      # Plot data points
  geom_line(data = eff_df, aes(x = c_prob, y = fit, group = interaction(source), colour = source), size = 1) +    # Plot fixed-effect lines
  geom_ribbon(data = eff_df, aes(x = c_prob, ymin = lower, ymax = upper, fill = source), alpha = 0.2) +   # Plot confidence ribbon
  facet_grid(~ relation_type) +              # Split plots by relation type and source facet_grid(relation_type ~ source)
  labs (x = "P(B|A)", y = "Judgment", colour = "Source", fill = "Source") +
  scale_color_manual(values = model_colors) +
  scale_fill_manual(values = model_colors) +
  scale_x_continuous(breaks = seq(-0.5, 0.5, by = 0.5)) +
  scale_y_continuous(breaks = seq(-0.5, 0.5, by = 0.5)) +
  theme_minimal(base_size = 14) +
  theme (legend.position = "none")

ggsave("Scatterplot_Vanilla_all_in_one.png", p, width = 19, height = 8, units = "cm", dpi = 1000)
message("Done with plotting Scatterplot_Vanilla_all_in_one")