library(tidyverse)

run <- read.csv("testdev_base_clipbase_owllarge_open.csv")

ggplot(run_short, aes(x=runtime_sec)) + 
  geom_histogram(binwidth = 0.1, fill="#9B79AA", color="#E1D5E7") +
  xlab("Answer Similarity") +
  ylab("# Questions") + 
  theme(text = element_text(size = 13)) 
