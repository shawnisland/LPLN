#
install.packages("dplyr")
install.packages("mice") # 
library(Boruta)
library(dplyr)
library(mice) # 
#?Boruta

set.seed(123)
data <- read.csv(file.choose())


boruta_output <- Boruta(state ~ Sex+Age,
                      data, pValue = 0.01, mcAdj = FALSE, maxRuns = 1000, doTrace = 1)
summary(boruta_output)

importance <- boruta_output$ImpHistory  #
importance
#write.csv(importance, 'Variable_Importance.csv', row.names = FALSE)

plot(boruta_output, las = 1, xlab = '', main = 'Variable Importance',cex.axis = 0.7)

