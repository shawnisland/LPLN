#使用Boruta进行特征选择的R语言示例
install.packages("dplyr")
install.packages("mice") # 安装mice包用于缺失值填补
library(Boruta)
library(dplyr)
library(mice) # 加载mice包
#?Boruta
#详情 ?Boruta
#pValue 指定置信水平，mcAdj=TRUE 意为将应用 Bonferroni 方法校正 p 值
#此外还可以提供 mtry 和 ntree 参数的值，这些值将传递给随机森林函数 randomForest()
set.seed(123)
data <- read.csv(file.choose())


boruta_output <- Boruta(state ~ Sex+Age,
                      data, pValue = 0.01, mcAdj = FALSE, maxRuns = 1000, doTrace = 1)
#+BMI  neoadjuvant+
summary(boruta_output)

importance <- boruta_output$ImpHistory  #
importance
#write.csv(importance, 'Variable_Importance.csv', row.names = FALSE)

plot(boruta_output, las = 1, xlab = '', main = 'Variable Importance',cex.axis = 0.7)

