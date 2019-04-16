library(MASS)
library(tidyverse)
library(tokenizers)
library(tidytext)
library(caret)


# Add pronouns to nrc lexicon ---------------------------------------------
nrc <- get_sentiments("nrc")
pronouns <- tibble(word = pos_df_pronouns$pronoun, sentiment = pos_df_pronouns$point_of_view)
XXXX <- tibble(word = "xxxx", sentiment = "XXXX")
new_nrc <- bind_rows(nrc, pronouns, XXXX)


# extract nrc features of all transcripts ---------------------------------
transcript_features_nrc <- 
  transcripts %>%
  unnest_tokens(token, vlog, token = 'words') %>%
  inner_join(new_nrc, by = c(token = 'word')) %>%
  count(Id, `sentiment`) %>%
  spread(sentiment, n, fill = 0)


# extract proportions of nrc features used and add total count ------------
proportions_nrc <- transcript_features_nrc %>% 
  mutate(sumVar = rowSums(.[,2:15])) %>% 
  mutate_at(vars(XXXX:trust), funs(./ sumVar))


# compute weighted sentiment sums with afinn scores -----------------------
# extract nrc features
features_nrc  <- 
  transcripts %>%
  unnest_tokens(token, vlog, token = 'words') %>%
  inner_join(get_sentiments('nrc'), by = c(token = 'word')) 

# extract afinn features
features_afinn <- 
  transcripts %>%
  unnest_tokens(token, vlog, token = 'words') %>%
  inner_join(get_sentiments('afinn'), by = c(token = 'word'))

# calculate the sum of afinn scores per feature
features_scored <- 
  left_join(features_nrc, features_afinn) %>% 
  group_by(Id, sentiment) %>% 
  summarise(sums = sum(score, na.rm = TRUE)) %>% 
  spread(sentiment, sums, fill = 0)


# join new computed variables with train and test set ---------------------
train_joined <- train_set %>% 
  inner_join(proportions_nrc, by = c('vlogId' = 'Id')) %>% 
  inner_join(features_scored, by = c('vlogId' = 'Id'))

test_joined <- test_set %>% 
  inner_join(proportions_nrc, by = c('vlogId' = 'Id')) %>% 
  inner_join(features_scored, by = c('vlogId' = 'Id'))

# recode gender to factor
train_joined <- train_joined %>% 
  mutate(gender = as.factor(gender))

test_joined <- test_joined %>% 
  mutate(gender = as.factor(gender))
####################

variables = train_joined %>% dplyr::select(-c(1:6, gender))

correlations <- cor(variables)

corrplot(correlations, method = 'circle')

eigenvalues <- eigen(correlations)

data_ext <- train_joined %>% dplyr::select(-c(vlogId, Agr:Open, gender))






VARIABLES_TO_USE <- data_ext %>% select(-1) %>% select(index)

# Remove outliers!

findOutlier <- function(matrix, cutoff = 3) {
  ## Calculate the sd
  sds <- apply(matrix, 2, sd, na.rm = TRUE)
  ## Identify the cells with value greater than cutoff * sd (column wise)
  result <- mapply(function(d, s) {
    which(abs(d - mean(d)) > cutoff * s)
  }, matrix, sds)
  result
}

outliers <- findOutlier(data_ext)
outliers_to_remove <- unlist(outliers) %>% unique()

data_ext_no_out <- data_ext %>% slice(-outliers_to_remove)



# EXPERIMENT 15 highest correlating variables

data_ext_experimental <- data_ext %>% dplyr::select(order(abs(cor(data_ext))[,1], decreasing = TRUE)[1:30])

#outliers_experimental <- findOutlier(data_ext_experimental)
#outliers_to_remove_experimental <- unlist(outliers_experimental) %>% unique()


#data_ext_final <- bind_cols(Extr = train_joined$Extr, VARIABLES_TO_USE)


data_ext_experimental_noOut <- data_ext_experimental %>% slice(-c(outliers_to_remove))

#data_ext_final_noOut <-  data_ext_final %>% slice(-c(outliers_to_remove))


control <- trainControl(method = 'repeatedcv', number = 5, repeats = 3, search = 'random')
 
model_ext <- train(Extr ~ ., data = data_ext_no_out,method = 'leapSeq',preProcess = c('center', 'scale'), trControl = control, tuneLength = 10 )

model_ext

ext_test <- test_joined %>% dplyr::select(-c(vlogId, gender))

### OTHER PERSONALITIES

# Agr 

data_Agr <- train_joined %>% dplyr::select(-c(vlogId, Extr, Cons:Open, gender))

data_Agr_exp  <- data_Agr %>% dplyr::select(order(cor(data_Agr)[,1], decreasing = TRUE)[1:15])

Agr_outliers <- findOutlier(data_Agr) %>% unlist() %>% unique()

data_Agr_exp_noout <- data_Agr_exp %>% slice(-Agr_outliers)

model_Agr <- train(Agr ~ ., data = data_Agr_exp_noout,method = 'leapBackward',preProcess = c('center', 'scale'), trControl = control, tuneLength = 10 )

# Cons 

data_Cons <- train_joined %>% dplyr::select(-c(vlogId, Extr,Agr, Emot,Open, gender))

data_Cons_exp  <- data_Cons %>% dplyr::select(order(cor(data_Cons)[,1], decreasing = TRUE)[1:15])

Cons_outliers <- findOutlier(data_Cons) %>% unlist() %>% unique()

data_Cons_exp_noout <- data_Cons_exp %>% slice(-Cons_outliers)

model_Cons <- train(Cons ~ ., data = data_Cons_exp_noout,method = 'leapBackward',preProcess = c('center', 'scale'), trControl = control, tuneLength = 10 )


# Emot

data_Emot <- train_joined %>% dplyr::select(-c(vlogId, Extr:Cons, Open, gender))

data_Emot_exp  <- data_Emot %>% dplyr::select(order(cor(data_Emot)[,1], decreasing = TRUE)[1:15])

Emot_outliers <- findOutlier(data_Emot) %>% unlist() %>% unique()

data_Emot_exp_noout <- data_Emot_exp %>% slice(-Emot_outliers)

model_Emot <- train(Emot ~ ., data = data_Emot_exp_noout,method = 'leapBackward',preProcess = c('center', 'scale'), trControl = control, tuneLength = 10 )


# Open

data_Open <- train_joined %>% dplyr::select(-c(vlogId, Extr:Emot, gender))

data_Open_exp  <- data_Open %>% dplyr::select(order(cor(data_Open)[,1], decreasing = TRUE)[1:15])

Open_outliers <- findOutlier(data_Open) %>% unlist() %>% unique()

data_Open_exp_noout <- data_Open_exp %>% slice(-Open_outliers)

model_Open <- train(Open ~ ., data = data_Open_exp_noout,method = 'leapBackward',preProcess = c('center', 'scale'), trControl = control, tuneLength = 10 )


################################################################


## PCA

pca <- data_ext %>% dplyr::select(-1) %>% psych::principal( rotate="varimax", nfactors=15, scores=TRUE)



summary(pca)

order(abs(cor(data_ext$Extr, pca$scores)), decreasing = TRUE)[1:15]


index <- numeric()

for(i in 1:6){
  index_store <- pca$rotation[,i] %>% order(decreasing = TRUE)
  
  if(any(index_store[1] %in% index)){
    
    index[i] <- setdiff(index_store, index)[1]
    
  } else{
    
    index[i] <- index_store[1]
    
  }
  print(index)
  print(index_store[1])
  
  
}























