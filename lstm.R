# FILENAME: lstm.R
# Used for tuning the hyperparameters on the LSTM model being trained to predict the HSBC stock price

library(keras)

flag_numeric("dropout", 0.2)
flag_integer("units", 1)

CreateLSTM <- function(){
  lstm <- keras_model_sequential() %>%
    layer_lstm(units=FLAGS$units,
               activation="tanh",
               return_sequences=FALSE,
               input_shape = dim(X_train_lstm)[-1]) %>%
    layer_dropout(FLAGS$dropout) %>%
    layer_dense(units=1)
  
  lstm %>% compile(optimizer="nadam",
                   loss="mean_squared_error")
  
  return(lstm)
}

lstm <- CreateLSTM()

epochs <- 10

