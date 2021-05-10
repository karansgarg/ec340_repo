# FILENAME: lstm_log_diff.R
# Used for tuning the hyperparameters on the LSTM model being trained to predict HSBC stock returns

library(keras)

FLAGS <- flags(flag_numeric("dropout", 0.2),
               flag_integer("units", 1))


CreateLSTM <- function(){
  lstm <- keras_model_sequential() %>%
    layer_lstm(units=FLAGS$units,
               activation="tanh",
               return_sequences=FALSE,
               input_shape = dim(X_log_diff_train_lstm)[-1]) %>%
    layer_dropout(FLAGS$dropout) %>%
    layer_dense(units=1)
  
  lstm %>% compile(optimizer="nadam",
                   loss="mean_squared_error")
  
  return(lstm)
}

lstm_log_diff <- CreateLSTM()

lstm_log_diff_fit <- lstm %>% fit(x=X_log_diff_train_lstm,
                         y=y_log_diff_train,
                         epochs=100,
                         batch_size=32,
                         validation_data=val_log_diff_list,
                         shuffle=FALSE,
                         verbose=0)

acc <- lstm %>% evaluate(x=X_log_diff_test_lstm,
                         y=y_log_diff_test,
                         verbose=0)

save_model_hdf5(lstm_log_diff, 'lstm_log_diff.h5')